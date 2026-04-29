#include "llama-context.h"
#include "llama-model.h"
#include "ggml-cpu.h"
#include "layer_coop_net.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

static llama_token greedy_sample(const float * logits, int32_t n_vocab) {
    int32_t best = 0;
    float best_v = logits[0];
    for (int32_t i = 1; i < n_vocab; ++i) {
        if (logits[i] > best_v) {
            best_v = logits[i];
            best = i;
        }
    }
    return (llama_token) best;
}

static std::vector<float> send_hidden(int fd, const std::vector<int32_t> & pos, const float * hidden,
        int32_t n_embd, int32_t n_vocab, bool reset_kv, double & server_ms) {
    LayerCoopRequest req {
        LAYER_COOP_MAGIC,
        LAYER_COOP_VERSION,
        (int32_t) pos.size(),
        n_embd,
        n_vocab,
        reset_kv ? 1 : 0,
    };
    const size_t hidden_count = (size_t) req.n_tokens * n_embd;
    if (!layer_coop_send_all(fd, &req, sizeof(req)) ||
            !layer_coop_send_all(fd, pos.data(), pos.size() * sizeof(int32_t)) ||
            !layer_coop_send_all(fd, hidden, hidden_count * sizeof(float))) {
        throw std::runtime_error("failed to send layer-coop request");
    }
    LayerCoopResponse resp {};
    if (!layer_coop_recv_all(fd, &resp, sizeof(resp)) ||
            resp.magic != LAYER_COOP_MAGIC ||
            resp.version != LAYER_COOP_VERSION ||
            resp.status != 0 ||
            resp.n_logits != n_vocab) {
        throw std::runtime_error("bad layer-coop response");
    }
    std::vector<float> logits(n_vocab);
    if (!layer_coop_recv_all(fd, logits.data(), logits.size() * sizeof(float))) {
        throw std::runtime_error("failed to receive logits");
    }
    server_ms += resp.server_decode_ms;
    return logits;
}

int main(int argc, char ** argv) {
    if (argc < 5) {
        std::cerr << "usage: " << argv[0] << " <model> <tokens> <threads> <host:port> <split_m_layers> [verbose]\n";
        return 1;
    }

    const std::string model_path = argv[1];
    const int32_t max_new_tokens = std::stoi(argv[2]);
    const int32_t threads = std::stoi(argv[3]);
    std::string host;
    uint16_t port = 0;
    if (!layer_coop_parse_host_port(argv[4], host, port)) {
        std::cerr << "bad endpoint\n";
        return 1;
    }
    const int32_t split_m = argc > 5 ? std::stoi(argv[5]) : 14;
    const bool verbose = argc > 6 ? std::stoi(argv[6]) != 0 : false;

    ggml_backend_load_all();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
#ifdef _WIN32
    mp.use_mmap = false;
#endif
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
        llama_model_load_from_file(model_path.c_str(), mp), llama_model_free);
    if (!model) {
        std::cerr << "failed to load model\n";
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model.get());
    const int32_t n_embd = llama_model_n_embd(model.get());
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 2048;
    cp.n_batch = 2048;
    cp.n_threads = threads;
    cp.n_threads_batch = threads;
    cp.embeddings = true;
    cp.layer_split_start = 0;
    cp.layer_split_end = split_m;

    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(
        llama_init_from_model(model.get(), cp), llama_free);
    if (!ctx) {
        std::cerr << "failed to create prefix context\n";
        return 1;
    }

    const int fd = layer_coop_connect(host, port);
    std::cout << "layer cooperative decode enabled: prefix_layers=[0," << split_m
              << ") tail_endpoint=" << host << ":" << port
              << " n_embd=" << n_embd << " n_vocab=" << n_vocab << std::endl;

    const std::string prompt = "Once upon a time";
    std::vector<llama_token> tokens(prompt.size() + 8);
    int32_t n_tokens = llama_tokenize(vocab, prompt.c_str(), (int32_t) prompt.size(),
            tokens.data(), (int32_t) tokens.size(), true, true);
    if (n_tokens <= 0) {
        std::cerr << "tokenize failed\n";
        return 1;
    }
    tokens.resize(n_tokens);

    llama_kv_cache_clear(ctx.get());
    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    batch.n_tokens = n_tokens;
    std::vector<int32_t> pos(n_tokens);
    for (int32_t i = 0; i < n_tokens; ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = 1;
        pos[i] = i;
    }

    double server_ms = 0.0;
    double prefix_ms = 0.0;
    const auto t_all0 = std::chrono::steady_clock::now();
    auto t0 = std::chrono::steady_clock::now();
    int ret = llama_decode(ctx.get(), batch);
    auto t1 = std::chrono::steady_clock::now();
    llama_batch_free(batch);
    if (ret != 0) {
        std::cerr << "prefix prefill failed\n";
        return 1;
    }
    prefix_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
    const float * hidden = llama_get_embeddings(ctx.get());
    if (!hidden) {
        std::cerr << "prefix embeddings missing\n";
        return 1;
    }
    std::vector<float> logits = send_hidden(fd, pos, hidden, n_embd, n_vocab, true, server_ms);

    std::vector<llama_token> generated;
    generated.reserve(max_new_tokens);
    int32_t total_tokens = n_tokens;
    for (int32_t gen = 0; gen < max_new_tokens; ++gen) {
        const llama_token next = greedy_sample(logits.data(), n_vocab);
        generated.push_back(next);
        if (verbose) {
            char buf[256];
            const int32_t len = llama_token_to_piece(vocab, next, buf, (int32_t) sizeof(buf), 0, true);
            std::cout << "token #" << gen << " id=" << next
                      << " piece=\"" << (len > 0 ? std::string(buf, buf + len) : "[ERR]") << "\"\n";
        }

        llama_batch gb = llama_batch_init(1, 0, 1);
        gb.n_tokens = 1;
        gb.token[0] = next;
        gb.pos[0] = total_tokens;
        gb.n_seq_id[0] = 1;
        gb.seq_id[0][0] = 0;
        gb.logits[0] = 1;
        t0 = std::chrono::steady_clock::now();
        ret = llama_decode(ctx.get(), gb);
        t1 = std::chrono::steady_clock::now();
        llama_batch_free(gb);
        if (ret != 0) {
            std::cerr << "prefix decode failed\n";
            return 1;
        }
        prefix_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        hidden = llama_get_embeddings_ith(ctx.get(), 0);
        int32_t p = total_tokens;
        logits = send_hidden(fd, std::vector<int32_t>{p}, hidden, n_embd, n_vocab, false, server_ms);
        total_tokens++;
    }
    const auto t_all1 = std::chrono::steady_clock::now();
    layer_coop_close(fd);

    std::string text;
    for (llama_token tok : generated) {
        char buf[256];
        const int32_t len = llama_token_to_piece(vocab, tok, buf, (int32_t) sizeof(buf), 0, true);
        text += len > 0 ? std::string(buf, buf + len) : "[ERR]";
    }
    const double total_ms = std::chrono::duration<double, std::milli>(t_all1 - t_all0).count();
    std::cout << "Generated text: " << text << std::endl;
    std::cout << "[layer-coop-client] generated=" << generated.size()
              << " prefix_ms=" << prefix_ms
              << " server_decode_ms=" << server_ms
              << " total_ms=" << total_ms
              << " throughput=" << (generated.empty() ? 0.0 : generated.size() / (total_ms / 1000.0))
              << " tok/s" << std::endl;
    std::cout << "SUCCEED (layer cooperative decode done)" << std::endl;
    return 0;
}
