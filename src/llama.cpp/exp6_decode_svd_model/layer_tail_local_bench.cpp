#include "llama.h"
#include "ggml-backend.h"
#include "layer_threadpool.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
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

static double ms_since(std::chrono::steady_clock::time_point t0, std::chrono::steady_clock::time_point t1) {
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

static bool tokenize_prompt(const llama_vocab * vocab, const std::string & prompt, std::vector<llama_token> & tokens) {
    tokens.resize(prompt.size() + 8);
    int32_t n_tokens = llama_tokenize(vocab, prompt.c_str(), (int32_t) prompt.size(),
            tokens.data(), (int32_t) tokens.size(), true, true);
    if (n_tokens < 0) {
        tokens.resize((size_t) -n_tokens);
        n_tokens = llama_tokenize(vocab, prompt.c_str(), (int32_t) prompt.size(),
                tokens.data(), (int32_t) tokens.size(), true, true);
    }
    if (n_tokens <= 0) {
        return false;
    }
    tokens.resize((size_t) n_tokens);
    return true;
}

static int run_full_token(llama_model * model, int32_t max_new_tokens, int32_t threads) {
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 2048;
    cp.n_batch = 2048;
    cp.n_threads = threads;
    cp.n_threads_batch = threads;
    cp.no_perf = false;

    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(
        llama_init_from_model(model, cp), llama_free);
    if (!ctx) {
        std::cerr << "failed to create full_token context\n";
        return 1;
    }
    LayerThreadpoolPtr threadpool = layer_attach_threadpool(ctx.get(), threads, "tail-local-bench");

    std::vector<llama_token> prompt_tokens;
    if (!tokenize_prompt(vocab, "Once upon a time", prompt_tokens)) {
        std::cerr << "failed to tokenize prompt\n";
        return 1;
    }

    llama_kv_cache_clear(ctx.get());
    llama_batch batch = llama_batch_init((int32_t) prompt_tokens.size(), 0, 1);
    batch.n_tokens = (int32_t) prompt_tokens.size();
    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        batch.token[i] = prompt_tokens[(size_t) i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = (i == batch.n_tokens - 1);
    }

    auto t0 = std::chrono::steady_clock::now();
    int ret = llama_decode(ctx.get(), batch);
    auto t1 = std::chrono::steady_clock::now();
    llama_batch_free(batch);
    if (ret != 0) {
        std::cerr << "full_token prompt decode failed ret=" << ret << "\n";
        return 1;
    }
    const double prefill_ms = ms_since(t0, t1);

    const float * logits = llama_get_logits_ith(ctx.get(), -1);
    llama_token next = logits ? greedy_sample(logits, n_vocab) : 0;

    double decode_ms = 0.0;
    int32_t pos = (int32_t) prompt_tokens.size();
    for (int32_t i = 0; i < max_new_tokens; ++i, ++pos) {
        llama_batch gb = llama_batch_init(1, 0, 1);
        gb.n_tokens = 1;
        gb.token[0] = next;
        gb.pos[0] = pos;
        gb.n_seq_id[0] = 1;
        gb.seq_id[0][0] = 0;
        gb.logits[0] = 1;

        t0 = std::chrono::steady_clock::now();
        ret = llama_decode(ctx.get(), gb);
        t1 = std::chrono::steady_clock::now();
        llama_batch_free(gb);
        decode_ms += ms_since(t0, t1);
        if (ret != 0) {
            std::cerr << "full_token decode failed ret=" << ret << " step=" << i << "\n";
            return 1;
        }
        logits = llama_get_logits_ith(ctx.get(), -1);
        next = logits ? greedy_sample(logits, n_vocab) : 0;
    }

    std::cout << "[tail-local-bench] mode=full_token tokens=" << max_new_tokens
              << " threads=" << threads
              << " prefill_ms=" << prefill_ms
              << " decode_ms=" << decode_ms
              << " avg_decode_ms=" << (decode_ms / std::max(1, max_new_tokens))
              << " throughput=" << (1000.0 * max_new_tokens / decode_ms)
              << " tok/s" << std::endl;
    llama_perf_context_print(ctx.get());
    return 0;
}

static int run_tail_embd(llama_model * model, int32_t max_new_tokens, int32_t threads, int32_t split_m, int32_t pos_offset) {
    const int32_t n_embd = llama_model_n_embd(model);
    const int32_t n_layer = llama_model_n_layer(model);
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 2048;
    cp.n_batch = 2048;
    cp.n_threads = threads;
    cp.n_threads_batch = threads;
    cp.layer_split_start = split_m;
    cp.layer_split_end = -1;
    cp.no_perf = false;

    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(
        llama_init_from_model(model, cp), llama_free);
    if (!ctx) {
        std::cerr << "failed to create tail_embd context\n";
        return 1;
    }
    LayerThreadpoolPtr threadpool = layer_attach_threadpool(ctx.get(), threads, "tail-local-bench");

    std::vector<float> hidden((size_t) n_embd);
    for (int32_t i = 0; i < n_embd; ++i) {
        hidden[(size_t) i] = 0.01f * std::sin(0.001f * (float) i);
    }

    llama_kv_cache_clear(ctx.get());
    double decode_ms = 0.0;
    llama_token next = 0;
    for (int32_t i = 0; i < max_new_tokens; ++i) {
        hidden[(size_t) (i % n_embd)] += 1.0e-5f * (float) ((next % 17) - 8);

        llama_batch batch = llama_batch_init(1, n_embd, 1);
        batch.n_tokens = 1;
        std::memcpy(batch.embd, hidden.data(), hidden.size() * sizeof(float));
        batch.pos[0] = pos_offset + i;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = 1;

        const auto t0 = std::chrono::steady_clock::now();
        const int ret = llama_decode(ctx.get(), batch);
        const auto t1 = std::chrono::steady_clock::now();
        llama_batch_free(batch);
        decode_ms += ms_since(t0, t1);
        if (ret != 0) {
            std::cerr << "tail_embd decode failed ret=" << ret << " step=" << i << "\n";
            return 1;
        }

        const float * logits = llama_get_logits_ith(ctx.get(), -1);
        next = logits ? greedy_sample(logits, n_vocab) : 0;
    }

    std::cout << "[tail-local-bench] mode=tail_embd tokens=" << max_new_tokens
              << " threads=" << threads
              << " split_m=" << split_m
              << " pos_offset=" << pos_offset
              << " tail_layers=[" << split_m << "," << n_layer << ")"
              << " n_embd=" << n_embd
              << " decode_ms=" << decode_ms
              << " avg_decode_ms=" << (decode_ms / std::max(1, max_new_tokens))
              << " throughput=" << (1000.0 * max_new_tokens / decode_ms)
              << " tok/s" << std::endl;
    llama_perf_context_print(ctx.get());
    return 0;
}

static int run_prefix_token(llama_model * model, int32_t max_new_tokens, int32_t threads, int32_t split_m) {
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    const int32_t n_layer = llama_model_n_layer(model);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 2048;
    cp.n_batch = 2048;
    cp.n_threads = threads;
    cp.n_threads_batch = threads;
    cp.embeddings = true;
    cp.layer_split_start = 0;
    cp.layer_split_end = split_m;
    cp.no_perf = false;

    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(
        llama_init_from_model(model, cp), llama_free);
    if (!ctx) {
        std::cerr << "failed to create prefix_token context\n";
        return 1;
    }
    LayerThreadpoolPtr threadpool = layer_attach_threadpool(ctx.get(), threads, "tail-local-bench");

    std::vector<llama_token> prompt_tokens;
    if (!tokenize_prompt(vocab, "Once upon a time", prompt_tokens)) {
        std::cerr << "failed to tokenize prompt\n";
        return 1;
    }

    llama_kv_cache_clear(ctx.get());
    llama_batch batch = llama_batch_init((int32_t) prompt_tokens.size(), 0, 1);
    batch.n_tokens = (int32_t) prompt_tokens.size();
    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        batch.token[i] = prompt_tokens[(size_t) i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = 0;
    }

    auto t0 = std::chrono::steady_clock::now();
    int ret = llama_decode(ctx.get(), batch);
    auto t1 = std::chrono::steady_clock::now();
    llama_batch_free(batch);
    if (ret != 0) {
        std::cerr << "prefix_token prompt decode failed ret=" << ret << "\n";
        return 1;
    }
    const double prefill_ms = ms_since(t0, t1);

    llama_token next = 0;
    double decode_ms = 0.0;
    int32_t pos = (int32_t) prompt_tokens.size();
    for (int32_t i = 0; i < max_new_tokens; ++i, ++pos) {
        llama_batch gb = llama_batch_init(1, 0, 1);
        gb.n_tokens = 1;
        gb.token[0] = next;
        gb.pos[0] = pos;
        gb.n_seq_id[0] = 1;
        gb.seq_id[0][0] = 0;
        gb.logits[0] = 1;

        t0 = std::chrono::steady_clock::now();
        ret = llama_decode(ctx.get(), gb);
        t1 = std::chrono::steady_clock::now();
        llama_batch_free(gb);
        decode_ms += ms_since(t0, t1);
        if (ret != 0) {
            std::cerr << "prefix_token decode failed ret=" << ret << " step=" << i << "\n";
            return 1;
        }
        const float * hidden = llama_get_embeddings_ith(ctx.get(), 0);
        if (hidden) {
            next = (llama_token) (((int32_t) std::fabs(hidden[0] * 1000.0f) + i) % n_vocab);
        }
    }

    std::cout << "[tail-local-bench] mode=prefix_token tokens=" << max_new_tokens
              << " threads=" << threads
              << " split_m=" << split_m
              << " prefix_layers=[0," << split_m << ")"
              << " n_layer=" << n_layer
              << " prefill_ms=" << prefill_ms
              << " decode_ms=" << decode_ms
              << " avg_decode_ms=" << (decode_ms / std::max(1, max_new_tokens))
              << " throughput=" << (1000.0 * max_new_tokens / decode_ms)
              << " tok/s" << std::endl;
    llama_perf_context_print(ctx.get());
    return 0;
}

int main(int argc, char ** argv) {
    if (argc < 5) {
        std::cerr << "usage: " << argv[0] << " <model> <tokens> <threads> <full_token|tail_embd|prefix_token> [split_m] [pos_offset]\n";
        return 1;
    }

    const std::string model_path = argv[1];
    const int32_t max_new_tokens = std::stoi(argv[2]);
    const int32_t threads = std::stoi(argv[3]);
    const std::string mode = argv[4];
    const int32_t split_m = argc > 5 ? std::stoi(argv[5]) : 14;
    const int32_t pos_offset = argc > 6 ? std::stoi(argv[6]) : 0;

    ggml_backend_load_all();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
#ifdef _WIN32
    mp.use_mmap = false;
#endif
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
        llama_model_load_from_file(model_path.c_str(), mp), llama_model_free);
    if (!model) {
        std::cerr << "failed to load model\n";
        return 1;
    }

    if (mode == "full_token") {
        return run_full_token(model.get(), max_new_tokens, threads);
    }
    if (mode == "tail_embd") {
        return run_tail_embd(model.get(), max_new_tokens, threads, split_m, pos_offset);
    }
    if (mode == "prefix_token") {
        return run_prefix_token(model.get(), max_new_tokens, threads, split_m);
    }

    std::cerr << "unknown mode: " << mode << "\n";
    return 1;
}
