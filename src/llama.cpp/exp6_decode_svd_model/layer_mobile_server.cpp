#include "llama-context.h"
#include "llama-model.h"
#include "ggml-cpu.h"
#include "layer_coop_net.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 4) {
        std::cerr << "usage: " << argv[0] << " <model> <port> <split_m_layers> [threads]\n";
        return 1;
    }

    const std::string model_path = argv[1];
    const uint16_t port = (uint16_t) std::stoi(argv[2]);
    const int32_t split_m = std::stoi(argv[3]);
    const int32_t threads = argc > 4 ? std::stoi(argv[4]) : std::max(1u, std::thread::hardware_concurrency() / 2);

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

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 2048;
    cp.n_batch = 2048;
    cp.n_threads = threads;
    cp.n_threads_batch = threads;
    cp.layer_split_start = split_m;
    cp.layer_split_end = -1;

    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(
        llama_init_from_model(model.get(), cp), llama_free);
    if (!ctx) {
        std::cerr << "failed to create context\n";
        return 1;
    }

    if (!layer_coop_net_init()) {
        std::cerr << "network init failed\n";
        return 1;
    }
    const int listen_fd = (int) socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd < 0) {
        std::cerr << "socket failed\n";
        return 1;
    }
    int one = 1;
    setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, (const char *) &one, sizeof(one));
    sockaddr_in addr {};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);
    if (bind(listen_fd, (sockaddr *) &addr, sizeof(addr)) != 0 || listen(listen_fd, 1) != 0) {
        std::cerr << "bind/listen failed\n";
        return 1;
    }

    const int n_embd = llama_model_n_embd(model.get());
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
    std::cout << "layer_mobile_server listening on 0.0.0.0:" << port
              << " split_m=" << split_m
              << " tail_layers=[" << split_m << "," << llama_model_n_layer(model.get()) << ")"
              << " threads=" << threads << std::endl;

    int64_t requests = 0;
    double decode_ms_total = 0.0;
    for (;;) {
        sockaddr_in peer {};
#ifdef _WIN32
        int peer_len = sizeof(peer);
        const int fd = (int) accept((SOCKET) listen_fd, (sockaddr *) &peer, &peer_len);
#else
        socklen_t peer_len = sizeof(peer);
        const int fd = accept(listen_fd, (sockaddr *) &peer, &peer_len);
#endif
        if (fd < 0) {
            continue;
        }
        int flag = 1;
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, (const char *) &flag, sizeof(flag));

        for (;;) {
            LayerCoopRequest req {};
            if (!layer_coop_recv_all(fd, &req, sizeof(req))) break;
            if (req.magic != LAYER_COOP_MAGIC || req.version != LAYER_COOP_VERSION ||
                    req.n_embd != n_embd || req.n_vocab != n_vocab || req.n_tokens <= 0) {
                LayerCoopResponse resp { LAYER_COOP_MAGIC, LAYER_COOP_VERSION, -1, 0, 0.0 };
                layer_coop_send_all(fd, &resp, sizeof(resp));
                break;
            }

            std::vector<int32_t> pos(req.n_tokens);
            std::vector<float> hidden((size_t) req.n_tokens * n_embd);
            if (!layer_coop_recv_all(fd, pos.data(), pos.size() * sizeof(int32_t)) ||
                    !layer_coop_recv_all(fd, hidden.data(), hidden.size() * sizeof(float))) {
                break;
            }
            if (req.reset_kv) {
                llama_kv_cache_clear(ctx.get());
            }

            llama_batch batch = llama_batch_init(req.n_tokens, n_embd, 1);
            batch.n_tokens = req.n_tokens;
            std::memcpy(batch.embd, hidden.data(), hidden.size() * sizeof(float));
            for (int32_t i = 0; i < req.n_tokens; ++i) {
                batch.pos[i] = pos[i];
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i] = (i == req.n_tokens - 1);
            }

            const auto t0 = std::chrono::steady_clock::now();
            const int ret = llama_decode(ctx.get(), batch);
            const auto t1 = std::chrono::steady_clock::now();
            llama_batch_free(batch);
            const double decode_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

            const float * logits = ret == 0 ? llama_get_logits_ith(ctx.get(), -1) : nullptr;
            LayerCoopResponse resp {
                LAYER_COOP_MAGIC,
                LAYER_COOP_VERSION,
                logits ? 0 : ret,
                logits ? n_vocab : 0,
                decode_ms,
            };
            if (!layer_coop_send_all(fd, &resp, sizeof(resp))) break;
            if (logits && !layer_coop_send_all(fd, logits, (size_t) n_vocab * sizeof(float))) break;

            requests++;
            decode_ms_total += decode_ms;
            if (requests % 16 == 0) {
                std::cout << "[layer-coop-server] requests=" << requests
                          << " avg_decode_ms=" << (decode_ms_total / requests) << std::endl;
            }
        }
        layer_coop_close(fd);
    }
}
