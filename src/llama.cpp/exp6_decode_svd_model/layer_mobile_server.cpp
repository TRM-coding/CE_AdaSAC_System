#include "llama-context.h"
#include "llama-model.h"
#include "ggml-cpu.h"
#include "layer_coop_net.h"
#include "layer_threadpool.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
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

static void warmup_tail_context(llama_context * ctx, int32_t n_embd) {
    std::vector<float> hidden((size_t) n_embd, 0.0f);
    llama_batch batch = llama_batch_init(1, n_embd, 1);
    batch.n_tokens = 1;
    std::memcpy(batch.embd, hidden.data(), hidden.size() * sizeof(float));
    batch.pos[0] = 0;
    batch.n_seq_id[0] = 1;
    batch.seq_id[0][0] = 0;
    batch.logits[0] = 1;

    const auto t0 = std::chrono::steady_clock::now();
    const int ret = llama_decode(ctx, batch);
    const auto t1 = std::chrono::steady_clock::now();
    llama_batch_free(batch);
    llama_kv_cache_clear(ctx);

    const double warmup_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cerr << "[layer-coop-server] warmup ret=" << ret
              << " ms=" << warmup_ms << std::endl;
}

static bool env_enabled(const char * name, bool default_value) {
    const char * value = std::getenv(name);
    if (!value || !value[0]) {
        return default_value;
    }
    return std::atoi(value) != 0;
}

static std::thread start_keep_hot_thread(std::atomic<bool> & active, std::atomic<bool> & stop) {
    return std::thread([&active, &stop]() {
        volatile uint64_t x = 0x9e3779b97f4a7c15ULL;
        while (!stop.load(std::memory_order_relaxed)) {
            if (active.load(std::memory_order_relaxed)) {
                for (int i = 0; i < 200000; ++i) {
                    x ^= x << 7;
                    x ^= x >> 9;
                    x += 0x9e3779b97f4a7c15ULL;
                }
            } else {
                std::this_thread::sleep_for(std::chrono::microseconds(200));
            }
        }
    });
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        std::cerr << "usage: " << argv[0] << " <model> <port> <split_m_layers> [threads] [connect_host:port|stdio]\n";
        return 1;
    }

    const std::string model_path = argv[1];
    const uint16_t port = (uint16_t) std::stoi(argv[2]);
    const int32_t split_m = std::stoi(argv[3]);
    const int32_t threads = argc > 4 ? std::stoi(argv[4]) : std::max(1u, std::thread::hardware_concurrency() / 2);
    const std::string connect_endpoint = argc > 5 ? argv[5] : "";

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
    LayerThreadpoolPtr threadpool = layer_attach_threadpool(ctx.get(), threads, "layer-coop-server");

    if (!layer_coop_net_init()) {
        std::cerr << "network init failed\n";
        return 1;
    }
    const int n_embd = llama_model_n_embd(model.get());
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
    warmup_tail_context(ctx.get(), n_embd);

    int64_t requests = 0;
    size_t recv_bytes_total = 0;
    size_t send_bytes_total = 0;
    double header_wait_ms_total = 0.0;
    double payload_recv_ms_total = 0.0;
    double decode_ms_total = 0.0;
    double send_ms_total = 0.0;
    std::atomic<bool> keep_hot_active(false);
    std::atomic<bool> keep_hot_stop(false);
    std::thread keep_hot_thread;
    if (env_enabled("LAYER_COOP_KEEP_HOT", false)) {
        keep_hot_thread = start_keep_hot_thread(keep_hot_active, keep_hot_stop);
        std::cerr << "[layer-coop-server] keep-hot enabled while waiting for network input\n";
    }
    auto stop_keep_hot = [&]() {
        keep_hot_active.store(false, std::memory_order_relaxed);
        keep_hot_stop.store(true, std::memory_order_relaxed);
        if (keep_hot_thread.joinable()) {
            keep_hot_thread.join();
        }
    };

    auto serve_connection = [&](int read_fd, int write_fd, bool socket_mode) {
        if (socket_mode) {
            int flag = 1;
            setsockopt(read_fd, IPPROTO_TCP, TCP_NODELAY, (const char *) &flag, sizeof(flag));
        }

        for (;;) {
            LayerCoopRequest req {};
            const auto t_header_wait0 = std::chrono::steady_clock::now();
            keep_hot_active.store(true, std::memory_order_relaxed);
            if (!layer_coop_recv_all(read_fd, &req, sizeof(req))) {
                keep_hot_active.store(false, std::memory_order_relaxed);
                break;
            }
            const auto t_header_wait1 = std::chrono::steady_clock::now();
            if (req.magic != LAYER_COOP_MAGIC || req.version != LAYER_COOP_VERSION ||
                    req.n_embd != n_embd || req.n_vocab != n_vocab || req.n_tokens <= 0) {
                keep_hot_active.store(false, std::memory_order_relaxed);
                LayerCoopResponse resp { LAYER_COOP_MAGIC, LAYER_COOP_VERSION, -1, 0, -1, 0, 0.0 };
                layer_coop_send_all(write_fd, &resp, sizeof(resp));
                break;
            }

            std::vector<int32_t> pos(req.n_tokens);
            std::vector<float> hidden((size_t) req.n_tokens * n_embd);
            const auto t_payload_recv0 = std::chrono::steady_clock::now();
            if (!layer_coop_recv_all(read_fd, pos.data(), pos.size() * sizeof(int32_t)) ||
                    !layer_coop_recv_all(read_fd, hidden.data(), hidden.size() * sizeof(float))) {
                keep_hot_active.store(false, std::memory_order_relaxed);
                break;
            }
            const auto t_payload_recv1 = std::chrono::steady_clock::now();
            keep_hot_active.store(false, std::memory_order_relaxed);
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
            const llama_token selected_token = logits ? greedy_sample(logits, n_vocab) : -1;
            LayerCoopResponse resp {
                LAYER_COOP_MAGIC,
                LAYER_COOP_VERSION,
                logits ? 0 : ret,
                0,
                selected_token,
                0,
                decode_ms,
            };
            const auto t_send0 = std::chrono::steady_clock::now();
            if (!layer_coop_send_all(write_fd, &resp, sizeof(resp))) break;
            const auto t_send1 = std::chrono::steady_clock::now();

            requests++;
            recv_bytes_total += sizeof(req) + pos.size() * sizeof(int32_t) + hidden.size() * sizeof(float);
            send_bytes_total += sizeof(resp);
            header_wait_ms_total += std::chrono::duration<double, std::milli>(t_header_wait1 - t_header_wait0).count();
            payload_recv_ms_total += std::chrono::duration<double, std::milli>(t_payload_recv1 - t_payload_recv0).count();
            decode_ms_total += decode_ms;
            send_ms_total += std::chrono::duration<double, std::milli>(t_send1 - t_send0).count();
            if (requests % 16 == 0) {
                std::cerr << "[layer-coop-server] requests=" << requests
                          << " avg_header_wait_ms=" << (header_wait_ms_total / requests)
                          << " avg_payload_recv_ms=" << (payload_recv_ms_total / requests)
                          << " avg_decode_ms=" << (decode_ms_total / requests)
                          << " avg_send_ms=" << (send_ms_total / requests)
                          << " recv_bytes=" << recv_bytes_total
                          << " send_bytes=" << send_bytes_total
                          << std::endl;
            }
        }
        keep_hot_active.store(false, std::memory_order_relaxed);
        if (requests > 0) {
            std::cerr << "[layer-coop-server-final] requests=" << requests
                      << " header_wait_ms=" << header_wait_ms_total
                      << " payload_recv_ms=" << payload_recv_ms_total
                      << " decode_ms=" << decode_ms_total
                      << " send_ms=" << send_ms_total
                      << " recv_bytes=" << recv_bytes_total
                      << " send_bytes=" << send_bytes_total
                      << std::endl;
        }
        if (socket_mode) {
            layer_coop_close(read_fd);
        }
    };

    if (!connect_endpoint.empty()) {
        if (connect_endpoint == "stdio") {
            std::cerr << "layer_mobile_server using stdio split_m=" << split_m
                      << " tail_layers=[" << split_m << "," << llama_model_n_layer(model.get()) << ")"
                      << " threads=" << threads << std::endl;
            serve_connection(0, 1, false);
            stop_keep_hot();
            return 0;
        }

        std::string host;
        uint16_t connect_port = 0;
        if (!layer_coop_parse_host_port(connect_endpoint, host, connect_port)) {
            std::cerr << "bad connect endpoint: " << connect_endpoint << "\n";
            return 1;
        }
        std::cout << "layer_mobile_server connecting to " << host << ":" << connect_port
                  << " split_m=" << split_m
                  << " tail_layers=[" << split_m << "," << llama_model_n_layer(model.get()) << ")"
                  << " threads=" << threads << std::endl;
        const bool reconnect = env_enabled("LAYER_COOP_RECONNECT", true);
        do {
            try {
                const int fd = layer_coop_connect(host, connect_port);
                std::cerr << "[layer-coop-server] connected to " << host << ":" << connect_port << "\n";
                serve_connection(fd, fd, true);
                std::cerr << "[layer-coop-server] disconnected from " << host << ":" << connect_port << "\n";
            } catch (const std::exception & e) {
                std::cerr << "[layer-coop-server] connect failed: " << e.what() << "\n";
            }
            if (reconnect) {
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
            }
        } while (reconnect);
        stop_keep_hot();
        return 0;
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

    std::cout << "layer_mobile_server listening on 0.0.0.0:" << port
              << " split_m=" << split_m
              << " tail_layers=[" << split_m << "," << llama_model_n_layer(model.get()) << ")"
              << " threads=" << threads << std::endl;

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
        serve_connection(fd, fd, true);
    }
}
