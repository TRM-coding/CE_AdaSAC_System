#include "llama-model.h"
#include "llama-context.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <arpa/inet.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <time.h>
#include <unordered_map>
#include <vector>

#ifdef GGML_USE_OPENMP
#include <omp.h>
#endif

namespace {

enum SvdOpKind {
    SVD_OP_UP = 0,
    SVD_OP_GATE = 1,
    SVD_OP_DOWN = 2,
};

enum RequestKind {
    REQ_MAT = 0,
    REQ_UP_GATE = 1,
};

struct WireRequest {
    uint32_t magic;
    uint32_t version;
    int32_t request_kind;
    int32_t layer_id;
    int32_t op_id;
    float offload_rate;
    int32_t rank_start;
    int32_t input_len;
};

struct WireResponse {
    uint32_t magic;
    uint32_t version;
    int32_t status;
    int32_t output_len;
    int32_t output_len_aux;
};

constexpr uint32_t kMagic = 0x5344564fU;
constexpr uint32_t kVersion = 2;

bool recv_all(int fd, void * data, size_t size) {
    char * ptr = static_cast<char *>(data);
    while (size > 0) {
        const ssize_t nread = recv(fd, ptr, size, 0);
        if (nread <= 0) {
            return false;
        }
        ptr += nread;
        size -= static_cast<size_t>(nread);
    }
    return true;
}

bool send_all(int fd, const void * data, size_t size) {
    const char * ptr = static_cast<const char *>(data);
    while (size > 0) {
        const ssize_t written = send(fd, ptr, size, MSG_NOSIGNAL);
        if (written <= 0) {
            return false;
        }
        ptr += written;
        size -= static_cast<size_t>(written);
    }
    return true;
}

struct SvdMatRefs {
    const ggml_tensor * u;
    const ggml_tensor * v;
};

struct MatExecutorKey {
    int32_t layer_id;
    int32_t op_id;
    int32_t rank_start;

    bool operator==(const MatExecutorKey & other) const {
        return layer_id == other.layer_id &&
               op_id == other.op_id &&
               rank_start == other.rank_start;
    }
};

struct UpGateExecutorKey {
    int32_t layer_id;
    int32_t rank_start;

    bool operator==(const UpGateExecutorKey & other) const {
        return layer_id == other.layer_id &&
               rank_start == other.rank_start;
    }
};

struct MatExecutorKeyHash {
    size_t operator()(const MatExecutorKey & key) const {
        size_t h = (size_t) key.layer_id;
        h = h * 1315423911u + (size_t) key.op_id;
        h = h * 1315423911u + (size_t) key.rank_start;
        return h;
    }
};

struct UpGateExecutorKeyHash {
    size_t operator()(const UpGateExecutorKey & key) const {
        size_t h = (size_t) key.layer_id;
        h = h * 1315423911u + (size_t) key.rank_start;
        return h;
    }
};

uint64_t now_us() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t) ts.tv_sec * 1000000ULL + (uint64_t) ts.tv_nsec / 1000ULL;
}

struct ServerProfile {
    uint64_t requests_up_gate = 0;
    uint64_t requests_mat_down = 0;
    uint64_t requests_mat_other = 0;
    uint64_t create_up_gate_us = 0;
    uint64_t create_mat_down_us = 0;
    uint64_t create_mat_other_us = 0;
    uint64_t run_up_gate_us = 0;
    uint64_t run_mat_down_us = 0;
    uint64_t run_mat_other_us = 0;
    uint64_t up_gate_cache_miss = 0;
    uint64_t mat_down_cache_miss = 0;
    uint64_t mat_other_cache_miss = 0;
};

SvdMatRefs get_svd_tensors(const llama_model & model, int32_t layer_id, int32_t op_id) {
    if (layer_id < 0 || layer_id >= static_cast<int32_t>(model.layers.size())) {
        throw std::runtime_error("invalid layer id");
    }

    const llama_layer & layer = model.layers[layer_id];
    switch (op_id) {
        case SVD_OP_UP:   return { layer.ffn_up_svd_u,   layer.ffn_up_svd_v };
        case SVD_OP_GATE: return { layer.ffn_gate_svd_u, layer.ffn_gate_svd_v };
        case SVD_OP_DOWN: return { layer.ffn_down_svd_u, layer.ffn_down_svd_v };
        default: throw std::runtime_error("invalid op id");
    }
}

void validate_svd_tensors(
        const ggml_tensor * u,
        const ggml_tensor * v,
        int32_t rank_start,
        int64_t expected_input_len) {
    if (u == nullptr || v == nullptr) {
        throw std::runtime_error("missing SVD tensors");
    }
    if (!ggml_is_contiguous(u) || !ggml_is_contiguous(v)) {
        throw std::runtime_error("non-contiguous SVD tensors are not supported");
    }
    if (expected_input_len != v->ne[0]) {
        throw std::runtime_error("unexpected input length");
    }
    const int64_t total_rank = v->ne[1];
    if (rank_start < 0 || rank_start >= total_rank) {
        throw std::runtime_error("invalid rank split");
    }
}

struct RemoteMatExecutor {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_tensor * input_tensor = nullptr;
    ggml_tensor * output_tensor = nullptr;
    ggml_cgraph * graph = nullptr;
    int64_t input_len = 0;
    int64_t output_len = 0;

    ~RemoteMatExecutor() {
        if (buffer != nullptr) {
            ggml_backend_buffer_free(buffer);
        }
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

struct RemoteUpGateExecutor {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_tensor * input_tensor = nullptr;
    ggml_tensor * output_up = nullptr;
    ggml_tensor * output_gate = nullptr;
    ggml_cgraph * graph = nullptr;
    int64_t input_len = 0;
    int64_t output_len = 0;

    ~RemoteUpGateExecutor() {
        if (buffer != nullptr) {
            ggml_backend_buffer_free(buffer);
        }
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

std::unique_ptr<RemoteMatExecutor> create_mat_executor(
        ggml_backend_t backend,
        const ggml_tensor * u,
        const ggml_tensor * v,
        int32_t rank_start,
        int64_t input_len) {
    validate_svd_tensors(u, v, rank_start, input_len);

    const int64_t total_rank = v->ne[1];
    const int64_t k_remote = total_rank - rank_start;

    ggml_init_params params {};
    params.mem_size = 16 * 1024 * 1024;
    params.mem_buffer = nullptr;
    params.no_alloc = true;

    auto exec = std::make_unique<RemoteMatExecutor>();
    exec->ctx = ggml_init(params);
    if (exec->ctx == nullptr) {
        throw std::runtime_error("ggml_init failed");
    }

    exec->input_len = input_len;
    exec->output_len = u->ne[1];
    exec->input_tensor = ggml_new_tensor_2d(exec->ctx, GGML_TYPE_F32, input_len, 1);
    ggml_tensor * v_tail = ggml_view_2d(
        exec->ctx,
        const_cast<ggml_tensor *>(v),
        v->ne[0],
        k_remote,
        v->nb[1],
        (size_t) rank_start * v->nb[1]);
    ggml_tensor * tmp = ggml_mul_mat(exec->ctx, v_tail, exec->input_tensor);
    ggml_tensor * u_tail = ggml_view_2d(
        exec->ctx,
        const_cast<ggml_tensor *>(u),
        k_remote,
        u->ne[1],
        u->nb[1],
        (size_t) rank_start * u->nb[0]);
    exec->output_tensor = ggml_mul_mat(exec->ctx, u_tail, tmp);

    exec->buffer = ggml_backend_alloc_ctx_tensors(exec->ctx, backend);
    if (exec->buffer == nullptr) {
        throw std::runtime_error("ggml_backend_alloc_ctx_tensors failed");
    }

    exec->graph = ggml_new_graph(exec->ctx);
    if (exec->graph == nullptr) {
        throw std::runtime_error("ggml_new_graph failed");
    }
    ggml_build_forward_expand(exec->graph, exec->output_tensor);

    return exec;
}

std::unique_ptr<RemoteUpGateExecutor> create_up_gate_executor(
        ggml_backend_t backend,
        const ggml_tensor * up_u,
        const ggml_tensor * up_v,
        const ggml_tensor * gate_u,
        const ggml_tensor * gate_v,
        int32_t rank_start,
        int64_t input_len) {
    validate_svd_tensors(up_u, up_v, rank_start, input_len);
    validate_svd_tensors(gate_u, gate_v, rank_start, input_len);

    const int64_t total_rank = up_v->ne[1];
    const int64_t k_remote = total_rank - rank_start;

    ggml_init_params params {};
    params.mem_size = 24 * 1024 * 1024;
    params.mem_buffer = nullptr;
    params.no_alloc = true;

    auto exec = std::make_unique<RemoteUpGateExecutor>();
    exec->ctx = ggml_init(params);
    if (exec->ctx == nullptr) {
        throw std::runtime_error("ggml_init failed");
    }

    exec->input_len = input_len;
    exec->output_len = up_u->ne[1];
    exec->input_tensor = ggml_new_tensor_2d(exec->ctx, GGML_TYPE_F32, input_len, 1);

    ggml_tensor * up_v_tail = ggml_view_2d(
        exec->ctx,
        const_cast<ggml_tensor *>(up_v),
        up_v->ne[0],
        k_remote,
        up_v->nb[1],
        (size_t) rank_start * up_v->nb[1]);
    ggml_tensor * up_tmp = ggml_mul_mat(exec->ctx, up_v_tail, exec->input_tensor);
    ggml_tensor * up_u_tail = ggml_view_2d(
        exec->ctx,
        const_cast<ggml_tensor *>(up_u),
        k_remote,
        up_u->ne[1],
        up_u->nb[1],
        (size_t) rank_start * up_u->nb[0]);
    exec->output_up = ggml_mul_mat(exec->ctx, up_u_tail, up_tmp);

    ggml_tensor * gate_v_tail = ggml_view_2d(
        exec->ctx,
        const_cast<ggml_tensor *>(gate_v),
        gate_v->ne[0],
        k_remote,
        gate_v->nb[1],
        (size_t) rank_start * gate_v->nb[1]);
    ggml_tensor * gate_tmp = ggml_mul_mat(exec->ctx, gate_v_tail, exec->input_tensor);
    ggml_tensor * gate_u_tail = ggml_view_2d(
        exec->ctx,
        const_cast<ggml_tensor *>(gate_u),
        k_remote,
        gate_u->ne[1],
        gate_u->nb[1],
        (size_t) rank_start * gate_u->nb[0]);
    exec->output_gate = ggml_mul_mat(exec->ctx, gate_u_tail, gate_tmp);

    exec->buffer = ggml_backend_alloc_ctx_tensors(exec->ctx, backend);
    if (exec->buffer == nullptr) {
        throw std::runtime_error("ggml_backend_alloc_ctx_tensors failed");
    }

    exec->graph = ggml_new_graph(exec->ctx);
    if (exec->graph == nullptr) {
        throw std::runtime_error("ggml_new_graph failed");
    }
    ggml_build_forward_expand(exec->graph, exec->output_up);
    ggml_build_forward_expand(exec->graph, exec->output_gate);

    return exec;
}

void run_mat_executor(
        ggml_backend_t backend,
        RemoteMatExecutor & exec,
        const std::vector<float> & input,
        std::vector<float> & output) {
    if ((int64_t) input.size() != exec.input_len) {
        throw std::runtime_error("unexpected input length");
    }

    output.assign((size_t) exec.output_len, 0.0f);
    ggml_backend_tensor_set(exec.input_tensor, input.data(), 0, sizeof(float) * input.size());
    const ggml_status status = ggml_backend_graph_compute(backend, exec.graph);
    if (status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("ggml_backend_graph_compute failed");
    }
    ggml_backend_tensor_get(exec.output_tensor, output.data(), 0, sizeof(float) * output.size());
}

void run_up_gate_executor(
        ggml_backend_t backend,
        RemoteUpGateExecutor & exec,
        const std::vector<float> & input,
        std::vector<float> & output,
        std::vector<float> & output_aux) {
    if ((int64_t) input.size() != exec.input_len) {
        throw std::runtime_error("unexpected input length");
    }

    output.assign((size_t) exec.output_len, 0.0f);
    output_aux.assign((size_t) exec.output_len, 0.0f);
    ggml_backend_tensor_set(exec.input_tensor, input.data(), 0, sizeof(float) * input.size());
    const ggml_status status = ggml_backend_graph_compute(backend, exec.graph);
    if (status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("ggml_backend_graph_compute failed");
    }
    ggml_backend_tensor_get(exec.output_up, output.data(), 0, sizeof(float) * output.size());
    ggml_backend_tensor_get(exec.output_gate, output_aux.data(), 0, sizeof(float) * output_aux.size());
}

} // namespace

int main(int argc, char ** argv) {
    const std::string model_path = argc > 1
        ? argv[1]
        : "/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/gguf_models/qwen.gguf.sort_svd.compact.gguf";
    const int port = argc > 2 ? std::stoi(argv[2]) : 7788;
    const int n_threads = argc > 3 ? std::stoi(argv[3]) : 8;

    ggml_backend_load_all();
    ggml_cpu_init();

#ifdef GGML_USE_OPENMP
    omp_set_num_threads(n_threads);
#endif

    ggml_backend_t backend = ggml_backend_cpu_init();
    if (backend == nullptr) {
        std::cerr << "ggml_backend_cpu_init failed" << std::endl;
        return 1;
    }
    ggml_backend_cpu_set_n_threads(backend, n_threads);

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;

    std::cout << "loading mobile-side model: " << model_path << std::endl;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (model == nullptr) {
        std::cerr << "failed to load model" << std::endl;
        return 1;
    }

    const int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd < 0) {
        std::cerr << "socket failed: " << strerror(errno) << std::endl;
        llama_model_free(model);
        return 1;
    }

    int one = 1;
    setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    setsockopt(listen_fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));

    sockaddr_in addr {};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(static_cast<uint16_t>(port));

    if (bind(listen_fd, reinterpret_cast<const sockaddr *>(&addr), sizeof(addr)) != 0 ||
        listen(listen_fd, 1) != 0) {
        std::cerr << "bind/listen failed: " << strerror(errno) << std::endl;
        close(listen_fd);
        llama_model_free(model);
        return 1;
    }

    std::cout << "svd_mobile_server listening on 0.0.0.0:" << port
              << " threads=" << n_threads << std::endl;

    std::unordered_map<MatExecutorKey, std::unique_ptr<RemoteMatExecutor>, MatExecutorKeyHash> mat_executors;
    std::unordered_map<UpGateExecutorKey, std::unique_ptr<RemoteUpGateExecutor>, UpGateExecutorKeyHash> up_gate_executors;

    while (true) {
        const int client_fd = accept(listen_fd, nullptr, nullptr);
        if (client_fd < 0) {
            std::cerr << "accept failed: " << strerror(errno) << std::endl;
            continue;
        }
        setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        std::cout << "client connected" << std::endl;
        ServerProfile prof;

        while (true) {
            WireRequest req {};
            if (!recv_all(client_fd, &req, sizeof(req))) {
                break;
            }
            if (req.magic != kMagic || req.version != kVersion || req.input_len <= 0) {
                break;
            }

            std::vector<float> input(static_cast<size_t>(req.input_len));
            if (!recv_all(client_fd, input.data(), sizeof(float) * input.size())) {
                break;
            }

            WireResponse rsp { kMagic, kVersion, 0, 0, 0 };
            std::vector<float> output;
            std::vector<float> output_aux;
            try {
                if (req.request_kind == REQ_UP_GATE) {
                    prof.requests_up_gate++;
                    const UpGateExecutorKey key { req.layer_id, req.rank_start };
                    const auto up = get_svd_tensors(*model, req.layer_id, SVD_OP_UP);
                    const auto gate = get_svd_tensors(*model, req.layer_id, SVD_OP_GATE);
                    auto & exec = up_gate_executors[key];
                    if (!exec) {
                        prof.up_gate_cache_miss++;
                        const uint64_t t0 = now_us();
                        exec = create_up_gate_executor(backend, up.u, up.v, gate.u, gate.v, req.rank_start, req.input_len);
                        prof.create_up_gate_us += now_us() - t0;
                    }
                    const uint64_t t0 = now_us();
                    run_up_gate_executor(backend, *exec, input, output, output_aux);
                    prof.run_up_gate_us += now_us() - t0;
                    rsp.output_len = static_cast<int32_t>(output.size());
                    rsp.output_len_aux = static_cast<int32_t>(output_aux.size());
                } else {
                    const bool is_down = req.op_id == SVD_OP_DOWN;
                    if (is_down) {
                        prof.requests_mat_down++;
                    } else {
                        prof.requests_mat_other++;
                    }
                    const MatExecutorKey key { req.layer_id, req.op_id, req.rank_start };
                    const auto mat = get_svd_tensors(*model, req.layer_id, req.op_id);
                    auto & exec = mat_executors[key];
                    if (!exec) {
                        const uint64_t t0 = now_us();
                        exec = create_mat_executor(backend, mat.u, mat.v, req.rank_start, req.input_len);
                        const uint64_t dt = now_us() - t0;
                        if (is_down) {
                            prof.mat_down_cache_miss++;
                            prof.create_mat_down_us += dt;
                        } else {
                            prof.mat_other_cache_miss++;
                            prof.create_mat_other_us += dt;
                        }
                    }
                    const uint64_t t0 = now_us();
                    run_mat_executor(backend, *exec, input, output);
                    const uint64_t dt = now_us() - t0;
                    if (is_down) {
                        prof.run_mat_down_us += dt;
                    } else {
                        prof.run_mat_other_us += dt;
                    }
                    rsp.output_len = static_cast<int32_t>(output.size());
                }
            } catch (const std::exception & e) {
                rsp.status = -1;
                std::cerr << "request failed: " << e.what() << std::endl;
            }

            if (!send_all(client_fd, &rsp, sizeof(rsp))) {
                break;
            }
            if (rsp.status == 0) {
                if (!send_all(client_fd, output.data(), sizeof(float) * output.size())) {
                    break;
                }
                if (rsp.output_len_aux > 0 &&
                    !send_all(client_fd, output_aux.data(), sizeof(float) * output_aux.size())) {
                    break;
                }
            }
        }

        std::cerr
            << "[svd-offload-server] up_gate_req=" << prof.requests_up_gate
            << " down_req=" << prof.requests_mat_down
            << " other_mat_req=" << prof.requests_mat_other
            << " up_gate_miss=" << prof.up_gate_cache_miss
            << " down_miss=" << prof.mat_down_cache_miss
            << " other_miss=" << prof.mat_other_cache_miss
            << " create_up_gate=" << (prof.create_up_gate_us / 1000.0) << " ms"
            << " create_down=" << (prof.create_mat_down_us / 1000.0) << " ms"
            << " create_other=" << (prof.create_mat_other_us / 1000.0) << " ms"
            << " run_up_gate=" << (prof.run_up_gate_us / 1000.0) << " ms"
            << " run_down=" << (prof.run_mat_down_us / 1000.0) << " ms"
            << " run_other=" << (prof.run_mat_other_us / 1000.0) << " ms"
            << std::endl;
        std::cout << "client disconnected" << std::endl;
        close(client_fd);
    }

    close(listen_fd);
    llama_model_free(model);
    ggml_backend_free(backend);
    return 0;
}
