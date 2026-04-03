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
#include <stdexcept>
#include <string>
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

void compute_remote_tail_with_ggml(
        const ggml_tensor * u,
        const ggml_tensor * v,
        int32_t rank_start,
        const std::vector<float> & input,
        std::vector<float> & output,
        int32_t n_threads) {
    if (u == nullptr || v == nullptr) {
        throw std::runtime_error("missing SVD tensors");
    }
    if (!ggml_is_contiguous(u) || !ggml_is_contiguous(v)) {
        throw std::runtime_error("non-contiguous SVD tensors are not supported");
    }
    if (input.size() != static_cast<size_t>(v->ne[0])) {
        throw std::runtime_error("unexpected input length");
    }

    const int64_t total_rank = v->ne[1];
    if (rank_start < 0 || rank_start >= total_rank) {
        throw std::runtime_error("invalid rank split");
    }

    const int64_t k_remote = total_rank - rank_start;
    output.assign(static_cast<size_t>(u->ne[1]), 0.0f);

    ggml_init_params params {};
    params.mem_size = 16 * 1024 * 1024;
    params.mem_buffer = nullptr;
    params.no_alloc = true;

    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        throw std::runtime_error("ggml_init failed");
    }

    ggml_backend_t backend = ggml_backend_cpu_init();
    if (backend == nullptr) {
        ggml_free(ctx);
        throw std::runtime_error("ggml_backend_cpu_init failed");
    }
    ggml_backend_cpu_set_n_threads(backend, n_threads);

    ggml_tensor * input_tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (int64_t) input.size(), 1);
    ggml_tensor * v_tail = ggml_view_2d(
        ctx,
        const_cast<ggml_tensor *>(v),
        v->ne[0],
        k_remote,
        v->nb[1],
        (size_t) rank_start * v->nb[1]);
    ggml_tensor * tmp = ggml_mul_mat(ctx, v_tail, input_tensor);
    ggml_tensor * u_tail = ggml_view_2d(
        ctx,
        const_cast<ggml_tensor *>(u),
        k_remote,
        u->ne[1],
        u->nb[1],
        (size_t) rank_start * u->nb[0]);
    ggml_tensor * output_tensor = ggml_mul_mat(ctx, u_tail, tmp);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buffer == nullptr) {
        ggml_backend_free(backend);
        ggml_free(ctx);
        throw std::runtime_error("ggml_backend_alloc_ctx_tensors failed");
    }

    ggml_backend_tensor_set(input_tensor, input.data(), 0, sizeof(float) * input.size());

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output_tensor);
    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        ggml_backend_buffer_free(buffer);
        ggml_backend_free(backend);
        ggml_free(ctx);
        throw std::runtime_error("ggml_backend_graph_compute failed");
    }

    ggml_backend_tensor_get(output_tensor, output.data(), 0, sizeof(float) * output.size());

    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    ggml_free(ctx);
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

    while (true) {
        const int client_fd = accept(listen_fd, nullptr, nullptr);
        if (client_fd < 0) {
            std::cerr << "accept failed: " << strerror(errno) << std::endl;
            continue;
        }
        setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        std::cout << "client connected" << std::endl;

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
                    const auto up = get_svd_tensors(*model, req.layer_id, SVD_OP_UP);
                    const auto gate = get_svd_tensors(*model, req.layer_id, SVD_OP_GATE);
                    compute_remote_tail_with_ggml(up.u, up.v, req.rank_start, input, output, n_threads);
                    compute_remote_tail_with_ggml(gate.u, gate.v, req.rank_start, input, output_aux, n_threads);
                    rsp.output_len = static_cast<int32_t>(output.size());
                    rsp.output_len_aux = static_cast<int32_t>(output_aux.size());
                } else {
                    const auto mat = get_svd_tensors(*model, req.layer_id, req.op_id);
                    compute_remote_tail_with_ggml(mat.u, mat.v, req.rank_start, input, output, n_threads);
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

        std::cout << "client disconnected" << std::endl;
        close(client_fd);
    }

    close(listen_fd);
    llama_model_free(model);
    return 0;
}
