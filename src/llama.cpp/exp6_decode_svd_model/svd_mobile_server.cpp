#include "llama-model.h"
#include "llama-context.h"

#include <arpa/inet.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

enum SvdOpKind {
    SVD_OP_UP = 0,
    SVD_OP_GATE = 1,
    SVD_OP_DOWN = 2,
};

struct WireRequest {
    uint32_t magic;
    uint32_t version;
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
};

constexpr uint32_t kMagic = 0x5344564fU;
constexpr uint32_t kVersion = 1;

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

float tensor_get_f32(const ggml_tensor * tensor, int64_t row, int64_t col) {
    const char * base = static_cast<const char *>(tensor->data) + row * tensor->nb[1] + col * tensor->nb[0];
    switch (tensor->type) {
        case GGML_TYPE_F32:
            return *reinterpret_cast<const float *>(base);
        case GGML_TYPE_F16:
            return ggml_fp16_to_fp32(*reinterpret_cast<const ggml_fp16_t *>(base));
        default:
            throw std::runtime_error("svd_mobile_server only supports F16/F32 SVD tensors");
    }
}

void compute_remote_tail(
        const llama_model & model,
        int32_t layer_id,
        int32_t op_id,
        int32_t rank_start,
        const std::vector<float> & input,
        std::vector<float> & output) {
    if (layer_id < 0 || layer_id >= static_cast<int32_t>(model.layers.size())) {
        throw std::runtime_error("invalid layer id");
    }

    const llama_layer & layer = model.layers[layer_id];
    const ggml_tensor * u = nullptr;
    const ggml_tensor * v = nullptr;

    switch (op_id) {
        case SVD_OP_UP:
            u = layer.ffn_up_svd_u;
            v = layer.ffn_up_svd_v;
            break;
        case SVD_OP_GATE:
            u = layer.ffn_gate_svd_u;
            v = layer.ffn_gate_svd_v;
            break;
        case SVD_OP_DOWN:
            u = layer.ffn_down_svd_u;
            v = layer.ffn_down_svd_v;
            break;
        default:
            throw std::runtime_error("invalid op id");
    }

    if (u == nullptr || v == nullptr) {
        throw std::runtime_error("missing SVD tensors");
    }
    if (input.size() != static_cast<size_t>(v->ne[0])) {
        throw std::runtime_error("unexpected input length");
    }

    const int64_t total_rank = v->ne[1];
    if (rank_start < 0 || rank_start >= total_rank) {
        throw std::runtime_error("invalid rank split");
    }

    output.assign(static_cast<size_t>(u->ne[1]), 0.0f);
    std::vector<float> tmp(static_cast<size_t>(total_rank - rank_start), 0.0f);

    for (int64_t r = rank_start; r < total_rank; ++r) {
        float acc = 0.0f;
        for (int64_t c = 0; c < v->ne[0]; ++c) {
            acc += tensor_get_f32(v, r, c) * input[static_cast<size_t>(c)];
        }
        tmp[static_cast<size_t>(r - rank_start)] = acc;
    }

    for (int64_t out_idx = 0; out_idx < u->ne[1]; ++out_idx) {
        float acc = 0.0f;
        for (int64_t r = rank_start; r < total_rank; ++r) {
            acc += tensor_get_f32(u, out_idx, r) * tmp[static_cast<size_t>(r - rank_start)];
        }
        output[static_cast<size_t>(out_idx)] = acc;
    }
}

} // namespace

int main(int argc, char ** argv) {
    const std::string model_path = argc > 1
        ? argv[1]
        : "/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/gguf_models/qwen.gguf.sort_svd.compact.gguf";
    const int port = argc > 2 ? std::stoi(argv[2]) : 7788;

    ggml_backend_load_all();

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

    std::cout << "svd_mobile_server listening on 0.0.0.0:" << port << std::endl;

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

            WireResponse rsp { kMagic, kVersion, 0, 0 };
            std::vector<float> output;
            try {
                compute_remote_tail(*model, req.layer_id, req.op_id, req.rank_start, input, output);
                rsp.output_len = static_cast<int32_t>(output.size());
            } catch (const std::exception & e) {
                rsp.status = -1;
                std::cerr << "request failed: " << e.what() << std::endl;
            }

            if (!send_all(client_fd, &rsp, sizeof(rsp))) {
                break;
            }
            if (rsp.status == 0 && !send_all(client_fd, output.data(), sizeof(float) * output.size())) {
                break;
            }
        }

        std::cout << "client disconnected" << std::endl;
        close(client_fd);
    }

    close(listen_fd);
    llama_model_free(model);
    return 0;
}
