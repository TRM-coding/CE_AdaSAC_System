#include "llama-model.h"
#include "llama-context.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#ifdef __aarch64__
ggml_backend_buffer_type_t ggml_backend_cpu_aarch64_buffer_type(void);
#endif

#ifdef _WIN32
#include <malloc.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#else
#include <arpa/inet.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
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
    REQ_FFN = 2,
};

enum class ExecutorMode {
    AUTO = 0,
    SVD,
    DENSE_TAIL,
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
constexpr size_t kWorkDataAlign = 64;

#ifdef _WIN32
bool ensure_winsock_initialized() {
    static bool initialized = false;
    static bool ok = false;
    if (!initialized) {
        WSADATA wsa_data;
        ok = WSAStartup(MAKEWORD(2, 2), &wsa_data) == 0;
        initialized = true;
    }
    return ok;
}

int close_socket(int fd) {
    return closesocket(static_cast<SOCKET>(fd));
}

void * aligned_alloc_work_data(size_t size) {
    return _aligned_malloc(size, kWorkDataAlign);
}

void free_work_data(void * ptr) {
    _aligned_free(ptr);
}
#else
bool ensure_winsock_initialized() {
    return true;
}

int close_socket(int fd) {
    return close(fd);
}

void * aligned_alloc_work_data(size_t size) {
    void * ptr = nullptr;
    if (posix_memalign(&ptr, kWorkDataAlign, size) != 0) {
        return nullptr;
    }
    return ptr;
}

void free_work_data(void * ptr) {
    free(ptr);
}
#endif

enum class QuantizationMode {
    NONE = 0,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q4_K,
    Q5_K,
    Q6_K,
};

enum class QuantizeFallbackReason {
    NONE = 0,
    TYPE_DISABLED,
    RANK_NOT_2D,
    ROW_NOT_ALIGNED,
    TRAIT_UNSUPPORTED,
    ALLOC_FAILED,
};

const char * quant_mode_name(QuantizationMode mode) {
    switch (mode) {
        case QuantizationMode::NONE: return "off";
        case QuantizationMode::Q4_0: return "q4_0";
        case QuantizationMode::Q4_1: return "q4_1";
        case QuantizationMode::Q5_0: return "q5_0";
        case QuantizationMode::Q5_1: return "q5_1";
        case QuantizationMode::Q8_0: return "q8_0";
        case QuantizationMode::Q4_K: return "q4_k";
        case QuantizationMode::Q5_K: return "q5_k";
        case QuantizationMode::Q6_K: return "q6_k";
    }
    return "unknown";
}

enum ggml_type quant_mode_to_ggml_type(QuantizationMode mode) {
    switch (mode) {
        case QuantizationMode::NONE: return GGML_TYPE_F16;
        case QuantizationMode::Q4_0: return GGML_TYPE_Q4_0;
        case QuantizationMode::Q4_1: return GGML_TYPE_Q4_1;
        case QuantizationMode::Q5_0: return GGML_TYPE_Q5_0;
        case QuantizationMode::Q5_1: return GGML_TYPE_Q5_1;
        case QuantizationMode::Q8_0: return GGML_TYPE_Q8_0;
        case QuantizationMode::Q4_K: return GGML_TYPE_Q4_K;
        case QuantizationMode::Q5_K: return GGML_TYPE_Q5_K;
        case QuantizationMode::Q6_K: return GGML_TYPE_Q6_K;
    }
    return GGML_TYPE_F16;
}

const char * executor_mode_name(ExecutorMode mode) {
    switch (mode) {
        case ExecutorMode::AUTO: return "auto";
        case ExecutorMode::SVD: return "svd";
        case ExecutorMode::DENSE_TAIL: return "dense_tail";
    }
    return "unknown";
}

ExecutorMode parse_executor_mode(const std::string & arg) {
    if (arg.empty() || arg == "auto") return ExecutorMode::AUTO;
    if (arg == "svd") return ExecutorMode::SVD;
    if (arg == "dense" || arg == "dense_tail") return ExecutorMode::DENSE_TAIL;
    throw std::runtime_error("unsupported executor mode: " + arg);
}

QuantizationMode quant_mode_from_tensor_type(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0: return QuantizationMode::Q4_0;
        case GGML_TYPE_Q4_1: return QuantizationMode::Q4_1;
        case GGML_TYPE_Q5_0: return QuantizationMode::Q5_0;
        case GGML_TYPE_Q5_1: return QuantizationMode::Q5_1;
        case GGML_TYPE_Q8_0: return QuantizationMode::Q8_0;
        case GGML_TYPE_Q4_K: return QuantizationMode::Q4_K;
        case GGML_TYPE_Q5_K: return QuantizationMode::Q5_K;
        case GGML_TYPE_Q6_K: return QuantizationMode::Q6_K;
        default: return QuantizationMode::NONE;
    }
}

QuantizationMode choose_dense_quant_mode(
        const ggml_tensor * u,
        const ggml_tensor * v,
        QuantizationMode quant_mode) {
    if (quant_mode != QuantizationMode::NONE) {
        return quant_mode;
    }
    const QuantizationMode u_mode = quant_mode_from_tensor_type(u->type);
    const QuantizationMode v_mode = quant_mode_from_tensor_type(v->type);
    if (u_mode != QuantizationMode::NONE && u_mode == v_mode) {
        return u_mode;
    }
    if (u_mode != QuantizationMode::NONE) {
        return u_mode;
    }
    if (v_mode != QuantizationMode::NONE) {
        return v_mode;
    }
    return QuantizationMode::NONE;
}

ExecutorMode resolve_executor_mode(
        ExecutorMode requested,
        const ggml_tensor * u,
        const ggml_tensor * v,
        QuantizationMode quant_mode) {
    if (requested != ExecutorMode::AUTO) {
        return requested;
    }
    return choose_dense_quant_mode(u, v, quant_mode) != QuantizationMode::NONE
        ? ExecutorMode::DENSE_TAIL
        : ExecutorMode::SVD;
}

const char * backend_path_name() {
#if defined(__aarch64__)
    return "cpu-aarch64";
#elif defined(__x86_64__) || defined(_M_X64)
    return "cpu-x86_64";
#else
    return "cpu-generic";
#endif
}

QuantizationMode parse_quantization_mode(const std::string & arg) {
    if (arg.empty() || arg == "off" || arg == "none" || arg == "fp16") {
        return QuantizationMode::NONE;
    }
    if (arg == "q4_0") return QuantizationMode::Q4_0;
    if (arg == "q4_1") return QuantizationMode::Q4_1;
    if (arg == "q5_0") return QuantizationMode::Q5_0;
    if (arg == "q5_1") return QuantizationMode::Q5_1;
    if (arg == "q8_0") return QuantizationMode::Q8_0;
    if (arg == "q4_k") return QuantizationMode::Q4_K;
    if (arg == "q5_k") return QuantizationMode::Q5_K;
    if (arg == "q6_k") return QuantizationMode::Q6_K;
    throw std::runtime_error("unsupported quantization mode: " + arg);
}

const char * quant_fallback_reason_name(QuantizeFallbackReason reason) {
    switch (reason) {
        case QuantizeFallbackReason::NONE: return "none";
        case QuantizeFallbackReason::TYPE_DISABLED: return "type_disabled";
        case QuantizeFallbackReason::RANK_NOT_2D: return "rank_not_2d";
        case QuantizeFallbackReason::ROW_NOT_ALIGNED: return "row_not_aligned";
        case QuantizeFallbackReason::TRAIT_UNSUPPORTED: return "trait_unsupported";
        case QuantizeFallbackReason::ALLOC_FAILED: return "alloc_failed";
    }
    return "unknown";
}

bool recv_all(int fd, void * data, size_t size) {
    char * ptr = static_cast<char *>(data);
    while (size > 0) {
        const int nread = recv(fd, ptr, static_cast<int>(size), 0);
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
        int flags = 0;
#ifdef MSG_NOSIGNAL
        flags |= MSG_NOSIGNAL;
#endif
        const int written = send(fd, ptr, static_cast<int>(size), flags);
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
#ifdef _WIN32
    return static_cast<uint64_t>(GetTickCount64()) * 1000ULL;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t) ts.tv_sec * 1000000ULL + (uint64_t) ts.tv_nsec / 1000ULL;
#endif
}

void accumulate_output_stats(
        const float * data,
        size_t len,
        double & sum_abs_total,
        float & max_abs_total,
        uint64_t & zero_vectors) {
    if (data == nullptr || len == 0) {
        return;
    }

    double sum_abs = 0.0;
    float max_abs = 0.0f;
    for (size_t i = 0; i < len; ++i) {
        const float v = std::fabs(data[i]);
        sum_abs += static_cast<double>(v);
        if (v > max_abs) {
            max_abs = v;
        }
    }

    sum_abs_total += sum_abs;
    if (max_abs > max_abs_total) {
        max_abs_total = max_abs;
    }
    if (sum_abs == 0.0) {
        zero_vectors++;
    }
}

struct TensorStats {
    double sum_abs = 0.0;
    float max_abs = 0.0f;
    bool has_nan = false;
};

TensorStats analyze_float_data(const float * data, size_t len) {
    TensorStats stats {};
    if (data == nullptr || len == 0) {
        return stats;
    }

    for (size_t i = 0; i < len; ++i) {
        const float raw = data[i];
        if (std::isnan(raw)) {
            stats.has_nan = true;
            continue;
        }
        const float v = std::fabs(raw);
        stats.sum_abs += static_cast<double>(v);
        if (v > stats.max_abs) {
            stats.max_abs = v;
        }
    }
    return stats;
}

struct ServerProfile {
    uint64_t requests_up_gate = 0;
    uint64_t requests_mat_down = 0;
    uint64_t requests_mat_other = 0;
    uint64_t requests_ffn = 0;
    uint64_t create_up_gate_us = 0;
    uint64_t create_mat_down_us = 0;
    uint64_t create_mat_other_us = 0;
    uint64_t create_ffn_us = 0;
    uint64_t run_up_gate_us = 0;
    uint64_t run_mat_down_us = 0;
    uint64_t run_mat_other_us = 0;
    uint64_t run_ffn_us = 0;
    uint64_t up_gate_cache_miss = 0;
    uint64_t mat_down_cache_miss = 0;
    uint64_t mat_other_cache_miss = 0;
    uint64_t ffn_cache_miss = 0;
    uint64_t dense_up_gate_cache_miss = 0;
    uint64_t dense_mat_down_cache_miss = 0;
    uint64_t dense_mat_other_cache_miss = 0;
    uint64_t dense_ffn_cache_miss = 0;
    uint64_t dense_weight_tensors = 0;
    uint64_t dense_weight_bytes = 0;
    uint64_t quantized_tail_tensors = 0;
    uint64_t quantized_tail_bytes = 0;
    uint64_t quantize_fallbacks = 0;
    uint64_t quantize_fallback_type_disabled = 0;
    uint64_t quantize_fallback_rank_not_2d = 0;
    uint64_t quantize_fallback_row_not_aligned = 0;
    uint64_t quantize_fallback_trait_unsupported = 0;
    uint64_t quantize_fallback_alloc_failed = 0;
    double output_up_sum_abs = 0.0;
    double output_gate_sum_abs = 0.0;
    double output_down_sum_abs = 0.0;
    double output_other_sum_abs = 0.0;
    double output_ffn_sum_abs = 0.0;
    float output_up_max_abs = 0.0f;
    float output_gate_max_abs = 0.0f;
    float output_down_max_abs = 0.0f;
    float output_other_max_abs = 0.0f;
    float output_ffn_max_abs = 0.0f;
    uint64_t output_up_zero_vectors = 0;
    uint64_t output_gate_zero_vectors = 0;
    uint64_t output_down_zero_vectors = 0;
    uint64_t output_other_zero_vectors = 0;
    uint64_t output_ffn_zero_vectors = 0;
    uint64_t debug_logged_requests = 0;
};

struct OwnedTensor {
    ggml_context * ctx = nullptr;
    ggml_tensor * tensor = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    std::vector<uint8_t> data;
    std::vector<float> row_buffer;

    ~OwnedTensor() {
        if (buffer != nullptr) {
            ggml_backend_buffer_free(buffer);
        }
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

struct PreparedTensor {
    ggml_tensor * tensor = nullptr;
    std::unique_ptr<OwnedTensor> owned;
    bool quantized = false;
    QuantizeFallbackReason fallback_reason = QuantizeFallbackReason::NONE;
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
    ggml_tensor * input_tensor = nullptr;
    ggml_tensor * output_tensor = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_cplan plan {};
    void * work_data = nullptr;
    int64_t input_len = 0;
    int64_t output_len = 0;
    std::vector<float> input_buffer;
    std::vector<float> output_buffer;
    std::vector<std::unique_ptr<OwnedTensor>> owned_tensors;

    ~RemoteMatExecutor() {
        if (work_data != nullptr) {
            free_work_data(work_data);
        }
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

struct RemoteUpGateExecutor {
    ggml_context * ctx = nullptr;
    ggml_tensor * input_tensor = nullptr;
    ggml_tensor * output_up = nullptr;
    ggml_tensor * output_gate = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_cplan plan {};
    void * work_data = nullptr;
    int64_t input_len = 0;
    int64_t output_len = 0;
    std::vector<float> input_buffer;
    std::vector<float> output_up_buffer;
    std::vector<float> output_gate_buffer;
    std::vector<std::unique_ptr<OwnedTensor>> owned_tensors;

    ~RemoteUpGateExecutor() {
        if (work_data != nullptr) {
            free_work_data(work_data);
        }
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

struct RemoteFfnExecutor {
    ggml_context * ctx = nullptr;
    ggml_tensor * input_tensor = nullptr;
    ggml_tensor * output_tensor = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_cplan plan {};
    void * work_data = nullptr;
    int64_t input_len = 0;
    int64_t output_len = 0;
    std::vector<float> input_buffer;
    std::vector<float> output_buffer;
    std::vector<std::unique_ptr<OwnedTensor>> owned_tensors;

    ~RemoteFfnExecutor() {
        if (work_data != nullptr) {
            free_work_data(work_data);
        }
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

void * alloc_work_data(size_t size) {
    if (size == 0) {
        return nullptr;
    }

    return aligned_alloc_work_data(size);
}

ggml_tensor * make_weight_shape_tensor(
        ggml_context * ctx,
        int64_t input_len,
        int64_t output_len) {
    ggml_tensor * weight_shape = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, input_len, output_len);
    if (weight_shape == nullptr) {
        throw std::runtime_error("ggml_new_tensor_2d failed");
    }
    return weight_shape;
}

int64_t tensor_row_count(const ggml_tensor * tensor) {
    return tensor->ne[1] * tensor->ne[2] * tensor->ne[3];
}

void maybe_upload_owned_tensor_to_aarch64_buffer(OwnedTensor * owned) {
#ifdef __aarch64__
    if (owned == nullptr || owned->ctx == nullptr || owned->tensor == nullptr || owned->data.empty()) {
        return;
    }
    if (owned->tensor->type == GGML_TYPE_F32) {
        return;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_cpu_aarch64_buffer_type();
    if (buft == nullptr) {
        return;
    }
    owned->buffer = ggml_backend_alloc_ctx_tensors_from_buft(owned->ctx, buft);
    if (owned->buffer == nullptr) {
        return;
    }
    ggml_backend_tensor_set(owned->tensor, owned->data.data(), 0, owned->data.size());
#else
    GGML_UNUSED(owned);
#endif
}

PreparedTensor maybe_quantize_tensor(
        const ggml_tensor * source,
        QuantizationMode quant_mode,
        ServerProfile * prof) {
    PreparedTensor result {};
    result.tensor = const_cast<ggml_tensor *>(source);

    if (quant_mode == QuantizationMode::NONE) {
        result.fallback_reason = QuantizeFallbackReason::TYPE_DISABLED;
        return result;
    }
    if (source->ne[2] != 1 || source->ne[3] != 1) {
        result.fallback_reason = QuantizeFallbackReason::RANK_NOT_2D;
        return result;
    }

    const enum ggml_type quant_type = quant_mode_to_ggml_type(quant_mode);
    const ggml_type_traits * src_traits = ggml_get_type_traits(source->type);
    const ggml_type_traits * dst_traits = ggml_get_type_traits(quant_type);
    if (src_traits == nullptr || dst_traits == nullptr || dst_traits->from_float_ref == nullptr) {
        result.fallback_reason = QuantizeFallbackReason::TRAIT_UNSUPPORTED;
        return result;
    }

    const int64_t row_len = source->ne[0];
    if (row_len % dst_traits->blck_size != 0) {
        result.fallback_reason = QuantizeFallbackReason::ROW_NOT_ALIGNED;
        return result;
    }

    ggml_quantize_init(quant_type);

    auto owned = std::make_unique<OwnedTensor>();
    ggml_init_params params {};
    params.mem_size = 1 * 1024 * 1024;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    owned->ctx = ggml_init(params);
    if (owned->ctx == nullptr) {
        result.fallback_reason = QuantizeFallbackReason::ALLOC_FAILED;
        return result;
    }

    owned->tensor = ggml_new_tensor_2d(owned->ctx, quant_type, source->ne[0], source->ne[1]);
    if (owned->tensor == nullptr) {
        result.fallback_reason = QuantizeFallbackReason::ALLOC_FAILED;
        return result;
    }

    owned->data.resize(ggml_nbytes(owned->tensor));
    owned->row_buffer.resize((size_t) row_len);
    owned->tensor->data = owned->data.data();

    const size_t dst_row_size = ggml_row_size(quant_type, row_len);
    const int64_t row_count = tensor_row_count(source);
    for (int64_t row = 0; row < row_count; ++row) {
        const uint8_t * src_row = static_cast<const uint8_t *>(source->data) + row * source->nb[1];
        uint8_t * dst_row = owned->data.data() + (size_t) row * dst_row_size;
        if (source->type == GGML_TYPE_F32) {
            memcpy(owned->row_buffer.data(), src_row, sizeof(float) * (size_t) row_len);
        } else if (src_traits->to_float != nullptr) {
            src_traits->to_float(src_row, owned->row_buffer.data(), row_len);
        } else {
            result.fallback_reason = QuantizeFallbackReason::TRAIT_UNSUPPORTED;
            return result;
        }
        dst_traits->from_float_ref(owned->row_buffer.data(), dst_row, row_len);
    }

    maybe_upload_owned_tensor_to_aarch64_buffer(owned.get());

    if (prof != nullptr) {
        prof->quantized_tail_tensors += 1;
        prof->quantized_tail_bytes += owned->data.size();
    }
    result.tensor = owned->tensor;
    result.owned = std::move(owned);
    result.quantized = true;
    result.fallback_reason = QuantizeFallbackReason::NONE;
    return result;
}

void record_quantize_fallback(ServerProfile * prof, QuantizeFallbackReason reason) {
    if (prof == nullptr || reason == QuantizeFallbackReason::NONE) {
        return;
    }
    prof->quantize_fallbacks += 1;
    switch (reason) {
        case QuantizeFallbackReason::TYPE_DISABLED:
            prof->quantize_fallback_type_disabled += 1;
            break;
        case QuantizeFallbackReason::RANK_NOT_2D:
            prof->quantize_fallback_rank_not_2d += 1;
            break;
        case QuantizeFallbackReason::ROW_NOT_ALIGNED:
            prof->quantize_fallback_row_not_aligned += 1;
            break;
        case QuantizeFallbackReason::TRAIT_UNSUPPORTED:
            prof->quantize_fallback_trait_unsupported += 1;
            break;
        case QuantizeFallbackReason::ALLOC_FAILED:
            prof->quantize_fallback_alloc_failed += 1;
            break;
        case QuantizeFallbackReason::NONE:
            break;
    }
}

void tensor_row_to_float(const ggml_tensor * source, int64_t row, float * dst) {
    const ggml_type_traits * src_traits = ggml_get_type_traits(source->type);
    const uint8_t * src_row = static_cast<const uint8_t *>(source->data) + row * source->nb[1];
    if (source->type == GGML_TYPE_F32) {
        memcpy(dst, src_row, sizeof(float) * (size_t) source->ne[0]);
        return;
    }
    if (src_traits != nullptr && src_traits->to_float != nullptr) {
        src_traits->to_float(src_row, dst, source->ne[0]);
        return;
    }
    throw std::runtime_error("tensor type is not convertible to float rows");
}

PreparedTensor make_tensor_from_float_rows(
        const float * rows,
        int64_t cols,
        int64_t row_count,
        QuantizationMode quant_mode,
        ServerProfile * prof,
        bool dense_weight) {
    PreparedTensor result {};

    enum ggml_type target_type = GGML_TYPE_F32;
    if (quant_mode != QuantizationMode::NONE) {
        const enum ggml_type quant_type = quant_mode_to_ggml_type(quant_mode);
        const ggml_type_traits * dst_traits = ggml_get_type_traits(quant_type);
        if (dst_traits != nullptr && dst_traits->from_float_ref != nullptr && cols % dst_traits->blck_size == 0) {
            target_type = quant_type;
            ggml_quantize_init(quant_type);
        }
    }

    auto owned = std::make_unique<OwnedTensor>();
    ggml_init_params params {};
    params.mem_size = 1 * 1024 * 1024;
    params.mem_buffer = nullptr;
    params.no_alloc = true;
    owned->ctx = ggml_init(params);
    if (owned->ctx == nullptr) {
        throw std::runtime_error("ggml_init failed for dense tensor");
    }

    owned->tensor = ggml_new_tensor_2d(owned->ctx, target_type, cols, row_count);
    if (owned->tensor == nullptr) {
        throw std::runtime_error("ggml_new_tensor_2d failed for dense tensor");
    }

    owned->data.resize(ggml_nbytes(owned->tensor));
    owned->tensor->data = owned->data.data();

    if (target_type == GGML_TYPE_F32) {
        memcpy(owned->data.data(), rows, sizeof(float) * (size_t) cols * (size_t) row_count);
    } else {
        const ggml_type_traits * dst_traits = ggml_get_type_traits(target_type);
        const size_t dst_row_size = ggml_row_size(target_type, cols);
        for (int64_t row = 0; row < row_count; ++row) {
            uint8_t * dst_row = owned->data.data() + (size_t) row * dst_row_size;
            dst_traits->from_float_ref(rows + (size_t) row * (size_t) cols, dst_row, cols);
        }
        maybe_upload_owned_tensor_to_aarch64_buffer(owned.get());
    }

    if (prof != nullptr) {
        if (dense_weight) {
            prof->dense_weight_tensors += 1;
            prof->dense_weight_bytes += owned->data.size();
        } else {
            prof->quantized_tail_tensors += 1;
            prof->quantized_tail_bytes += owned->data.size();
        }
    }

    result.tensor = owned->tensor;
    result.owned = std::move(owned);
    result.quantized = target_type != GGML_TYPE_F32;
    result.fallback_reason = QuantizeFallbackReason::NONE;
    return result;
}

PreparedTensor build_dense_tail_weight(
        const ggml_tensor * u,
        const ggml_tensor * v,
        int32_t rank_start,
        QuantizationMode quant_mode,
        ServerProfile * prof) {
    validate_svd_tensors(u, v, rank_start, v->ne[0]);

    const int64_t total_rank = v->ne[1];
    const int64_t k_remote = total_rank - rank_start;
    const int64_t input_len = v->ne[0];
    const int64_t output_len = u->ne[1];

    std::vector<float> v_rows((size_t) k_remote * (size_t) input_len);
    for (int64_t r = 0; r < k_remote; ++r) {
        tensor_row_to_float(v, rank_start + r, v_rows.data() + (size_t) r * (size_t) input_len);
    }

    std::vector<float> dense_rows((size_t) output_len * (size_t) input_len, 0.0f);
    for (int64_t out_row = 0; out_row < output_len; ++out_row) {
        std::vector<float> u_full_row((size_t) total_rank);
        tensor_row_to_float(u, out_row, u_full_row.data());
        float * dst_row = dense_rows.data() + (size_t) out_row * (size_t) input_len;
        for (int64_t r = 0; r < k_remote; ++r) {
            const float coeff = u_full_row[(size_t) rank_start + (size_t) r];
            const float * v_row = v_rows.data() + (size_t) r * (size_t) input_len;
            for (int64_t col = 0; col < input_len; ++col) {
                dst_row[col] += coeff * v_row[col];
            }
        }
    }

    return make_tensor_from_float_rows(dense_rows.data(), input_len, output_len, quant_mode, prof, true);
}

PreparedTensor build_tail_v_tensor(
        const ggml_tensor * v,
        int32_t rank_start,
        QuantizationMode quant_mode,
        ServerProfile * prof) {
    const int64_t total_rank = v->ne[1];
    const int64_t k_remote = total_rank - rank_start;
    const int64_t input_len = v->ne[0];

    std::vector<float> v_rows((size_t) k_remote * (size_t) input_len);
    for (int64_t r = 0; r < k_remote; ++r) {
        tensor_row_to_float(v, rank_start + r, v_rows.data() + (size_t) r * (size_t) input_len);
    }

    return make_tensor_from_float_rows(v_rows.data(), input_len, k_remote, quant_mode, prof, false);
}

PreparedTensor build_tail_u_tensor(
        const ggml_tensor * u,
        int32_t rank_start,
        int64_t k_remote,
        QuantizationMode quant_mode,
        ServerProfile * prof) {
    const int64_t output_len = u->ne[1];
    const int64_t total_rank = u->ne[0];
    std::vector<float> u_rows((size_t) output_len * (size_t) k_remote);
    std::vector<float> full_row((size_t) total_rank);

    for (int64_t out_row = 0; out_row < output_len; ++out_row) {
        tensor_row_to_float(u, out_row, full_row.data());
        memcpy(
            u_rows.data() + (size_t) out_row * (size_t) k_remote,
            full_row.data() + (size_t) rank_start,
            sizeof(float) * (size_t) k_remote);
    }

    return make_tensor_from_float_rows(u_rows.data(), k_remote, output_len, quant_mode, prof, false);
}

std::unique_ptr<RemoteMatExecutor> create_mat_executor_dense(
        ggml_threadpool_t threadpool,
        int n_threads,
        const ggml_tensor * u,
        const ggml_tensor * v,
        int32_t rank_start,
        int64_t input_len,
        QuantizationMode quant_mode,
        ServerProfile * prof) {
    validate_svd_tensors(u, v, rank_start, input_len);

    ggml_init_params params {};
    params.mem_size = 64 * 1024 * 1024;
    params.mem_buffer = nullptr;
    params.no_alloc = true;

    auto exec = std::make_unique<RemoteMatExecutor>();
    exec->ctx = ggml_init(params);
    if (exec->ctx == nullptr) {
        throw std::runtime_error("ggml_init failed");
    }

    exec->input_len = input_len;
    exec->output_len = u->ne[1];
    exec->input_buffer.resize((size_t) exec->input_len);
    exec->output_buffer.resize((size_t) exec->output_len);

    PreparedTensor dense_weight = build_dense_tail_weight(u, v, rank_start, choose_dense_quant_mode(u, v, quant_mode), prof);
    if (dense_weight.owned) {
        exec->owned_tensors.push_back(std::move(dense_weight.owned));
    }

    exec->input_tensor = ggml_new_tensor_2d(exec->ctx, GGML_TYPE_F32, input_len, 1);
    exec->output_tensor = ggml_mul_mat(exec->ctx, dense_weight.tensor, exec->input_tensor);
    exec->input_tensor->data = exec->input_buffer.data();
    exec->output_tensor->data = exec->output_buffer.data();

    exec->graph = ggml_new_graph(exec->ctx);
    if (exec->graph == nullptr) {
        throw std::runtime_error("ggml_new_graph failed");
    }
    ggml_build_forward_expand(exec->graph, exec->output_tensor);

    exec->plan = ggml_graph_plan(exec->graph, n_threads, threadpool);
    exec->work_data = alloc_work_data(exec->plan.work_size);
    if (exec->plan.work_size > 0 && exec->work_data == nullptr) {
        throw std::runtime_error("alloc_work_data failed");
    }
    exec->plan.work_data = static_cast<uint8_t *>(exec->work_data);

    return exec;
}

std::unique_ptr<RemoteUpGateExecutor> create_up_gate_executor_dense(
        ggml_threadpool_t threadpool,
        int n_threads,
        const ggml_tensor * up_u,
        const ggml_tensor * up_v,
        const ggml_tensor * gate_u,
        const ggml_tensor * gate_v,
        int32_t rank_start,
        int64_t input_len,
        QuantizationMode quant_mode,
        ServerProfile * prof) {
    validate_svd_tensors(up_u, up_v, rank_start, input_len);
    validate_svd_tensors(gate_u, gate_v, rank_start, input_len);

    ggml_init_params params {};
    params.mem_size = 96 * 1024 * 1024;
    params.mem_buffer = nullptr;
    params.no_alloc = true;

    auto exec = std::make_unique<RemoteUpGateExecutor>();
    exec->ctx = ggml_init(params);
    if (exec->ctx == nullptr) {
        throw std::runtime_error("ggml_init failed");
    }

    exec->input_len = input_len;
    exec->output_len = up_u->ne[1];
    exec->input_buffer.resize((size_t) exec->input_len);
    exec->output_up_buffer.resize((size_t) exec->output_len);
    exec->output_gate_buffer.resize((size_t) exec->output_len);

    PreparedTensor dense_up = build_dense_tail_weight(up_u, up_v, rank_start, choose_dense_quant_mode(up_u, up_v, quant_mode), prof);
    PreparedTensor dense_gate = build_dense_tail_weight(gate_u, gate_v, rank_start, choose_dense_quant_mode(gate_u, gate_v, quant_mode), prof);
    if (dense_up.owned) {
        exec->owned_tensors.push_back(std::move(dense_up.owned));
    }
    if (dense_gate.owned) {
        exec->owned_tensors.push_back(std::move(dense_gate.owned));
    }

    exec->input_tensor = ggml_new_tensor_2d(exec->ctx, GGML_TYPE_F32, input_len, 1);
    exec->output_up = ggml_mul_mat(exec->ctx, dense_up.tensor, exec->input_tensor);
    exec->output_gate = ggml_mul_mat(exec->ctx, dense_gate.tensor, exec->input_tensor);
    exec->input_tensor->data = exec->input_buffer.data();
    exec->output_up->data = exec->output_up_buffer.data();
    exec->output_gate->data = exec->output_gate_buffer.data();

    exec->graph = ggml_new_graph(exec->ctx);
    if (exec->graph == nullptr) {
        throw std::runtime_error("ggml_new_graph failed");
    }
    ggml_build_forward_expand(exec->graph, exec->output_up);
    ggml_build_forward_expand(exec->graph, exec->output_gate);

    exec->plan = ggml_graph_plan(exec->graph, n_threads, threadpool);
    exec->work_data = alloc_work_data(exec->plan.work_size);
    if (exec->plan.work_size > 0 && exec->work_data == nullptr) {
        throw std::runtime_error("alloc_work_data failed");
    }
    exec->plan.work_data = static_cast<uint8_t *>(exec->work_data);

    return exec;
}

std::unique_ptr<RemoteMatExecutor> create_mat_executor(
        ggml_threadpool_t threadpool,
        int n_threads,
        const ggml_tensor * u,
        const ggml_tensor * v,
        int32_t rank_start,
        int64_t input_len,
        QuantizationMode quant_mode,
        ServerProfile * prof) {
    validate_svd_tensors(u, v, rank_start, input_len);

    const int64_t total_rank = v->ne[1];
    const int64_t k_remote = total_rank - rank_start;

    ggml_init_params params {};
    params.mem_size = 64 * 1024 * 1024;
    params.mem_buffer = nullptr;
    params.no_alloc = true;

    auto exec = std::make_unique<RemoteMatExecutor>();
    exec->ctx = ggml_init(params);
    if (exec->ctx == nullptr) {
        throw std::runtime_error("ggml_init failed");
    }

    exec->input_len = input_len;
    exec->output_len = u->ne[1];
    exec->input_buffer.resize((size_t) exec->input_len);
    exec->output_buffer.resize((size_t) exec->output_len);

    exec->input_tensor = ggml_new_tensor_2d(exec->ctx, GGML_TYPE_F32, input_len, 1);
    ggml_tensor * w_shape = make_weight_shape_tensor(exec->ctx, input_len, exec->output_len);
    PreparedTensor prepared_v = build_tail_v_tensor(v, rank_start, quant_mode, prof);
    PreparedTensor prepared_u = build_tail_u_tensor(u, rank_start, k_remote, quant_mode, prof);
    if (prepared_v.owned) {
        exec->owned_tensors.push_back(std::move(prepared_v.owned));
    }
    if (prepared_u.owned) {
        exec->owned_tensors.push_back(std::move(prepared_u.owned));
    }
    exec->output_tensor = ggml_mul_mat_svd(exec->ctx, w_shape, prepared_v.tensor, prepared_u.tensor, exec->input_tensor, 0);
    exec->input_tensor->data = exec->input_buffer.data();
    exec->output_tensor->data = exec->output_buffer.data();

    exec->graph = ggml_new_graph(exec->ctx);
    if (exec->graph == nullptr) {
        throw std::runtime_error("ggml_new_graph failed");
    }
    ggml_build_forward_expand(exec->graph, exec->output_tensor);

    exec->plan = ggml_graph_plan(exec->graph, n_threads, threadpool);
    exec->work_data = alloc_work_data(exec->plan.work_size);
    if (exec->plan.work_size > 0 && exec->work_data == nullptr) {
        throw std::runtime_error("alloc_work_data failed");
    }
    exec->plan.work_data = static_cast<uint8_t *>(exec->work_data);

    return exec;
}

std::unique_ptr<RemoteUpGateExecutor> create_up_gate_executor(
        ggml_threadpool_t threadpool,
        int n_threads,
        const ggml_tensor * up_u,
        const ggml_tensor * up_v,
        const ggml_tensor * gate_u,
        const ggml_tensor * gate_v,
        int32_t rank_start,
        int64_t input_len,
        QuantizationMode quant_mode,
        ServerProfile * prof) {
    validate_svd_tensors(up_u, up_v, rank_start, input_len);
    validate_svd_tensors(gate_u, gate_v, rank_start, input_len);

    const int64_t total_rank = up_v->ne[1];
    const int64_t k_remote = total_rank - rank_start;

    ggml_init_params params {};
    params.mem_size = 96 * 1024 * 1024;
    params.mem_buffer = nullptr;
    params.no_alloc = true;

    auto exec = std::make_unique<RemoteUpGateExecutor>();
    exec->ctx = ggml_init(params);
    if (exec->ctx == nullptr) {
        throw std::runtime_error("ggml_init failed");
    }

    exec->input_len = input_len;
    exec->output_len = up_u->ne[1];
    exec->input_buffer.resize((size_t) exec->input_len);
    exec->output_up_buffer.resize((size_t) exec->output_len);
    exec->output_gate_buffer.resize((size_t) exec->output_len);

    exec->input_tensor = ggml_new_tensor_2d(exec->ctx, GGML_TYPE_F32, input_len, 1);
    ggml_tensor * up_w_shape = make_weight_shape_tensor(exec->ctx, input_len, exec->output_len);
    ggml_tensor * gate_w_shape = make_weight_shape_tensor(exec->ctx, input_len, exec->output_len);

    PreparedTensor prepared_up_v = build_tail_v_tensor(up_v, rank_start, quant_mode, prof);
    PreparedTensor prepared_up_u = build_tail_u_tensor(up_u, rank_start, k_remote, quant_mode, prof);
    if (prepared_up_v.owned) {
        exec->owned_tensors.push_back(std::move(prepared_up_v.owned));
    }
    if (prepared_up_u.owned) {
        exec->owned_tensors.push_back(std::move(prepared_up_u.owned));
    }
    exec->output_up = ggml_mul_mat_svd(exec->ctx, up_w_shape, prepared_up_v.tensor, prepared_up_u.tensor, exec->input_tensor, 0);

    PreparedTensor prepared_gate_v = build_tail_v_tensor(gate_v, rank_start, quant_mode, prof);
    PreparedTensor prepared_gate_u = build_tail_u_tensor(gate_u, rank_start, k_remote, quant_mode, prof);
    if (prepared_gate_v.owned) {
        exec->owned_tensors.push_back(std::move(prepared_gate_v.owned));
    }
    if (prepared_gate_u.owned) {
        exec->owned_tensors.push_back(std::move(prepared_gate_u.owned));
    }
    exec->output_gate = ggml_mul_mat_svd(exec->ctx, gate_w_shape, prepared_gate_v.tensor, prepared_gate_u.tensor, exec->input_tensor, 0);
    exec->input_tensor->data = exec->input_buffer.data();
    exec->output_up->data = exec->output_up_buffer.data();
    exec->output_gate->data = exec->output_gate_buffer.data();

    exec->graph = ggml_new_graph(exec->ctx);
    if (exec->graph == nullptr) {
        throw std::runtime_error("ggml_new_graph failed");
    }
    ggml_build_forward_expand(exec->graph, exec->output_up);
    ggml_build_forward_expand(exec->graph, exec->output_gate);

    exec->plan = ggml_graph_plan(exec->graph, n_threads, threadpool);
    exec->work_data = alloc_work_data(exec->plan.work_size);
    if (exec->plan.work_size > 0 && exec->work_data == nullptr) {
        throw std::runtime_error("alloc_work_data failed");
    }
    exec->plan.work_data = static_cast<uint8_t *>(exec->work_data);

    return exec;
}

std::unique_ptr<RemoteFfnExecutor> create_ffn_executor(
        ggml_threadpool_t threadpool,
        int n_threads,
        const ggml_tensor * up_u,
        const ggml_tensor * up_v,
        const ggml_tensor * gate_u,
        const ggml_tensor * gate_v,
        const ggml_tensor * down_u,
        const ggml_tensor * down_v,
        int64_t input_len,
        QuantizationMode quant_mode,
        ServerProfile * prof) {
    validate_svd_tensors(up_u, up_v, 0, input_len);
    validate_svd_tensors(gate_u, gate_v, 0, input_len);
    validate_svd_tensors(down_u, down_v, 0, up_u->ne[1]);

    ggml_init_params params {};
    params.mem_size = 256 * 1024 * 1024;
    params.mem_buffer = nullptr;
    params.no_alloc = false;

    auto exec = std::make_unique<RemoteFfnExecutor>();
    exec->ctx = ggml_init(params);
    if (exec->ctx == nullptr) {
        throw std::runtime_error("ggml_init failed");
    }

    exec->input_len = input_len;
    exec->output_len = down_u->ne[1];
    exec->input_buffer.resize((size_t) exec->input_len);
    exec->output_buffer.resize((size_t) exec->output_len);

    exec->input_tensor = ggml_new_tensor_2d(exec->ctx, GGML_TYPE_F32, input_len, 1);

    PreparedTensor prepared_up_v = maybe_quantize_tensor(up_v, quant_mode, prof);
    PreparedTensor prepared_up_u = maybe_quantize_tensor(up_u, quant_mode, prof);
    PreparedTensor prepared_gate_v = maybe_quantize_tensor(gate_v, quant_mode, prof);
    PreparedTensor prepared_gate_u = maybe_quantize_tensor(gate_u, quant_mode, prof);
    PreparedTensor prepared_down_v = maybe_quantize_tensor(down_v, quant_mode, prof);
    PreparedTensor prepared_down_u = maybe_quantize_tensor(down_u, quant_mode, prof);
    record_quantize_fallback(prof, prepared_up_v.fallback_reason);
    record_quantize_fallback(prof, prepared_up_u.fallback_reason);
    record_quantize_fallback(prof, prepared_gate_v.fallback_reason);
    record_quantize_fallback(prof, prepared_gate_u.fallback_reason);
    record_quantize_fallback(prof, prepared_down_v.fallback_reason);
    record_quantize_fallback(prof, prepared_down_u.fallback_reason);
    if (prepared_up_v.owned) exec->owned_tensors.push_back(std::move(prepared_up_v.owned));
    if (prepared_up_u.owned) exec->owned_tensors.push_back(std::move(prepared_up_u.owned));
    if (prepared_gate_v.owned) exec->owned_tensors.push_back(std::move(prepared_gate_v.owned));
    if (prepared_gate_u.owned) exec->owned_tensors.push_back(std::move(prepared_gate_u.owned));
    if (prepared_down_v.owned) exec->owned_tensors.push_back(std::move(prepared_down_v.owned));
    if (prepared_down_u.owned) exec->owned_tensors.push_back(std::move(prepared_down_u.owned));

    ggml_tensor * up_shape = make_weight_shape_tensor(exec->ctx, input_len, up_u->ne[1]);
    ggml_tensor * gate_shape = make_weight_shape_tensor(exec->ctx, input_len, gate_u->ne[1]);
    ggml_tensor * down_shape = make_weight_shape_tensor(exec->ctx, up_u->ne[1], down_u->ne[1]);

    ggml_tensor * up = ggml_mul_mat_svd(exec->ctx, up_shape, prepared_up_v.tensor, prepared_up_u.tensor, exec->input_tensor, 0);
    ggml_tensor * gate = ggml_mul_mat_svd(exec->ctx, gate_shape, prepared_gate_v.tensor, prepared_gate_u.tensor, exec->input_tensor, 0);
    ggml_tensor * gate_silu = ggml_silu(exec->ctx, gate);
    ggml_tensor * fused = ggml_mul(exec->ctx, gate_silu, up);
    exec->output_tensor = ggml_mul_mat_svd(exec->ctx, down_shape, prepared_down_v.tensor, prepared_down_u.tensor, fused, 0);

    exec->graph = ggml_new_graph(exec->ctx);
    if (exec->graph == nullptr) {
        throw std::runtime_error("ggml_new_graph failed");
    }
    ggml_build_forward_expand(exec->graph, exec->output_tensor);

    exec->plan = ggml_graph_plan(exec->graph, n_threads, threadpool);
    exec->work_data = alloc_work_data(exec->plan.work_size);
    if (exec->plan.work_size > 0 && exec->work_data == nullptr) {
        throw std::runtime_error("alloc_work_data failed");
    }
    exec->plan.work_data = static_cast<uint8_t *>(exec->work_data);

    return exec;
}

void run_mat_executor(
        RemoteMatExecutor & exec,
        const std::vector<float> & input) {
    if ((int64_t) input.size() != exec.input_len) {
        throw std::runtime_error("unexpected input length");
    }

    memcpy(exec.input_buffer.data(), input.data(), sizeof(float) * input.size());
    const ggml_status status = ggml_graph_compute(exec.graph, &exec.plan);
    if (status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("ggml_graph_compute failed");
    }
}

void run_ffn_executor(
        RemoteFfnExecutor & exec,
        const std::vector<float> & input) {
    if ((int64_t) input.size() != exec.input_len) {
        throw std::runtime_error("unexpected input length");
    }

    memcpy(exec.input_tensor->data, input.data(), sizeof(float) * input.size());
    const ggml_status status = ggml_graph_compute(exec.graph, &exec.plan);
    if (status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("ggml_graph_compute failed");
    }
    memcpy(exec.output_buffer.data(), exec.output_tensor->data, sizeof(float) * exec.output_buffer.size());
}

void run_up_gate_executor(
        RemoteUpGateExecutor & exec,
        const std::vector<float> & input) {
    if ((int64_t) input.size() != exec.input_len) {
        throw std::runtime_error("unexpected input length");
    }

    memcpy(exec.input_buffer.data(), input.data(), sizeof(float) * input.size());
    const ggml_status status = ggml_graph_compute(exec.graph, &exec.plan);
    if (status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("ggml_graph_compute failed");
    }
}

} // namespace

int main(int argc, char ** argv) {
    const std::string model_path = argc > 1
        ? argv[1]
        : "/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/gguf_models/qwen.gguf.sort_svd.compact.gguf";
    const int port = argc > 2 ? std::stoi(argv[2]) : 7788;
    const int n_threads = argc > 3 ? std::stoi(argv[3]) : 8;
    const std::string quant_arg = argc > 4 ? argv[4] : "off";
    const std::string executor_arg = argc > 5 ? argv[5] : "svd";
    const QuantizationMode quant_mode = parse_quantization_mode(quant_arg);
    const ExecutorMode executor_mode = parse_executor_mode(executor_arg);

    ggml_cpu_init();

#ifdef GGML_USE_OPENMP
    omp_set_num_threads(n_threads);
#endif

    ggml_threadpool_params threadpool_params = ggml_threadpool_params_default(n_threads);
    ggml_threadpool_t threadpool = ggml_threadpool_new(&threadpool_params);
    if (threadpool == nullptr) {
        std::cerr << "ggml_threadpool_new failed" << std::endl;
        return 1;
    }

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
#ifdef _WIN32
    model_params.use_mmap = false;
#endif

    std::cout << "loading mobile-side model: " << model_path << std::endl;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (model == nullptr) {
        std::cerr << "failed to load model" << std::endl;
        ggml_threadpool_free(threadpool);
        return 1;
    }

    if (!ensure_winsock_initialized()) {
        std::cerr << "winsock initialization failed" << std::endl;
        llama_model_free(model);
        ggml_threadpool_free(threadpool);
        return 1;
    }

    const int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd < 0) {
        std::cerr << "socket failed: " << strerror(errno) << std::endl;
        llama_model_free(model);
        ggml_threadpool_free(threadpool);
        return 1;
    }

    int one = 1;
    setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char *>(&one), sizeof(one));
    setsockopt(listen_fd, IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<const char *>(&one), sizeof(one));

    sockaddr_in addr {};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(static_cast<uint16_t>(port));

    if (bind(listen_fd, reinterpret_cast<const sockaddr *>(&addr), sizeof(addr)) != 0 ||
        listen(listen_fd, 1) != 0) {
        std::cerr << "bind/listen failed: " << strerror(errno) << std::endl;
        close_socket(listen_fd);
        llama_model_free(model);
        ggml_threadpool_free(threadpool);
        return 1;
    }

    std::cout << "svd_mobile_server listening on 0.0.0.0:" << port
              << " threads=" << n_threads
              << " quant_mode=" << quant_mode_name(quant_mode)
              << " executor_mode=" << executor_mode_name(executor_mode)
              << " backend=" << backend_path_name() << std::endl;

    std::unordered_map<MatExecutorKey, std::unique_ptr<RemoteMatExecutor>, MatExecutorKeyHash> mat_executors;
    std::unordered_map<UpGateExecutorKey, std::unique_ptr<RemoteUpGateExecutor>, UpGateExecutorKeyHash> up_gate_executors;
    std::unordered_map<int32_t, std::unique_ptr<RemoteFfnExecutor>> ffn_executors;

    while (true) {
        const int client_fd = accept(listen_fd, nullptr, nullptr);
        if (client_fd < 0) {
            std::cerr << "accept failed: " << strerror(errno) << std::endl;
            continue;
        }
        setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<const char *>(&one), sizeof(one));
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

            if (prof.debug_logged_requests < 8 || req.request_kind == REQ_FFN) {
                std::cerr
                    << "[svd-offload-server-recv] kind=" << req.request_kind
                    << " layer_id=" << req.layer_id
                    << " op_id=" << req.op_id
                    << " rank_start=" << req.rank_start
                    << " input_len=" << req.input_len
                    << std::endl;
            }

            WireResponse rsp { kMagic, kVersion, 0, 0, 0 };
            const float * output_data = nullptr;
            size_t output_len = 0;
            const float * output_aux_data = nullptr;
            size_t output_aux_len = 0;
            try {
                if (req.request_kind == REQ_UP_GATE) {
                    prof.requests_up_gate++;
                    const UpGateExecutorKey key { req.layer_id, req.rank_start };
                    const auto up = get_svd_tensors(*model, req.layer_id, SVD_OP_UP);
                    const auto gate = get_svd_tensors(*model, req.layer_id, SVD_OP_GATE);
                    const ExecutorMode effective_mode = resolve_executor_mode(executor_mode, up.u, up.v, quant_mode);
                    auto & exec = up_gate_executors[key];
                    if (!exec) {
                        const uint64_t t0 = now_us();
                        if (effective_mode == ExecutorMode::DENSE_TAIL) {
                            prof.dense_up_gate_cache_miss++;
                            exec = create_up_gate_executor_dense(threadpool, n_threads, up.u, up.v, gate.u, gate.v, req.rank_start, req.input_len, quant_mode, &prof);
                        } else {
                            prof.up_gate_cache_miss++;
                            exec = create_up_gate_executor(threadpool, n_threads, up.u, up.v, gate.u, gate.v, req.rank_start, req.input_len, quant_mode, &prof);
                        }
                        prof.create_up_gate_us += now_us() - t0;
                    }
                    const uint64_t t0 = now_us();
                    run_up_gate_executor(*exec, input);
                    prof.run_up_gate_us += now_us() - t0;
                    rsp.output_len = static_cast<int32_t>(exec->output_up_buffer.size());
                    rsp.output_len_aux = static_cast<int32_t>(exec->output_gate_buffer.size());
                    output_data = exec->output_up_buffer.data();
                    output_len = exec->output_up_buffer.size();
                    output_aux_data = exec->output_gate_buffer.data();
                    output_aux_len = exec->output_gate_buffer.size();
                    if (prof.debug_logged_requests < 2) {
                        const TensorStats input_stats = analyze_float_data(input.data(), input.size());
                        const TensorStats up_stats = analyze_float_data(output_data, output_len);
                        const TensorStats gate_stats = analyze_float_data(output_aux_data, output_aux_len);
                        std::cerr
                            << "[svd-offload-server-debug] kind=up_gate"
                            << " layer_id=" << req.layer_id
                            << " op_id=" << req.op_id
                            << " rank_start=" << req.rank_start
                            << " input_len=" << req.input_len
                            << " output_len=" << output_len
                            << " output_aux_len=" << output_aux_len
                            << " u_type=" << ggml_type_name(up.u->type)
                            << " v_type=" << ggml_type_name(up.v->type)
                            << " executor_mode=" << executor_mode_name(effective_mode)
                            << " backend=" << backend_path_name()
                            << " input_sum_abs=" << input_stats.sum_abs
                            << " input_max_abs=" << input_stats.max_abs
                            << " input_has_nan=" << (input_stats.has_nan ? 1 : 0)
                            << " up_sum_abs=" << up_stats.sum_abs
                            << " up_max_abs=" << up_stats.max_abs
                            << " up_has_nan=" << (up_stats.has_nan ? 1 : 0)
                            << " gate_sum_abs=" << gate_stats.sum_abs
                            << " gate_max_abs=" << gate_stats.max_abs
                            << " gate_has_nan=" << (gate_stats.has_nan ? 1 : 0)
                            << std::endl;
                        prof.debug_logged_requests++;
                    }
                    accumulate_output_stats(output_data, output_len, prof.output_up_sum_abs, prof.output_up_max_abs, prof.output_up_zero_vectors);
                    accumulate_output_stats(output_aux_data, output_aux_len, prof.output_gate_sum_abs, prof.output_gate_max_abs, prof.output_gate_zero_vectors);
                } else if (req.request_kind == REQ_FFN) {
                    prof.requests_ffn++;
                    const auto up = get_svd_tensors(*model, req.layer_id, SVD_OP_UP);
                    const auto gate = get_svd_tensors(*model, req.layer_id, SVD_OP_GATE);
                    const auto down = get_svd_tensors(*model, req.layer_id, SVD_OP_DOWN);
                    auto & exec = ffn_executors[req.layer_id];
                    if (!exec) {
                        const uint64_t t0 = now_us();
                        prof.ffn_cache_miss++;
                        exec = create_ffn_executor(
                            threadpool,
                            n_threads,
                            up.u, up.v,
                            gate.u, gate.v,
                            down.u, down.v,
                            req.input_len,
                            quant_mode,
                            &prof);
                        prof.create_ffn_us += now_us() - t0;
                    }
                    const uint64_t t0 = now_us();
                    run_ffn_executor(*exec, input);
                    prof.run_ffn_us += now_us() - t0;
                    rsp.output_len = static_cast<int32_t>(exec->output_buffer.size());
                    output_data = exec->output_buffer.data();
                    output_len = exec->output_buffer.size();
                    if (prof.debug_logged_requests < 2) {
                        const TensorStats input_stats = analyze_float_data(input.data(), input.size());
                        const TensorStats out_stats = analyze_float_data(output_data, output_len);
                        std::cerr
                            << "[svd-offload-server-debug] kind=ffn"
                            << " layer_id=" << req.layer_id
                            << " input_len=" << req.input_len
                            << " output_len=" << output_len
                            << " up_u_type=" << ggml_type_name(up.u->type)
                            << " up_v_type=" << ggml_type_name(up.v->type)
                            << " down_u_type=" << ggml_type_name(down.u->type)
                            << " down_v_type=" << ggml_type_name(down.v->type)
                            << " backend=" << backend_path_name()
                            << " input_sum_abs=" << input_stats.sum_abs
                            << " input_max_abs=" << input_stats.max_abs
                            << " input_has_nan=" << (input_stats.has_nan ? 1 : 0)
                            << " output_sum_abs=" << out_stats.sum_abs
                            << " output_max_abs=" << out_stats.max_abs
                            << " output_has_nan=" << (out_stats.has_nan ? 1 : 0)
                            << std::endl;
                        prof.debug_logged_requests++;
                    }
                    accumulate_output_stats(output_data, output_len, prof.output_ffn_sum_abs, prof.output_ffn_max_abs, prof.output_ffn_zero_vectors);
                } else {
                    const bool is_down = req.op_id == SVD_OP_DOWN;
                    if (is_down) {
                        prof.requests_mat_down++;
                    } else {
                        prof.requests_mat_other++;
                    }
                    const MatExecutorKey key { req.layer_id, req.op_id, req.rank_start };
                    const auto mat = get_svd_tensors(*model, req.layer_id, req.op_id);
                    const ExecutorMode effective_mode = resolve_executor_mode(executor_mode, mat.u, mat.v, quant_mode);
                    auto & exec = mat_executors[key];
                    if (!exec) {
                        const uint64_t t0 = now_us();
                        if (effective_mode == ExecutorMode::DENSE_TAIL) {
                            exec = create_mat_executor_dense(threadpool, n_threads, mat.u, mat.v, req.rank_start, req.input_len, quant_mode, &prof);
                        } else {
                            exec = create_mat_executor(threadpool, n_threads, mat.u, mat.v, req.rank_start, req.input_len, quant_mode, &prof);
                        }
                        const uint64_t dt = now_us() - t0;
                        if (is_down) {
                            if (effective_mode == ExecutorMode::DENSE_TAIL) {
                                prof.dense_mat_down_cache_miss++;
                            } else {
                                prof.mat_down_cache_miss++;
                            }
                            prof.create_mat_down_us += dt;
                        } else {
                            if (effective_mode == ExecutorMode::DENSE_TAIL) {
                                prof.dense_mat_other_cache_miss++;
                            } else {
                                prof.mat_other_cache_miss++;
                            }
                            prof.create_mat_other_us += dt;
                        }
                    }
                    const uint64_t t0 = now_us();
                    run_mat_executor(*exec, input);
                    const uint64_t dt = now_us() - t0;
                    if (is_down) {
                        prof.run_mat_down_us += dt;
                    } else {
                        prof.run_mat_other_us += dt;
                    }
                    rsp.output_len = static_cast<int32_t>(exec->output_buffer.size());
                    output_data = exec->output_buffer.data();
                    output_len = exec->output_buffer.size();
                    if (prof.debug_logged_requests < 2) {
                        const TensorStats input_stats = analyze_float_data(input.data(), input.size());
                        const TensorStats out_stats = analyze_float_data(output_data, output_len);
                        std::cerr
                            << "[svd-offload-server-debug] kind=mat"
                            << " layer_id=" << req.layer_id
                            << " op_id=" << req.op_id
                            << " rank_start=" << req.rank_start
                            << " input_len=" << req.input_len
                            << " output_len=" << output_len
                            << " u_type=" << ggml_type_name(mat.u->type)
                            << " v_type=" << ggml_type_name(mat.v->type)
                            << " executor_mode=" << executor_mode_name(effective_mode)
                            << " backend=" << backend_path_name()
                            << " input_sum_abs=" << input_stats.sum_abs
                            << " input_max_abs=" << input_stats.max_abs
                            << " input_has_nan=" << (input_stats.has_nan ? 1 : 0)
                            << " output_sum_abs=" << out_stats.sum_abs
                            << " output_max_abs=" << out_stats.max_abs
                            << " output_has_nan=" << (out_stats.has_nan ? 1 : 0)
                            << std::endl;
                        prof.debug_logged_requests++;
                    }
                    if (is_down) {
                        accumulate_output_stats(output_data, output_len, prof.output_down_sum_abs, prof.output_down_max_abs, prof.output_down_zero_vectors);
                    } else {
                        accumulate_output_stats(output_data, output_len, prof.output_other_sum_abs, prof.output_other_max_abs, prof.output_other_zero_vectors);
                    }
                }
            } catch (const std::exception & e) {
                rsp.status = -1;
                std::cerr << "request failed: " << e.what() << std::endl;
            }

            if (!send_all(client_fd, &rsp, sizeof(rsp))) {
                break;
            }
            if (rsp.status == 0) {
                if (!send_all(client_fd, output_data, sizeof(float) * output_len)) {
                    break;
                }
                if (rsp.output_len_aux > 0 &&
                    !send_all(client_fd, output_aux_data, sizeof(float) * output_aux_len)) {
                    break;
                }
            }
        }

        std::cerr
            << "[svd-offload-server] up_gate_req=" << prof.requests_up_gate
            << " down_req=" << prof.requests_mat_down
            << " other_mat_req=" << prof.requests_mat_other
            << " ffn_req=" << prof.requests_ffn
            << " up_gate_miss=" << prof.up_gate_cache_miss
            << " down_miss=" << prof.mat_down_cache_miss
            << " other_miss=" << prof.mat_other_cache_miss
            << " ffn_miss=" << prof.ffn_cache_miss
            << " dense_up_gate_miss=" << prof.dense_up_gate_cache_miss
            << " dense_down_miss=" << prof.dense_mat_down_cache_miss
            << " dense_other_miss=" << prof.dense_mat_other_cache_miss
            << " dense_ffn_miss=" << prof.dense_ffn_cache_miss
            << " create_up_gate=" << (prof.create_up_gate_us / 1000.0) << " ms"
            << " create_down=" << (prof.create_mat_down_us / 1000.0) << " ms"
            << " create_other=" << (prof.create_mat_other_us / 1000.0) << " ms"
            << " create_ffn=" << (prof.create_ffn_us / 1000.0) << " ms"
            << " run_up_gate=" << (prof.run_up_gate_us / 1000.0) << " ms"
            << " run_down=" << (prof.run_mat_down_us / 1000.0) << " ms"
            << " run_other=" << (prof.run_mat_other_us / 1000.0) << " ms"
            << " run_ffn=" << (prof.run_ffn_us / 1000.0) << " ms"
            << " dense_weight_tensors=" << prof.dense_weight_tensors
            << " dense_weight_mib=" << (prof.dense_weight_bytes / 1024.0 / 1024.0)
            << " quant_tail_tensors=" << prof.quantized_tail_tensors
            << " quant_tail_mib=" << (prof.quantized_tail_bytes / 1024.0 / 1024.0)
            << " quant_fallbacks=" << prof.quantize_fallbacks
            << " fallback_type_disabled=" << prof.quantize_fallback_type_disabled
            << " fallback_rank_not_2d=" << prof.quantize_fallback_rank_not_2d
            << " fallback_row_not_aligned=" << prof.quantize_fallback_row_not_aligned
            << " fallback_trait_unsupported=" << prof.quantize_fallback_trait_unsupported
            << " fallback_alloc_failed=" << prof.quantize_fallback_alloc_failed
            << std::endl;
        std::cerr
            << "[svd-offload-server-output] "
            << "up_sum_abs=" << prof.output_up_sum_abs
            << " up_max_abs=" << prof.output_up_max_abs
            << " up_zero=" << prof.output_up_zero_vectors
            << " gate_sum_abs=" << prof.output_gate_sum_abs
            << " gate_max_abs=" << prof.output_gate_max_abs
            << " gate_zero=" << prof.output_gate_zero_vectors
            << " down_sum_abs=" << prof.output_down_sum_abs
            << " down_max_abs=" << prof.output_down_max_abs
            << " down_zero=" << prof.output_down_zero_vectors
            << " other_sum_abs=" << prof.output_other_sum_abs
            << " other_max_abs=" << prof.output_other_max_abs
            << " other_zero=" << prof.output_other_zero_vectors
            << " ffn_sum_abs=" << prof.output_ffn_sum_abs
            << " ffn_max_abs=" << prof.output_ffn_max_abs
            << " ffn_zero=" << prof.output_ffn_zero_vectors
            << std::endl;
        std::cerr
            << "[svd-offload-server-stage-profile] create_total="
            << ((prof.create_up_gate_us + prof.create_mat_down_us + prof.create_mat_other_us + prof.create_ffn_us) / 1000.0) << " ms"
            << " remote_compute_total="
            << ((prof.run_up_gate_us + prof.run_mat_down_us + prof.run_mat_other_us + prof.run_ffn_us) / 1000.0) << " ms"
            << std::endl;
        std::cout << "client disconnected" << std::endl;
        close_socket(client_fd);
    }

    close_socket(listen_fd);
    up_gate_executors.clear();
    mat_executors.clear();
    ffn_executors.clear();
    llama_model_free(model);
    ggml_threadpool_free(threadpool);
    return 0;
}
