#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum ggml_svd_op_kind {
    GGML_SVD_OP_UP = 0,
    GGML_SVD_OP_GATE = 1,
    GGML_SVD_OP_DOWN = 2,
};

struct ggml_svd_offload_client_config {
    bool enabled;
    const char * host;
    uint16_t port;
    int32_t timeout_ms;
};

struct ggml_svd_offload_request_handle {
    int32_t socket_fd;
    int32_t response_ready;
};

void ggml_svd_offload_set_client_config(const struct ggml_svd_offload_client_config * config);
bool ggml_svd_offload_client_enabled(void);

bool ggml_svd_offload_begin_request(
        int32_t layer_id,
        int32_t op_id,
        float offload_rate,
        int64_t rank_start,
        const float * input,
        int64_t input_len,
        struct ggml_svd_offload_request_handle * handle);

bool ggml_svd_offload_finish_request(
        struct ggml_svd_offload_request_handle * handle,
        float * output,
        int64_t output_len);

void ggml_svd_offload_close_client(void);

#ifdef __cplusplus
}
#endif
