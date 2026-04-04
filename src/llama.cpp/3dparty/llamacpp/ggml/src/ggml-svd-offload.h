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

enum ggml_svd_offload_request_kind {
    GGML_SVD_OFFLOAD_REQ_MAT = 0,
    GGML_SVD_OFFLOAD_REQ_UP_GATE = 1,
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
    int32_t request_kind;
    int32_t layer_id;
    int32_t op_id;
    int32_t rank_start;
    uint64_t input_hash;
    uint64_t start_us;
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

bool ggml_svd_offload_begin_up_gate_request(
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

bool ggml_svd_offload_finish_up_gate_request(
        struct ggml_svd_offload_request_handle * handle,
        float * output,
        int64_t output_len,
        int64_t paired_output_len);

bool ggml_svd_offload_has_cached_up(
        int32_t layer_id,
        int64_t rank_start,
        const float * input,
        int64_t input_len,
        int64_t output_len);

bool ggml_svd_offload_take_cached_up(
        int32_t layer_id,
        int64_t rank_start,
        const float * input,
        int64_t input_len,
        float * output,
        int64_t output_len);

bool ggml_svd_offload_has_cached_gate(
        int32_t layer_id,
        int64_t rank_start,
        const float * input,
        int64_t input_len,
        int64_t output_len);

bool ggml_svd_offload_take_cached_gate(
        int32_t layer_id,
        int64_t rank_start,
        const float * input,
        int64_t input_len,
        float * output,
        int64_t output_len);

void ggml_svd_local_profile_print_and_reset(void);

void ggml_svd_offload_close_client(void);

#ifdef __cplusplus
}
#endif
