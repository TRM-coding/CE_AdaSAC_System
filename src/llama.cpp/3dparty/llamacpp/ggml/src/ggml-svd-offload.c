#include "ggml-svd-offload.h"

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#endif
#include <errno.h>
#include <pthread.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#define GGML_SVD_SELECT_NFDS(fd) 0
#else
#define GGML_SVD_SELECT_NFDS(fd) ((fd) + 1)
#endif

struct ggml_svd_offload_wire_request {
    uint32_t magic;
    uint32_t version;
    int32_t request_kind;
    int32_t layer_id;
    int32_t op_id;
    float offload_rate;
    int32_t rank_start;
    int32_t input_len;
};

struct ggml_svd_offload_wire_response {
    uint32_t magic;
    uint32_t version;
    int32_t status;
    int32_t output_len;
    int32_t output_len_aux;
};

static struct {
    bool enabled;
    char host[128];
    uint16_t port;
    int32_t timeout_ms;
    int socket_fd;
    pthread_mutex_t mutex;
} g_svd_client = {
    false,
    { 0 },
    0,
    0,
    -1,
    PTHREAD_MUTEX_INITIALIZER,
};

struct ggml_svd_pair_cache_entry {
    int32_t layer_id;
    int32_t rank_start;
    int32_t output_len;
    uint64_t input_hash;
    float * up_data;
    float * gate_data;
    bool valid_up;
    bool valid_gate;
};

enum {
    GGML_SVD_GATE_CACHE_SLOTS = 64,
};

static struct ggml_svd_pair_cache_entry g_svd_pair_cache[GGML_SVD_GATE_CACHE_SLOTS] = { 0 };

static struct {
    uint64_t requests_up_gate;
    uint64_t requests_down;
    uint64_t requests_other_mat;
    uint64_t requests_ffn;
    uint64_t gate_cache_hits;
    uint64_t gate_cache_checks;
    uint64_t gate_cache_miss_invalid;
    uint64_t gate_cache_miss_layer;
    uint64_t gate_cache_miss_rank;
    uint64_t gate_cache_miss_hash;
    uint64_t gate_cache_miss_output_len;
    uint64_t gate_cache_miss_data;
    uint64_t wait_up_gate_us;
    uint64_t wait_down_us;
    uint64_t wait_other_mat_us;
    uint64_t wait_ffn_us;
    uint64_t send_up_gate_us;
    uint64_t send_down_us;
    uint64_t send_other_mat_us;
    uint64_t send_ffn_us;
    uint64_t finish_fail_up_gate;
    uint64_t finish_fail_down;
    uint64_t finish_fail_other_mat;
    uint64_t finish_fail_ffn;
    uint64_t cache_take_fail_up;
    uint64_t cache_take_fail_gate;
    double recv_up_sum_abs;
    double recv_gate_sum_abs;
    double recv_down_sum_abs;
    double recv_other_sum_abs;
    double recv_ffn_sum_abs;
    float recv_up_max_abs;
    float recv_gate_max_abs;
    float recv_down_max_abs;
    float recv_other_max_abs;
    float recv_ffn_max_abs;
    uint64_t recv_up_zero_vectors;
    uint64_t recv_gate_zero_vectors;
    uint64_t recv_down_zero_vectors;
    uint64_t recv_other_zero_vectors;
    uint64_t recv_ffn_zero_vectors;
} g_svd_profile = { 0 };

enum {
    GGML_SVD_OFFLOAD_MAGIC = 0x5344564fU,
    GGML_SVD_OFFLOAD_VERSION = 2,
};

#ifdef _WIN32
static bool ggml_svd_winsock_initialized(void) {
    static bool initialized = false;
    static bool ok = false;
    if (!initialized) {
        WSADATA wsa_data;
        ok = WSAStartup(MAKEWORD(2, 2), &wsa_data) == 0;
        initialized = true;
    }
    return ok;
}

static int ggml_svd_close_socket(int fd) {
    return closesocket((SOCKET) fd);
}
#else
static bool ggml_svd_winsock_initialized(void) {
    return true;
}

static int ggml_svd_close_socket(int fd) {
    return close(fd);
}
#endif

static uint64_t ggml_svd_now_us(void) {
#ifdef _WIN32
    return (uint64_t) GetTickCount64() * 1000ULL;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t) ts.tv_sec * 1000000ULL + (uint64_t) ts.tv_nsec / 1000ULL;
#endif
}

static void ggml_svd_profile_accum_output(
        const float * data,
        int64_t len,
        double * sum_abs_total,
        float * max_abs_total,
        uint64_t * zero_vectors) {
    if (data == NULL || len <= 0) {
        return;
    }

    double sum_abs = 0.0;
    float max_abs = 0.0f;
    for (int64_t i = 0; i < len; ++i) {
        const float v = fabsf(data[i]);
        sum_abs += (double) v;
        if (v > max_abs) {
            max_abs = v;
        }
    }

    *sum_abs_total += sum_abs;
    if (max_abs > *max_abs_total) {
        *max_abs_total = max_abs;
    }
    if (sum_abs == 0.0) {
        *zero_vectors += 1;
    }
}

static void ggml_svd_pair_cache_reset_entry_locked(struct ggml_svd_pair_cache_entry * entry) {
    entry->valid_up = false;
    entry->valid_gate = false;
    entry->layer_id = -1;
    entry->rank_start = -1;
    entry->output_len = 0;
    entry->input_hash = 0;
}

static void ggml_svd_gate_cache_clear_locked(void) {
    for (int i = 0; i < GGML_SVD_GATE_CACHE_SLOTS; ++i) {
        ggml_svd_pair_cache_reset_entry_locked(&g_svd_pair_cache[i]);
    }
}

static bool ggml_svd_pair_cache_resize_locked(float ** data, int32_t output_len) {
    if (output_len <= 0) {
        return false;
    }
    if (*data != NULL) {
        return true;
    }
    float * new_data = (float *) malloc(sizeof(float) * (size_t) output_len);
    if (new_data == NULL) {
        return false;
    }
    *data = new_data;
    return true;
}

static bool ggml_svd_pair_cache_matches(
        const struct ggml_svd_pair_cache_entry * entry,
        int32_t layer_id,
        int64_t rank_start,
        uint64_t input_hash,
        int64_t output_len) {
    return (entry->valid_up || entry->valid_gate) &&
        entry->layer_id == layer_id &&
        entry->rank_start == rank_start &&
        entry->input_hash == input_hash &&
        entry->output_len == output_len;
}

static struct ggml_svd_pair_cache_entry * ggml_svd_pair_cache_find_locked(
        int32_t layer_id,
        int64_t rank_start,
        uint64_t input_hash) {
    for (int i = 0; i < GGML_SVD_GATE_CACHE_SLOTS; ++i) {
        struct ggml_svd_pair_cache_entry * entry = &g_svd_pair_cache[i];
        if ((entry->valid_up || entry->valid_gate) &&
            entry->layer_id == layer_id &&
            entry->rank_start == rank_start &&
            entry->input_hash == input_hash) {
            return entry;
        }
    }
    return NULL;
}

static struct ggml_svd_pair_cache_entry * ggml_svd_pair_cache_prepare_slot_locked(
        int32_t layer_id,
        int32_t rank_start,
        uint64_t input_hash) {
    for (int i = 0; i < GGML_SVD_GATE_CACHE_SLOTS; ++i) {
        struct ggml_svd_pair_cache_entry * entry = &g_svd_pair_cache[i];
        if ((entry->valid_up || entry->valid_gate) &&
            entry->layer_id == layer_id &&
            entry->rank_start == rank_start &&
            entry->input_hash == input_hash) {
            return entry;
        }
    }
    for (int i = 0; i < GGML_SVD_GATE_CACHE_SLOTS; ++i) {
        if (!g_svd_pair_cache[i].valid_up && !g_svd_pair_cache[i].valid_gate) {
            return &g_svd_pair_cache[i];
        }
    }
    return &g_svd_pair_cache[0];
}

static bool ggml_svd_send_all(int fd, const void * data, size_t size) {
    const char * ptr = (const char *) data;
    while (size > 0) {
        int flags = 0;
#ifdef MSG_NOSIGNAL
        flags |= MSG_NOSIGNAL;
#endif
        const int written = send(fd, ptr, (int) size, flags);
        if (written <= 0) {
            return false;
        }
        ptr += written;
        size -= (size_t) written;
    }
    return true;
}

static bool ggml_svd_recv_all(int fd, void * data, size_t size) {
    char * ptr = (char *) data;
    while (size > 0) {
        const int nread = recv(fd, ptr, (int) size, 0);
        if (nread <= 0) {
            return false;
        }
        ptr += nread;
        size -= (size_t) nread;
    }
    return true;
}

static uint64_t ggml_svd_hash_input(const float * input, int64_t input_len) {
    const uint8_t * bytes = (const uint8_t *) input;
    const size_t nbytes = sizeof(float) * (size_t) input_len;
    uint64_t hash = 1469598103934665603ULL;

    for (size_t i = 0; i < nbytes; ++i) {
        hash ^= bytes[i];
        hash *= 1099511628211ULL;
    }

    return hash;
}

static void ggml_svd_set_socket_timeouts(int fd, int32_t timeout_ms) {
    if (timeout_ms <= 0) {
        return;
    }

    struct timeval tv;
    tv.tv_sec = timeout_ms / 1000;
    tv.tv_usec = (timeout_ms % 1000) * 1000;
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, (const char *) &tv, sizeof(tv));
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, (const char *) &tv, sizeof(tv));
}

static bool ggml_svd_client_connect_locked(void) {
    static int g_svd_connect_log_budget = 0;
    if (!g_svd_client.enabled || g_svd_client.port == 0 || g_svd_client.host[0] == '\0') {
        if (__atomic_fetch_add(&g_svd_connect_log_budget, 1, __ATOMIC_RELAXED) < 32) {
            fprintf(stderr,
                    "[svd-offload-connect] disabled enabled=%d host=%s port=%u\n",
                    g_svd_client.enabled ? 1 : 0,
                    g_svd_client.host[0] ? g_svd_client.host : "<empty>",
                    (unsigned) g_svd_client.port);
        }
        return false;
    }
    if (g_svd_client.socket_fd >= 0) {
        return true;
    }
    if (!ggml_svd_winsock_initialized()) {
        if (__atomic_fetch_add(&g_svd_connect_log_budget, 1, __ATOMIC_RELAXED) < 32) {
            fprintf(stderr, "[svd-offload-connect] winsock/socket init failed\n");
        }
        return false;
    }

    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        if (__atomic_fetch_add(&g_svd_connect_log_budget, 1, __ATOMIC_RELAXED) < 32) {
            fprintf(stderr, "[svd-offload-connect] socket() failed errno=%d\n", errno);
        }
        return false;
    }

    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, (const char *) &one, sizeof(one));
    ggml_svd_set_socket_timeouts(fd, g_svd_client.timeout_ms);

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(g_svd_client.port);
    if (inet_pton(AF_INET, g_svd_client.host, &addr.sin_addr) != 1) {
        if (__atomic_fetch_add(&g_svd_connect_log_budget, 1, __ATOMIC_RELAXED) < 32) {
            fprintf(stderr,
                    "[svd-offload-connect] inet_pton failed host=%s errno=%d\n",
                    g_svd_client.host,
                    errno);
        }
        ggml_svd_close_socket(fd);
        return false;
    }

    if (connect(fd, (const struct sockaddr *) &addr, sizeof(addr)) != 0) {
        if (__atomic_fetch_add(&g_svd_connect_log_budget, 1, __ATOMIC_RELAXED) < 32) {
            fprintf(stderr,
                    "[svd-offload-connect] connect failed host=%s port=%u errno=%d\n",
                    g_svd_client.host,
                    (unsigned) g_svd_client.port,
                    errno);
        }
        ggml_svd_close_socket(fd);
        return false;
    }

    g_svd_client.socket_fd = fd;
    if (__atomic_fetch_add(&g_svd_connect_log_budget, 1, __ATOMIC_RELAXED) < 32) {
        fprintf(stderr,
                "[svd-offload-connect] connected host=%s port=%u fd=%d\n",
                g_svd_client.host,
                (unsigned) g_svd_client.port,
                fd);
    }
    return true;
}

void ggml_svd_offload_set_client_config(const struct ggml_svd_offload_client_config * config) {
    pthread_mutex_lock(&g_svd_client.mutex);

    if (g_svd_client.socket_fd >= 0) {
        ggml_svd_close_socket(g_svd_client.socket_fd);
        g_svd_client.socket_fd = -1;
    }
    ggml_svd_gate_cache_clear_locked();

    g_svd_client.enabled = config != NULL && config->enabled;
    g_svd_client.port = config != NULL ? config->port : 0;
    g_svd_client.timeout_ms = config != NULL ? config->timeout_ms : 0;
    g_svd_client.host[0] = '\0';

    if (config != NULL && config->host != NULL) {
        strncpy(g_svd_client.host, config->host, sizeof(g_svd_client.host) - 1);
        g_svd_client.host[sizeof(g_svd_client.host) - 1] = '\0';
    }

    pthread_mutex_unlock(&g_svd_client.mutex);
}

bool ggml_svd_offload_client_enabled(void) {
    return g_svd_client.enabled && g_svd_client.port != 0 && g_svd_client.host[0] != '\0';
}

int32_t ggml_svd_offload_get_timeout_ms(void) {
    return g_svd_client.timeout_ms;
}

bool ggml_svd_offload_wait_ready(
        const struct ggml_svd_offload_request_handle * handle,
        int32_t timeout_ms) {
    if (handle == NULL || !handle->response_ready || handle->socket_fd < 0) {
        return false;
    }

    fd_set readfds;
    FD_ZERO(&readfds);
    FD_SET(handle->socket_fd, &readfds);

    struct timeval tv;
    struct timeval * tv_ptr = NULL;
    if (timeout_ms >= 0) {
        tv.tv_sec = timeout_ms / 1000;
        tv.tv_usec = (timeout_ms % 1000) * 1000;
        tv_ptr = &tv;
    }

    const int rc = select(GGML_SVD_SELECT_NFDS(handle->socket_fd), &readfds, NULL, NULL, tv_ptr);
    return rc > 0 && FD_ISSET(handle->socket_fd, &readfds);
}

void ggml_svd_offload_abort_request(
        struct ggml_svd_offload_request_handle * handle) {
    if (handle == NULL || !handle->response_ready) {
        return;
    }

    pthread_mutex_lock(&g_svd_client.mutex);
    const int fd = handle->socket_fd;
    if (fd >= 0) {
        ggml_svd_close_socket(fd);
        if (g_svd_client.socket_fd == fd) {
            g_svd_client.socket_fd = -1;
        }
    }
    ggml_svd_gate_cache_clear_locked();
    pthread_mutex_unlock(&g_svd_client.mutex);

    handle->socket_fd = -1;
    handle->response_ready = 0;
}

static bool ggml_svd_offload_begin_request_impl(
        enum ggml_svd_offload_request_kind request_kind,
        int32_t layer_id,
        int32_t op_id,
        float offload_rate,
        int64_t rank_start,
        const float * input,
        int64_t input_len,
        struct ggml_svd_offload_request_handle * handle) {
    static int g_svd_begin_log_budget = 0;
    if (!ggml_svd_offload_client_enabled() || handle == NULL || input == NULL || input_len <= 0 || rank_start < 0) {
        if (__atomic_fetch_add(&g_svd_begin_log_budget, 1, __ATOMIC_RELAXED) < 64) {
            fprintf(stderr,
                    "[svd-offload-begin] rejected kind=%d layer=%d op=%d enabled=%d handle=%d input=%d input_len=%lld rank_start=%lld host=%s port=%u\n",
                    (int) request_kind,
                    layer_id,
                    op_id,
                    ggml_svd_offload_client_enabled() ? 1 : 0,
                    handle != NULL ? 1 : 0,
                    input != NULL ? 1 : 0,
                    (long long) input_len,
                    (long long) rank_start,
                    g_svd_client.host[0] ? g_svd_client.host : "<empty>",
                    (unsigned) g_svd_client.port);
        }
        return false;
    }

    const uint64_t t_begin_us = ggml_svd_now_us();
    pthread_mutex_lock(&g_svd_client.mutex);

    if (!ggml_svd_client_connect_locked()) {
        if (__atomic_fetch_add(&g_svd_begin_log_budget, 1, __ATOMIC_RELAXED) < 64) {
            fprintf(stderr,
                    "[svd-offload-begin] connect_failed kind=%d layer=%d op=%d host=%s port=%u\n",
                    (int) request_kind,
                    layer_id,
                    op_id,
                    g_svd_client.host,
                    (unsigned) g_svd_client.port);
        }
        pthread_mutex_unlock(&g_svd_client.mutex);
        return false;
    }

    struct ggml_svd_offload_wire_request req;
    req.magic = GGML_SVD_OFFLOAD_MAGIC;
    req.version = GGML_SVD_OFFLOAD_VERSION;
    req.request_kind = (int32_t) request_kind;
    req.layer_id = layer_id;
    req.op_id = op_id;
    req.offload_rate = offload_rate;
    req.rank_start = (int32_t) rank_start;
    req.input_len = (int32_t) input_len;

    if (!ggml_svd_send_all(g_svd_client.socket_fd, &req, sizeof(req)) ||
        !ggml_svd_send_all(g_svd_client.socket_fd, input, sizeof(float) * (size_t) input_len)) {
        if (__atomic_fetch_add(&g_svd_begin_log_budget, 1, __ATOMIC_RELAXED) < 64) {
            fprintf(stderr,
                    "[svd-offload-begin] send_failed kind=%d layer=%d op=%d fd=%d errno=%d input_len=%lld\n",
                    (int) request_kind,
                    layer_id,
                    op_id,
                    g_svd_client.socket_fd,
                    errno,
                    (long long) input_len);
        }
        ggml_svd_close_socket(g_svd_client.socket_fd);
        g_svd_client.socket_fd = -1;
        ggml_svd_gate_cache_clear_locked();
        pthread_mutex_unlock(&g_svd_client.mutex);
        return false;
    }

    handle->socket_fd = g_svd_client.socket_fd;
    handle->response_ready = 1;
    handle->request_kind = (int32_t) request_kind;
    handle->layer_id = layer_id;
    handle->op_id = op_id;
    handle->rank_start = (int32_t) rank_start;
    handle->input_hash = ggml_svd_hash_input(input, input_len);
    handle->start_us = ggml_svd_now_us();

    if (__atomic_fetch_add(&g_svd_begin_log_budget, 1, __ATOMIC_RELAXED) < 64) {
        fprintf(stderr,
                "[svd-offload-begin] started kind=%d layer=%d op=%d fd=%d input_len=%lld rank_start=%lld offload_rate=%.3f\n",
                (int) request_kind,
                layer_id,
                op_id,
                handle->socket_fd,
                (long long) input_len,
                (long long) rank_start,
                offload_rate);
    }

    const uint64_t send_us = handle->start_us - t_begin_us;
    if (request_kind == GGML_SVD_OFFLOAD_REQ_UP_GATE) {
        g_svd_profile.requests_up_gate++;
        g_svd_profile.send_up_gate_us += send_us;
    } else if (request_kind == GGML_SVD_OFFLOAD_REQ_FFN) {
        g_svd_profile.requests_ffn++;
        g_svd_profile.send_ffn_us += send_us;
    } else if (op_id == GGML_SVD_OP_DOWN) {
        g_svd_profile.requests_down++;
        g_svd_profile.send_down_us += send_us;
    } else {
        g_svd_profile.requests_other_mat++;
        g_svd_profile.send_other_mat_us += send_us;
    }

    pthread_mutex_unlock(&g_svd_client.mutex);
    return true;
}

bool ggml_svd_offload_begin_request(
        int32_t layer_id,
        int32_t op_id,
        float offload_rate,
        int64_t rank_start,
        const float * input,
        int64_t input_len,
        struct ggml_svd_offload_request_handle * handle) {
    return ggml_svd_offload_begin_request_impl(
            GGML_SVD_OFFLOAD_REQ_MAT,
            layer_id,
            op_id,
            offload_rate,
            rank_start,
            input,
            input_len,
            handle);
}

bool ggml_svd_offload_begin_up_gate_request(
        int32_t layer_id,
        int32_t op_id,
        float offload_rate,
        int64_t rank_start,
        const float * input,
        int64_t input_len,
        struct ggml_svd_offload_request_handle * handle) {
    return ggml_svd_offload_begin_request_impl(
            GGML_SVD_OFFLOAD_REQ_UP_GATE,
            layer_id,
            op_id,
            offload_rate,
            rank_start,
            input,
            input_len,
            handle);
}

bool ggml_svd_offload_begin_ffn_request(
        int32_t layer_id,
        float offload_rate,
        const float * input,
        int64_t input_len,
        struct ggml_svd_offload_request_handle * handle) {
    return ggml_svd_offload_begin_request_impl(
            GGML_SVD_OFFLOAD_REQ_FFN,
            layer_id,
            -1,
            offload_rate,
            0,
            input,
            input_len,
            handle);
}

bool ggml_svd_offload_finish_request(
        struct ggml_svd_offload_request_handle * handle,
        float * output,
        int64_t output_len) {
    if (handle == NULL || !handle->response_ready || output == NULL || output_len <= 0) {
        return false;
    }

    pthread_mutex_lock(&g_svd_client.mutex);

    struct ggml_svd_offload_wire_response rsp;
    const int fd = handle->socket_fd;

    if (!ggml_svd_recv_all(fd, &rsp, sizeof(rsp)) ||
        rsp.magic != GGML_SVD_OFFLOAD_MAGIC ||
        rsp.version != GGML_SVD_OFFLOAD_VERSION ||
        rsp.status != 0 ||
        rsp.output_len_aux != 0 ||
        rsp.output_len != output_len ||
        !ggml_svd_recv_all(fd, output, sizeof(float) * (size_t) output_len)) {
        if (handle->op_id == GGML_SVD_OP_DOWN) {
            g_svd_profile.finish_fail_down++;
        } else {
            g_svd_profile.finish_fail_other_mat++;
        }
        ggml_svd_close_socket(fd);
        if (g_svd_client.socket_fd == fd) {
            g_svd_client.socket_fd = -1;
        }
        ggml_svd_gate_cache_clear_locked();
        pthread_mutex_unlock(&g_svd_client.mutex);
        handle->response_ready = 0;
        return false;
    }

    const uint64_t wait_us = ggml_svd_now_us() - handle->start_us;
    if (handle->op_id == GGML_SVD_OP_DOWN) {
        g_svd_profile.wait_down_us += wait_us;
        ggml_svd_profile_accum_output(
                output, output_len,
                &g_svd_profile.recv_down_sum_abs,
                &g_svd_profile.recv_down_max_abs,
                &g_svd_profile.recv_down_zero_vectors);
    } else {
        g_svd_profile.wait_other_mat_us += wait_us;
        ggml_svd_profile_accum_output(
                output, output_len,
                &g_svd_profile.recv_other_sum_abs,
                &g_svd_profile.recv_other_max_abs,
                &g_svd_profile.recv_other_zero_vectors);
    }

    pthread_mutex_unlock(&g_svd_client.mutex);
    handle->response_ready = 0;
    return true;
}

bool ggml_svd_offload_finish_ffn_request(
        struct ggml_svd_offload_request_handle * handle,
        float * output,
        int64_t output_len) {
    if (handle == NULL || !handle->response_ready || output == NULL || output_len <= 0) {
        return false;
    }

    pthread_mutex_lock(&g_svd_client.mutex);

    struct ggml_svd_offload_wire_response rsp;
    const int fd = handle->socket_fd;

    if (!ggml_svd_recv_all(fd, &rsp, sizeof(rsp)) ||
        rsp.magic != GGML_SVD_OFFLOAD_MAGIC ||
        rsp.version != GGML_SVD_OFFLOAD_VERSION ||
        rsp.status != 0 ||
        rsp.output_len_aux != 0 ||
        rsp.output_len != output_len ||
        !ggml_svd_recv_all(fd, output, sizeof(float) * (size_t) output_len)) {
        g_svd_profile.finish_fail_ffn++;
        ggml_svd_close_socket(fd);
        if (g_svd_client.socket_fd == fd) {
            g_svd_client.socket_fd = -1;
        }
        ggml_svd_gate_cache_clear_locked();
        pthread_mutex_unlock(&g_svd_client.mutex);
        handle->response_ready = 0;
        return false;
    }

    const uint64_t wait_us = ggml_svd_now_us() - handle->start_us;
    g_svd_profile.wait_ffn_us += wait_us;
    ggml_svd_profile_accum_output(
            output, output_len,
            &g_svd_profile.recv_ffn_sum_abs,
            &g_svd_profile.recv_ffn_max_abs,
            &g_svd_profile.recv_ffn_zero_vectors);

    pthread_mutex_unlock(&g_svd_client.mutex);
    handle->response_ready = 0;
    return true;
}

bool ggml_svd_offload_finish_up_gate_request(
        struct ggml_svd_offload_request_handle * handle,
        float * output,
        int64_t output_len,
        int64_t paired_output_len) {
    if (handle == NULL || !handle->response_ready || output == NULL || output_len <= 0 || paired_output_len <= 0) {
        return false;
    }

    pthread_mutex_lock(&g_svd_client.mutex);

    struct ggml_svd_offload_wire_response rsp;
    const int fd = handle->socket_fd;

    bool ok =
        ggml_svd_recv_all(fd, &rsp, sizeof(rsp)) &&
        rsp.magic == GGML_SVD_OFFLOAD_MAGIC &&
        rsp.version == GGML_SVD_OFFLOAD_VERSION &&
        rsp.status == 0 &&
        rsp.output_len == output_len &&
        rsp.output_len_aux == paired_output_len &&
        true;

    if (!ok) {
        g_svd_profile.finish_fail_up_gate++;
        ggml_svd_close_socket(fd);
        if (g_svd_client.socket_fd == fd) {
            g_svd_client.socket_fd = -1;
        }
        ggml_svd_gate_cache_clear_locked();
        pthread_mutex_unlock(&g_svd_client.mutex);
        handle->response_ready = 0;
        return false;
    }

    const uint64_t wait_us = ggml_svd_now_us() - handle->start_us;
    g_svd_profile.wait_up_gate_us += wait_us;

    struct ggml_svd_pair_cache_entry * entry =
        ggml_svd_pair_cache_prepare_slot_locked(handle->layer_id, handle->rank_start, handle->input_hash);
    ok = ok &&
        ggml_svd_pair_cache_resize_locked(&entry->up_data, (int32_t) output_len) &&
        ggml_svd_pair_cache_resize_locked(&entry->gate_data, (int32_t) paired_output_len);

    if (!ok) {
        g_svd_profile.finish_fail_up_gate++;
        ggml_svd_close_socket(fd);
        if (g_svd_client.socket_fd == fd) {
            g_svd_client.socket_fd = -1;
        }
        ggml_svd_gate_cache_clear_locked();
        pthread_mutex_unlock(&g_svd_client.mutex);
        handle->response_ready = 0;
        return false;
    }

    float * first_out = handle->op_id == GGML_SVD_OP_GATE ? entry->up_data : output;
    float * second_out = handle->op_id == GGML_SVD_OP_GATE ? output : entry->gate_data;
    ok = ggml_svd_recv_all(fd, first_out, sizeof(float) * (size_t) output_len) &&
         ggml_svd_recv_all(fd, second_out, sizeof(float) * (size_t) paired_output_len);

    if (!ok) {
        g_svd_profile.finish_fail_up_gate++;
        ggml_svd_close_socket(fd);
        if (g_svd_client.socket_fd == fd) {
            g_svd_client.socket_fd = -1;
        }
        ggml_svd_gate_cache_clear_locked();
        pthread_mutex_unlock(&g_svd_client.mutex);
        handle->response_ready = 0;
        return false;
    }

    entry->layer_id = handle->layer_id;
    entry->rank_start = handle->rank_start;
    entry->output_len = (int32_t) output_len;
    entry->input_hash = handle->input_hash;
    entry->valid_up = true;
    entry->valid_gate = true;

    ggml_svd_profile_accum_output(
            handle->op_id == GGML_SVD_OP_GATE ? entry->up_data : output,
            output_len,
            &g_svd_profile.recv_up_sum_abs,
            &g_svd_profile.recv_up_max_abs,
            &g_svd_profile.recv_up_zero_vectors);
    ggml_svd_profile_accum_output(
            handle->op_id == GGML_SVD_OP_GATE ? output : entry->gate_data,
            paired_output_len,
            &g_svd_profile.recv_gate_sum_abs,
            &g_svd_profile.recv_gate_max_abs,
            &g_svd_profile.recv_gate_zero_vectors);

    pthread_mutex_unlock(&g_svd_client.mutex);
    handle->response_ready = 0;
    return true;
}

static bool ggml_svd_offload_has_cached_pair_impl(
        int32_t layer_id,
        int64_t rank_start,
        const float * input,
        int64_t input_len,
        int64_t output_len,
        bool want_gate) {
    if (input == NULL || input_len <= 0 || output_len <= 0) {
        return false;
    }

    const uint64_t input_hash = ggml_svd_hash_input(input, input_len);
    bool ok = false;
    pthread_mutex_lock(&g_svd_client.mutex);
    g_svd_profile.gate_cache_checks++;
    struct ggml_svd_pair_cache_entry * found = NULL;
    for (int i = 0; i < GGML_SVD_GATE_CACHE_SLOTS; ++i) {
        struct ggml_svd_pair_cache_entry * entry = &g_svd_pair_cache[i];
        if (!ggml_svd_pair_cache_matches(entry, layer_id, rank_start, input_hash, output_len)) {
            continue;
        }
        if ((want_gate && entry->valid_gate) || (!want_gate && entry->valid_up)) {
            found = entry;
            break;
        }
    }
    if (found == NULL) {
        bool any_valid = false;
        for (int i = 0; i < GGML_SVD_GATE_CACHE_SLOTS; ++i) {
            if (g_svd_pair_cache[i].valid_up || g_svd_pair_cache[i].valid_gate) {
                any_valid = true;
                break;
            }
        }
        if (!any_valid) {
            g_svd_profile.gate_cache_miss_invalid++;
        } else {
            bool same_layer = false;
            bool same_rank = false;
            bool same_hash = false;
            bool same_output = false;
            bool same_data = false;
            for (int i = 0; i < GGML_SVD_GATE_CACHE_SLOTS; ++i) {
                struct ggml_svd_pair_cache_entry * entry = &g_svd_pair_cache[i];
                if (!entry->valid_up && !entry->valid_gate) {
                    continue;
                }
                if (entry->layer_id == layer_id) {
                    same_layer = true;
                }
                if (entry->layer_id == layer_id && entry->rank_start == rank_start) {
                    same_rank = true;
                }
                if (entry->layer_id == layer_id && entry->rank_start == rank_start && entry->input_hash == input_hash) {
                    same_hash = true;
                }
                if (entry->layer_id == layer_id && entry->rank_start == rank_start && entry->input_hash == input_hash && entry->output_len == output_len) {
                    same_output = true;
                }
                if (entry->layer_id == layer_id && entry->rank_start == rank_start && entry->input_hash == input_hash && entry->output_len == output_len &&
                        ((want_gate && entry->gate_data != NULL && entry->valid_gate) || (!want_gate && entry->up_data != NULL && entry->valid_up))) {
                    same_data = true;
                }
            }
            if (!same_layer) {
                g_svd_profile.gate_cache_miss_layer++;
            } else if (!same_rank) {
                g_svd_profile.gate_cache_miss_rank++;
            } else if (!same_hash) {
                g_svd_profile.gate_cache_miss_hash++;
            } else if (!same_output) {
                g_svd_profile.gate_cache_miss_output_len++;
            } else if (!same_data) {
                g_svd_profile.gate_cache_miss_data++;
            }
        }
    }
    ok = found != NULL;
    pthread_mutex_unlock(&g_svd_client.mutex);
    return ok;
}

bool ggml_svd_offload_has_cached_up(
        int32_t layer_id,
        int64_t rank_start,
        const float * input,
        int64_t input_len,
        int64_t output_len) {
    return ggml_svd_offload_has_cached_pair_impl(layer_id, rank_start, input, input_len, output_len, false);
}

bool ggml_svd_offload_has_cached_gate(
        int32_t layer_id,
        int64_t rank_start,
        const float * input,
        int64_t input_len,
        int64_t output_len) {
    return ggml_svd_offload_has_cached_pair_impl(layer_id, rank_start, input, input_len, output_len, true);
}

static bool ggml_svd_offload_take_cached_pair_impl(
        int32_t layer_id,
        int64_t rank_start,
        const float * input,
        int64_t input_len,
        float * output,
        int64_t output_len,
        bool want_gate) {
    if (input == NULL || input_len <= 0 || output == NULL || output_len <= 0) {
        return false;
    }

    const uint64_t input_hash = ggml_svd_hash_input(input, input_len);
    bool ok = false;
    pthread_mutex_lock(&g_svd_client.mutex);
    struct ggml_svd_pair_cache_entry * entry = NULL;
    for (int i = 0; i < GGML_SVD_GATE_CACHE_SLOTS; ++i) {
        struct ggml_svd_pair_cache_entry * candidate = &g_svd_pair_cache[i];
        if (!ggml_svd_pair_cache_matches(candidate, layer_id, rank_start, input_hash, output_len)) {
            continue;
        }
        if ((want_gate && candidate->valid_gate) || (!want_gate && candidate->valid_up)) {
            entry = candidate;
            break;
        }
    }
    ok = entry != NULL;
    if (ok) {
        g_svd_profile.gate_cache_hits++;
        memcpy(output, want_gate ? entry->gate_data : entry->up_data, sizeof(float) * (size_t) output_len);
        if (want_gate) {
            entry->valid_gate = false;
        } else {
            entry->valid_up = false;
        }
        if (!entry->valid_up && !entry->valid_gate) {
            ggml_svd_pair_cache_reset_entry_locked(entry);
        }
    } else if (want_gate) {
        g_svd_profile.cache_take_fail_gate++;
    } else {
        g_svd_profile.cache_take_fail_up++;
    }
    pthread_mutex_unlock(&g_svd_client.mutex);
    return ok;
}

bool ggml_svd_offload_take_cached_up(
        int32_t layer_id,
        int64_t rank_start,
        const float * input,
        int64_t input_len,
        float * output,
        int64_t output_len) {
    return ggml_svd_offload_take_cached_pair_impl(layer_id, rank_start, input, input_len, output, output_len, false);
}

bool ggml_svd_offload_take_cached_gate(
        int32_t layer_id,
        int64_t rank_start,
        const float * input,
        int64_t input_len,
        float * output,
        int64_t output_len) {
    return ggml_svd_offload_take_cached_pair_impl(layer_id, rank_start, input, input_len, output, output_len, true);
}

void ggml_svd_offload_close_client(void) {
    ggml_svd_local_profile_print_and_reset();
    pthread_mutex_lock(&g_svd_client.mutex);
    const uint64_t send_total_us =
        g_svd_profile.send_up_gate_us +
        g_svd_profile.send_down_us +
        g_svd_profile.send_other_mat_us +
        g_svd_profile.send_ffn_us;
    const uint64_t wait_total_us =
        g_svd_profile.wait_up_gate_us +
        g_svd_profile.wait_down_us +
        g_svd_profile.wait_other_mat_us +
        g_svd_profile.wait_ffn_us;
    fprintf(stderr,
            "[svd-offload-client] up_gate_req=%llu down_req=%llu other_mat_req=%llu ffn_req=%llu gate_cache_hits=%llu gate_cache_checks=%llu "
            "miss_invalid=%llu miss_layer=%llu miss_rank=%llu miss_hash=%llu miss_output=%llu miss_data=%llu "
            "send_up_gate=%.3f ms send_down=%.3f ms send_other=%.3f ms send_ffn=%.3f ms "
            "wait_up_gate=%.3f ms wait_down=%.3f ms wait_other=%.3f ms wait_ffn=%.3f ms\n",
            (unsigned long long) g_svd_profile.requests_up_gate,
            (unsigned long long) g_svd_profile.requests_down,
            (unsigned long long) g_svd_profile.requests_other_mat,
            (unsigned long long) g_svd_profile.requests_ffn,
            (unsigned long long) g_svd_profile.gate_cache_hits,
            (unsigned long long) g_svd_profile.gate_cache_checks,
            (unsigned long long) g_svd_profile.gate_cache_miss_invalid,
            (unsigned long long) g_svd_profile.gate_cache_miss_layer,
            (unsigned long long) g_svd_profile.gate_cache_miss_rank,
            (unsigned long long) g_svd_profile.gate_cache_miss_hash,
            (unsigned long long) g_svd_profile.gate_cache_miss_output_len,
            (unsigned long long) g_svd_profile.gate_cache_miss_data,
            g_svd_profile.send_up_gate_us / 1000.0,
            g_svd_profile.send_down_us / 1000.0,
            g_svd_profile.send_other_mat_us / 1000.0,
            g_svd_profile.send_ffn_us / 1000.0,
            g_svd_profile.wait_up_gate_us / 1000.0,
            g_svd_profile.wait_down_us / 1000.0,
            g_svd_profile.wait_other_mat_us / 1000.0,
            g_svd_profile.wait_ffn_us / 1000.0);
    fprintf(stderr,
            "[svd-offload-client-stage-profile] rpc_send_total=%.3f ms "
            "rpc_wait_total=%.3f ms\n",
            send_total_us / 1000.0,
            wait_total_us / 1000.0);
    fprintf(stderr,
            "[svd-offload-client-debug] finish_fail_up_gate=%llu finish_fail_down=%llu finish_fail_other=%llu finish_fail_ffn=%llu "
            "cache_take_fail_up=%llu cache_take_fail_gate=%llu "
            "recv_up_sum_abs=%.6e recv_up_max_abs=%.6e recv_up_zero=%llu "
            "recv_gate_sum_abs=%.6e recv_gate_max_abs=%.6e recv_gate_zero=%llu "
            "recv_down_sum_abs=%.6e recv_down_max_abs=%.6e recv_down_zero=%llu "
            "recv_other_sum_abs=%.6e recv_other_max_abs=%.6e recv_other_zero=%llu "
            "recv_ffn_sum_abs=%.6e recv_ffn_max_abs=%.6e recv_ffn_zero=%llu\n",
            (unsigned long long) g_svd_profile.finish_fail_up_gate,
            (unsigned long long) g_svd_profile.finish_fail_down,
            (unsigned long long) g_svd_profile.finish_fail_other_mat,
            (unsigned long long) g_svd_profile.finish_fail_ffn,
            (unsigned long long) g_svd_profile.cache_take_fail_up,
            (unsigned long long) g_svd_profile.cache_take_fail_gate,
            g_svd_profile.recv_up_sum_abs,
            (double) g_svd_profile.recv_up_max_abs,
            (unsigned long long) g_svd_profile.recv_up_zero_vectors,
            g_svd_profile.recv_gate_sum_abs,
            (double) g_svd_profile.recv_gate_max_abs,
            (unsigned long long) g_svd_profile.recv_gate_zero_vectors,
            g_svd_profile.recv_down_sum_abs,
            (double) g_svd_profile.recv_down_max_abs,
            (unsigned long long) g_svd_profile.recv_down_zero_vectors,
            g_svd_profile.recv_other_sum_abs,
            (double) g_svd_profile.recv_other_max_abs,
            (unsigned long long) g_svd_profile.recv_other_zero_vectors,
            g_svd_profile.recv_ffn_sum_abs,
            (double) g_svd_profile.recv_ffn_max_abs,
            (unsigned long long) g_svd_profile.recv_ffn_zero_vectors);
    if (g_svd_client.socket_fd >= 0) {
        ggml_svd_close_socket(g_svd_client.socket_fd);
        g_svd_client.socket_fd = -1;
    }
    ggml_svd_gate_cache_clear_locked();
    memset(&g_svd_profile, 0, sizeof(g_svd_profile));
    pthread_mutex_unlock(&g_svd_client.mutex);
}
