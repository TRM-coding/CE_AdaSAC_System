#include "ggml-svd-offload.h"

#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

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
    uint64_t send_up_gate_us;
    uint64_t send_down_us;
    uint64_t send_other_mat_us;
} g_svd_profile = { 0 };

enum {
    GGML_SVD_OFFLOAD_MAGIC = 0x5344564fU,
    GGML_SVD_OFFLOAD_VERSION = 2,
};

static uint64_t ggml_svd_now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t) ts.tv_sec * 1000000ULL + (uint64_t) ts.tv_nsec / 1000ULL;
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
        const ssize_t written = send(fd, ptr, size, MSG_NOSIGNAL);
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
        const ssize_t nread = recv(fd, ptr, size, 0);
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
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
}

static bool ggml_svd_client_connect_locked(void) {
    if (!g_svd_client.enabled || g_svd_client.port == 0 || g_svd_client.host[0] == '\0') {
        return false;
    }
    if (g_svd_client.socket_fd >= 0) {
        return true;
    }

    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        return false;
    }

    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    ggml_svd_set_socket_timeouts(fd, g_svd_client.timeout_ms);

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(g_svd_client.port);
    if (inet_pton(AF_INET, g_svd_client.host, &addr.sin_addr) != 1) {
        close(fd);
        return false;
    }

    if (connect(fd, (const struct sockaddr *) &addr, sizeof(addr)) != 0) {
        close(fd);
        return false;
    }

    g_svd_client.socket_fd = fd;
    return true;
}

void ggml_svd_offload_set_client_config(const struct ggml_svd_offload_client_config * config) {
    pthread_mutex_lock(&g_svd_client.mutex);

    if (g_svd_client.socket_fd >= 0) {
        close(g_svd_client.socket_fd);
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

static bool ggml_svd_offload_begin_request_impl(
        enum ggml_svd_offload_request_kind request_kind,
        int32_t layer_id,
        int32_t op_id,
        float offload_rate,
        int64_t rank_start,
        const float * input,
        int64_t input_len,
        struct ggml_svd_offload_request_handle * handle) {
    if (!ggml_svd_offload_client_enabled() || handle == NULL || input == NULL || input_len <= 0 || rank_start < 0) {
        return false;
    }

    const uint64_t t_begin_us = ggml_svd_now_us();
    pthread_mutex_lock(&g_svd_client.mutex);

    if (!ggml_svd_client_connect_locked()) {
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
        close(g_svd_client.socket_fd);
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

    const uint64_t send_us = handle->start_us - t_begin_us;
    if (request_kind == GGML_SVD_OFFLOAD_REQ_UP_GATE) {
        g_svd_profile.requests_up_gate++;
        g_svd_profile.send_up_gate_us += send_us;
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
        close(fd);
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
    } else {
        g_svd_profile.wait_other_mat_us += wait_us;
    }

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
        close(fd);
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
        close(fd);
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
        close(fd);
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
    pthread_mutex_lock(&g_svd_client.mutex);
    fprintf(stderr,
            "[svd-offload-client] up_gate_req=%llu down_req=%llu other_mat_req=%llu gate_cache_hits=%llu gate_cache_checks=%llu "
            "miss_invalid=%llu miss_layer=%llu miss_rank=%llu miss_hash=%llu miss_output=%llu miss_data=%llu "
            "send_up_gate=%.3f ms send_down=%.3f ms send_other=%.3f ms "
            "wait_up_gate=%.3f ms wait_down=%.3f ms wait_other=%.3f ms\n",
            (unsigned long long) g_svd_profile.requests_up_gate,
            (unsigned long long) g_svd_profile.requests_down,
            (unsigned long long) g_svd_profile.requests_other_mat,
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
            g_svd_profile.wait_up_gate_us / 1000.0,
            g_svd_profile.wait_down_us / 1000.0,
            g_svd_profile.wait_other_mat_us / 1000.0);
    if (g_svd_client.socket_fd >= 0) {
        close(g_svd_client.socket_fd);
        g_svd_client.socket_fd = -1;
    }
    ggml_svd_gate_cache_clear_locked();
    memset(&g_svd_profile, 0, sizeof(g_svd_profile));
    pthread_mutex_unlock(&g_svd_client.mutex);
}
