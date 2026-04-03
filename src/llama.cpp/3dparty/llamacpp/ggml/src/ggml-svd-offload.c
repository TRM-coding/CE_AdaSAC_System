#include "ggml-svd-offload.h"

#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

struct ggml_svd_offload_wire_request {
    uint32_t magic;
    uint32_t version;
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

enum {
    GGML_SVD_OFFLOAD_MAGIC = 0x5344564fU,
    GGML_SVD_OFFLOAD_VERSION = 1,
};

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

bool ggml_svd_offload_begin_request(
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

    pthread_mutex_lock(&g_svd_client.mutex);

    if (!ggml_svd_client_connect_locked()) {
        pthread_mutex_unlock(&g_svd_client.mutex);
        return false;
    }

    struct ggml_svd_offload_wire_request req;
    req.magic = GGML_SVD_OFFLOAD_MAGIC;
    req.version = GGML_SVD_OFFLOAD_VERSION;
    req.layer_id = layer_id;
    req.op_id = op_id;
    req.offload_rate = offload_rate;
    req.rank_start = (int32_t) rank_start;
    req.input_len = (int32_t) input_len;

    if (!ggml_svd_send_all(g_svd_client.socket_fd, &req, sizeof(req)) ||
        !ggml_svd_send_all(g_svd_client.socket_fd, input, sizeof(float) * (size_t) input_len)) {
        close(g_svd_client.socket_fd);
        g_svd_client.socket_fd = -1;
        pthread_mutex_unlock(&g_svd_client.mutex);
        return false;
    }

    handle->socket_fd = g_svd_client.socket_fd;
    handle->response_ready = 1;

    pthread_mutex_unlock(&g_svd_client.mutex);
    return true;
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
        rsp.output_len != output_len ||
        !ggml_svd_recv_all(fd, output, sizeof(float) * (size_t) output_len)) {
        close(fd);
        if (g_svd_client.socket_fd == fd) {
            g_svd_client.socket_fd = -1;
        }
        pthread_mutex_unlock(&g_svd_client.mutex);
        handle->response_ready = 0;
        return false;
    }

    pthread_mutex_unlock(&g_svd_client.mutex);
    handle->response_ready = 0;
    return true;
}

void ggml_svd_offload_close_client(void) {
    pthread_mutex_lock(&g_svd_client.mutex);
    if (g_svd_client.socket_fd >= 0) {
        close(g_svd_client.socket_fd);
        g_svd_client.socket_fd = -1;
    }
    pthread_mutex_unlock(&g_svd_client.mutex);
}
