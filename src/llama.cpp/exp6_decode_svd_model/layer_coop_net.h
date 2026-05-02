#pragma once

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

struct LayerCoopRequest {
    uint32_t magic;
    uint32_t version;
    int32_t  n_tokens;
    int32_t  n_embd;
    int32_t  n_vocab;
    int32_t  reset_kv;
};

struct LayerCoopResponse {
    uint32_t magic;
    uint32_t version;
    int32_t  status;
    int32_t  n_logits;
    int32_t  selected_token;
    int32_t  reserved;
    double   server_decode_ms;
};

constexpr uint32_t LAYER_COOP_MAGIC = 0x4c434f4fU; // LCOO
constexpr uint32_t LAYER_COOP_VERSION = 2;

#ifdef _WIN32
inline bool layer_coop_net_init() {
    static bool done = false;
    static bool ok = false;
    if (!done) {
        WSADATA wsa;
        ok = WSAStartup(MAKEWORD(2, 2), &wsa) == 0;
        done = true;
    }
    return ok;
}

inline int layer_coop_close(int fd) {
    return closesocket((SOCKET) fd);
}
#else
inline bool layer_coop_net_init() {
    return true;
}

inline int layer_coop_close(int fd) {
    return close(fd);
}
#endif

inline bool layer_coop_send_all(int fd, const void * data, size_t size) {
    const char * p = (const char *) data;
    while (size > 0) {
#ifdef _WIN32
        const int n = send((SOCKET) fd, p, (int) size, 0);
#else
        const ssize_t n = write(fd, p, size);
#endif
        if (n <= 0) return false;
        p += n;
        size -= (size_t) n;
    }
    return true;
}

inline bool layer_coop_recv_all(int fd, void * data, size_t size) {
    char * p = (char *) data;
    while (size > 0) {
#ifdef _WIN32
        const int n = recv((SOCKET) fd, p, (int) size, 0);
#else
        const ssize_t n = read(fd, p, size);
#endif
        if (n <= 0) return false;
        p += n;
        size -= (size_t) n;
    }
    return true;
}

inline int layer_coop_connect(const std::string & host, uint16_t port) {
    if (!layer_coop_net_init()) {
        throw std::runtime_error("network init failed");
    }
    const int fd = (int) socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        throw std::runtime_error("socket failed");
    }
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, (const char *) &one, sizeof(one));

    sockaddr_in addr {};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
        layer_coop_close(fd);
        throw std::runtime_error("invalid host: " + host);
    }
    if (connect(fd, (sockaddr *) &addr, sizeof(addr)) != 0) {
        layer_coop_close(fd);
        throw std::runtime_error("connect failed");
    }
    return fd;
}

inline bool layer_coop_parse_host_port(const std::string & arg, std::string & host, uint16_t & port) {
    const size_t pos = arg.find(':');
    if (pos == std::string::npos) return false;
    host = arg.substr(0, pos);
    port = (uint16_t) std::stoi(arg.substr(pos + 1));
    return !host.empty() && port != 0;
}
