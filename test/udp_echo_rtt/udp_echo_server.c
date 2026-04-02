#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

static volatile sig_atomic_t g_running = 1;

static void handle_signal(int sig) {
    (void) sig;
    g_running = 0;
}

int main(int argc, char ** argv) {
    const int port = argc > 1 ? atoi(argv[1]) : 19090;
    const int max_packet = argc > 2 ? atoi(argv[2]) : 2048;

    signal(SIGINT, handle_signal);
    signal(SIGTERM, handle_signal);

    int fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (fd < 0) {
        perror("socket");
        return 1;
    }

    int reuse = 1;
    if (setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) != 0) {
        perror("setsockopt(SO_REUSEADDR)");
        close(fd);
        return 1;
    }

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons((uint16_t) port);

    if (bind(fd, (struct sockaddr *) &addr, sizeof(addr)) != 0) {
        perror("bind");
        close(fd);
        return 1;
    }

    char * buf = (char *) malloc((size_t) max_packet);
    if (buf == NULL) {
        fprintf(stderr, "malloc failed\n");
        close(fd);
        return 1;
    }

    fprintf(stderr, "udp_echo_server listening on port %d\n", port);
    while (g_running) {
        struct sockaddr_in peer;
        socklen_t peer_len = sizeof(peer);
        const ssize_t n = recvfrom(fd, buf, (size_t) max_packet, 0, (struct sockaddr *) &peer, &peer_len);
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            perror("recvfrom");
            break;
        }

        const ssize_t sent = sendto(fd, buf, (size_t) n, 0, (struct sockaddr *) &peer, peer_len);
        if (sent != n) {
            perror("sendto");
            break;
        }
    }

    free(buf);
    close(fd);
    return 0;
}
