#include <arpa/inet.h>
#include <errno.h>
#include <inttypes.h>
#include <netinet/in.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t) ts.tv_sec * 1000000000ull + (uint64_t) ts.tv_nsec;
}

static int cmp_u64(const void * a, const void * b) {
    const uint64_t aa = *(const uint64_t *) a;
    const uint64_t bb = *(const uint64_t *) b;
    return (aa > bb) - (aa < bb);
}

int main(int argc, char ** argv) {
    const char * ip = argc > 1 ? argv[1] : "10.20.0.3";
    const int port = argc > 2 ? atoi(argv[2]) : 19090;
    const int payload_size = argc > 3 ? atoi(argv[3]) : 4;
    const int rounds = argc > 4 ? atoi(argv[4]) : 1000;

    if (payload_size <= 0 || payload_size > 65507) {
        fprintf(stderr, "invalid payload_size: %d\n", payload_size);
        return 1;
    }
    if (rounds <= 0) {
        fprintf(stderr, "invalid rounds: %d\n", rounds);
        return 1;
    }

    int fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (fd < 0) {
        perror("socket");
        return 1;
    }

    struct timeval tv;
    tv.tv_sec = 2;
    tv.tv_usec = 0;
    if (setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) != 0) {
        perror("setsockopt(SO_RCVTIMEO)");
        close(fd);
        return 1;
    }

    struct sockaddr_in server;
    memset(&server, 0, sizeof(server));
    server.sin_family = AF_INET;
    server.sin_port = htons((uint16_t) port);
    if (inet_pton(AF_INET, ip, &server.sin_addr) != 1) {
        fprintf(stderr, "invalid ip: %s\n", ip);
        close(fd);
        return 1;
    }

    char * send_buf = (char *) malloc((size_t) payload_size);
    char * recv_buf = (char *) malloc((size_t) payload_size);
    uint64_t * rtts = (uint64_t *) malloc(sizeof(uint64_t) * (size_t) rounds);
    if (send_buf == NULL || recv_buf == NULL || rtts == NULL) {
        fprintf(stderr, "malloc failed\n");
        free(send_buf);
        free(recv_buf);
        free(rtts);
        close(fd);
        return 1;
    }

    for (int i = 0; i < payload_size; ++i) {
        send_buf[i] = (char) (i & 0x7f);
    }

    struct sockaddr_in peer;
    socklen_t peer_len = sizeof(peer);

    for (int i = 0; i < rounds; ++i) {
        send_buf[0] = (char) (i & 0xff);
        const uint64_t t0 = now_ns();
        const ssize_t sent = sendto(fd, send_buf, (size_t) payload_size, 0, (struct sockaddr *) &server, sizeof(server));
        if (sent != payload_size) {
            perror("sendto");
            free(send_buf);
            free(recv_buf);
            free(rtts);
            close(fd);
            return 1;
        }

        const ssize_t recvd = recvfrom(fd, recv_buf, (size_t) payload_size, 0, (struct sockaddr *) &peer, &peer_len);
        const uint64_t t1 = now_ns();
        if (recvd != payload_size) {
            if (recvd < 0) {
                perror("recvfrom");
            } else {
                fprintf(stderr, "short recv: %zd\n", recvd);
            }
            free(send_buf);
            free(recv_buf);
            free(rtts);
            close(fd);
            return 1;
        }
        if (memcmp(send_buf, recv_buf, (size_t) payload_size) != 0) {
            fprintf(stderr, "payload mismatch at round %d\n", i);
            free(send_buf);
            free(recv_buf);
            free(rtts);
            close(fd);
            return 1;
        }

        rtts[i] = t1 - t0;
    }

    uint64_t sum = 0;
    uint64_t min = rtts[0];
    uint64_t max = rtts[0];
    for (int i = 0; i < rounds; ++i) {
        const uint64_t v = rtts[i];
        sum += v;
        if (v < min) min = v;
        if (v > max) max = v;
    }
    qsort(rtts, (size_t) rounds, sizeof(uint64_t), cmp_u64);

    const double avg_us = (double) sum / (double) rounds / 1000.0;
    const double min_us = (double) min / 1000.0;
    const double p50_us = (double) rtts[rounds / 2] / 1000.0;
    const double p95_us = (double) rtts[(rounds * 95) / 100] / 1000.0;
    const double max_us = (double) max / 1000.0;

    printf("udp_rtt_result\n");
    printf("target_ip=%s\n", ip);
    printf("port=%d\n", port);
    printf("payload_bytes=%d\n", payload_size);
    printf("rounds=%d\n", rounds);
    printf("avg_rtt_us=%.3f\n", avg_us);
    printf("min_rtt_us=%.3f\n", min_us);
    printf("p50_rtt_us=%.3f\n", p50_us);
    printf("p95_rtt_us=%.3f\n", p95_us);
    printf("max_rtt_us=%.3f\n", max_us);
    printf("avg_one_way_us=%.3f\n", avg_us / 2.0);

    free(send_buf);
    free(recv_buf);
    free(rtts);
    close(fd);
    return 0;
}
