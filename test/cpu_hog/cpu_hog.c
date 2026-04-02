#define _GNU_SOURCE
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

static volatile sig_atomic_t g_stop = 0;

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t) ts.tv_sec * 1000000000ull + (uint64_t) ts.tv_nsec;
}

static void on_sig(int sig) {
    (void) sig;
    g_stop = 1;
}

struct worker_arg {
    int cpu_id;
    int busy_percent;
};

static void * worker_main(void * opaque) {
    struct worker_arg * arg = (struct worker_arg *) opaque;

    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(arg->cpu_id, &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);

    const uint64_t period_ns = 10ull * 1000ull * 1000ull;
    const uint64_t busy_ns = (period_ns * (uint64_t) arg->busy_percent) / 100ull;

    volatile uint64_t sink = 0;
    while (!g_stop) {
        const uint64_t start = now_ns();
        while (!g_stop && now_ns() - start < busy_ns) {
            sink += (uint64_t) arg->cpu_id + 1;
            sink ^= sink << 1;
        }

        const uint64_t used = now_ns() - start;
        if (!g_stop && used < period_ns) {
            struct timespec req;
            const uint64_t sleep_ns = period_ns - used;
            req.tv_sec = (time_t) (sleep_ns / 1000000000ull);
            req.tv_nsec = (long) (sleep_ns % 1000000000ull);
            nanosleep(&req, NULL);
        }
    }

    return (void *) (uintptr_t) sink;
}

int main(int argc, char ** argv) {
    int n_threads = argc > 1 ? atoi(argv[1]) : 32;
    const int busy_percent = argc > 2 ? atoi(argv[2]) : 40;
    const int base_cpu = argc > 3 ? atoi(argv[3]) : 0;
    const int duration_sec = argc > 4 ? atoi(argv[4]) : 600;

    if (n_threads <= 0 || busy_percent < 0 || busy_percent > 100 || duration_sec <= 0) {
        fprintf(stderr, "usage: %s <threads> <busy_percent> <base_cpu> <duration_sec>\n", argv[0]);
        return 1;
    }

    signal(SIGINT, on_sig);
    signal(SIGTERM, on_sig);

    pthread_t * threads = (pthread_t *) calloc((size_t) n_threads, sizeof(pthread_t));
    struct worker_arg * args = (struct worker_arg *) calloc((size_t) n_threads, sizeof(struct worker_arg));
    if (threads == NULL || args == NULL) {
        fprintf(stderr, "alloc failed\n");
        free(threads);
        free(args);
        return 1;
    }

    for (int i = 0; i < n_threads; ++i) {
        args[i].cpu_id = base_cpu + i;
        args[i].busy_percent = busy_percent;
        if (pthread_create(&threads[i], NULL, worker_main, &args[i]) != 0) {
            fprintf(stderr, "pthread_create failed at %d\n", i);
            g_stop = 1;
            n_threads = i;
            break;
        }
    }

    for (int i = 0; i < duration_sec && !g_stop; ++i) {
        sleep(1);
    }
    g_stop = 1;

    for (int i = 0; i < n_threads; ++i) {
        if (threads[i]) {
            pthread_join(threads[i], NULL);
        }
    }

    free(threads);
    free(args);
    return 0;
}
