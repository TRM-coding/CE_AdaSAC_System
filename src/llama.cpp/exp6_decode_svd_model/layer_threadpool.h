#pragma once

#include "ggml-cpu.h"
#include "llama.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>

struct LayerThreadpoolDeleter {
    void operator()(ggml_threadpool * tp) const {
        if (tp) {
            ggml_threadpool_free(tp);
        }
    }
};

using LayerThreadpoolPtr = std::unique_ptr<ggml_threadpool, LayerThreadpoolDeleter>;

static bool layer_parse_cpu_mask(const std::string & mask, bool (&boolmask)[GGML_MAX_N_THREADS]) {
    std::fill(boolmask, boolmask + GGML_MAX_N_THREADS, false);

    size_t start_i = 0;
    if (mask.length() >= 2 && mask[0] == '0' && (mask[1] == 'x' || mask[1] == 'X')) {
        start_i = 2;
    }
    if (start_i >= mask.length()) {
        return false;
    }

    size_t num_digits = mask.length() - start_i;
    if (num_digits > 128) {
        num_digits = 128;
    }

    const size_t end_i = num_digits + start_i;
    for (size_t i = start_i, n = num_digits * 4 - 1; i < end_i; ++i, n -= 4) {
        const char c = mask[i];
        int8_t id = 0;
        if (c >= '0' && c <= '9') {
            id = (int8_t) (c - '0');
        } else if (c >= 'a' && c <= 'f') {
            id = (int8_t) (c - 'a' + 10);
        } else if (c >= 'A' && c <= 'F') {
            id = (int8_t) (c - 'A' + 10);
        } else {
            return false;
        }
        if (n < GGML_MAX_N_THREADS) {
            boolmask[n] = boolmask[n] || ((id & 8) != 0);
        }
        if (n >= 1 && n - 1 < GGML_MAX_N_THREADS) {
            boolmask[n - 1] = boolmask[n - 1] || ((id & 4) != 0);
        }
        if (n >= 2 && n - 2 < GGML_MAX_N_THREADS) {
            boolmask[n - 2] = boolmask[n - 2] || ((id & 2) != 0);
        }
        if (n >= 3 && n - 3 < GGML_MAX_N_THREADS) {
            boolmask[n - 3] = boolmask[n - 3] || ((id & 1) != 0);
        }
    }
    return true;
}

static std::string layer_default_cpu_mask() {
    const char * env_mask = std::getenv("LAYER_COOP_CPU_MASK");
    if (env_mask && env_mask[0]) {
        return env_mask;
    }
#ifdef __ANDROID__
    return "0xff";
#else
    return "0x0";
#endif
}

static bool layer_default_cpu_strict() {
    const char * env_strict = std::getenv("LAYER_COOP_CPU_STRICT");
    if (env_strict && env_strict[0]) {
        return std::atoi(env_strict) != 0;
    }
#ifdef __ANDROID__
    return true;
#else
    return false;
#endif
}

static uint32_t layer_default_poll() {
    const char * env_poll = std::getenv("LAYER_COOP_POLL");
    if (env_poll && env_poll[0]) {
        const int value = std::atoi(env_poll);
        return (uint32_t) std::max(0, std::min(100, value));
    }
    return 50;
}

static ggml_sched_priority layer_default_priority() {
    const char * env_prio = std::getenv("LAYER_COOP_PRIO");
    if (env_prio && env_prio[0]) {
        const int value = std::atoi(env_prio);
        return (ggml_sched_priority) std::max(0, std::min(3, value));
    }
    return GGML_SCHED_PRIO_NORMAL;
}

static LayerThreadpoolPtr layer_attach_threadpool(llama_context * ctx, int32_t threads, const char * tag) {
    std::string mask = layer_default_cpu_mask();
    struct ggml_threadpool_params tpp = ggml_threadpool_params_default(threads);
    if (!layer_parse_cpu_mask(mask, tpp.cpumask)) {
        std::cerr << "[" << tag << "] bad LAYER_COOP_CPU_MASK=" << mask << ", using default affinity\n";
        std::memset(tpp.cpumask, 0, sizeof(tpp.cpumask));
        mask = "0x0";
    }
    tpp.strict_cpu = layer_default_cpu_strict();
    tpp.poll = layer_default_poll();
    tpp.prio = layer_default_priority();

    LayerThreadpoolPtr tp(ggml_threadpool_new(&tpp));
    if (!tp) {
        std::cerr << "[" << tag << "] failed to create ggml threadpool, falling back to backend threads\n";
        llama_set_n_threads(ctx, threads, threads);
        return nullptr;
    }

    llama_attach_threadpool(ctx, tp.get(), nullptr);
    llama_set_n_threads(ctx, threads, threads);
    std::cerr << "[" << tag << "] attached ggml threadpool threads=" << threads
              << " cpu_mask=" << mask
              << " strict=" << (tpp.strict_cpu ? 1 : 0)
              << " poll=" << tpp.poll
              << " prio=" << (int) tpp.prio
              << std::endl;
    return tp;
}
