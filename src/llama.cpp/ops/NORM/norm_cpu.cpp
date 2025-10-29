#include "ops.h"

void NORM(
    ggml_tensor * input,
    ggml_tensor *&out,
    float eps,
    ggml_context *ctx)
{
    out = ggml_norm(ctx,input,eps);
    return;
}

ggml_tensor * RUN_NORM(
    int times,
    ggml_type type_src,
    const std::array<int64_t, 4UL> &ne,
    OPS_INFO &info)
{
    ggml_init_params params{};
    params.mem_size = 16 * 1024 * 1024;
    params.no_alloc = true;
    ggml_context *ctx = ggml_init(params);
    if (!ctx)
    {
        std::cout << "ggml_init failed!" << std::endl;
        return nullptr;
    }

    ggml_tensor *src = ggml_new_tensor(ctx, type_src, 4, ne.data());
    ggml_tensor *out = nullptr;

    auto N = 1;
    for (auto x : ne)
        N *= x;
    auto eps = 1e-5f;

    NORM(src, out, eps,ctx);

    ggml_backend_t be = ggml_backend_cpu_init();
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (!buf)
    {
        std::cout << "ggml_backend_alloc_ctx_tensors failed!" << std::endl;
        ggml_backend_free(be);
        return nullptr;
    }

    std::vector<float> h(N);
    for (int i = 0; i < N; i++)
        h[i] = float(i);
    ggml_backend_tensor_set(src, h.data(), 0, sizeof(float) * N);

    double avg_excute_time = 0;
    struct timespec start, end;

    for (int i = 0; i < times; i++)
    {
        ggml_cgraph *gf = ggml_new_graph(ctx);

        ggml_build_forward_expand(gf, out);
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &start);
        ggml_backend_graph_compute(be, gf);
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &end);
        avg_excute_time += (end.tv_sec - start.tv_sec) * 1e3 + (end.tv_nsec - start.tv_nsec) / 1e6;;
    }
    avg_excute_time /= times;
    info.time_per_op_ms = avg_excute_time;
    // std::printf("CPY avg execution time over %d iterations: %.3f ms\n ", times, avg_excute_time);
    ggml_backend_buffer_free(buf);  
    ggml_backend_free(be);
    ggml_free(ctx);
    return out;
}
