#include "../ops.h"
#include <cstdint>

void SWIGLU(ggml_tensor * a,
         ggml_tensor *& c,
         ggml_context * ctx) {

    c = ggml_swiglu(ctx, a);
}

ggml_tensor * RUN_SWIGLU(int times,
             ggml_type type,
             const std::array<int64_t, 4UL>& ne,
             OPS_INFO& info)
{
    ggml_init_params params{};
    params.mem_size = 16 * 1024 * 1024;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    
    if(!ctx)
    {
        std::cout<<"ggml_init failed!"<<std::endl;
        return nullptr;
    }
    
    const int64_t N = ne[0]*ne[1]*ne[2]*ne[3];

    ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
    ggml_tensor * out = nullptr;

    SWIGLU(a, out, ctx);

    ggml_backend_t be = ggml_backend_cpu_init();       
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (!buf) 
    {
        std::cout<<"ggml_backend_alloc_ctx_tensors failed!"<<std::endl;
        ggml_backend_free(be);
        return nullptr;
    }

    std::vector<float> h(N);
    
    for (int64_t i = 0; i < N; ++i)
        h[i] = -5.0f + 10.0f * float(i) / float(N - 1);
    ggml_backend_tensor_set(a, h.data(), 0, N*sizeof(float));

    double avg_excute_time=0;
    struct timespec start, end;
    
    for(int i=0; i<times; i++)
    {
        ggml_cgraph* gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);

        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &start);
        ggml_backend_graph_compute(be, gf);
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &end);

        avg_excute_time += (end.tv_sec - start.tv_sec) * 1e3 + (end.tv_nsec - start.tv_nsec) / 1e6;;
        
        int64_t out_N = out->ne[0] * out->ne[1] * out->ne[2] * out->ne[3];
        std::vector<float> y(out_N);
        ggml_backend_tensor_get(out, y.data(), 0, out_N*sizeof(float));
        // std::printf("Iteration %d: y[0]=%.1f y[1]=%.1f y[last]=%.1f\n", i, y[0], y[1], y.back());
    }
    avg_excute_time /= times;
    info.time_per_op_ms = avg_excute_time;
    // std::printf("ADD avg execution time over %d iterations: %.3f ms\n ", times, avg_excute_time);
    
    ggml_backend_buffer_free(buf);
    ggml_backend_free(be);
    ggml_free(ctx);
    return out;
}