#include "../ops.h"

void MUL(ggml_tensor * a,
         ggml_tensor * b, 
         ggml_tensor *& c,
         ggml_context * ctx) {

    c = ggml_mul(ctx, a, b);
}

ggml_tensor * RUN_MUL(int times,
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
    
    // 步骤1: 先创建 tensor (只是定义,还没有内存)
    ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
    ggml_tensor * b = ggml_new_tensor_1d(ctx, type, 1);
    
    ggml_tensor * out = nullptr;

    //步骤2 ：创建计算图到context中
    MUL(a, b, out, ctx);
    
    // 步骤3：创建 backend 并分配内存 (这会为 ctx 中所有 tensor 分配内存,包括 out)
    ggml_backend_t be = ggml_backend_cpu_init();       
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (!buf) 
    {
        std::cout<<"ggml_backend_alloc_ctx_tensors failed!"<<std::endl;
        ggml_backend_free(be);
        return nullptr;
    }
    
    // 步骤4：设置数据 (内存已分配)
    std::vector<float> h(N);
    
    // init a with elements = 1
    for (int64_t i = 0; i < N; ++i) h[i] = float(1);
    ggml_backend_tensor_set(a, h.data(), 0, N*sizeof(float));
    
    // init b with element = 2
    h[0] = float(2);  // b 只有1个元素
    ggml_backend_tensor_set(b, h.data(), 0, sizeof(float));

    // 步骤5: 执行计算
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
