#include "../ops.h"
#include<iostream>
#include<vector>

void ADD(ggml_tensor * a,
         ggml_tensor * b, 
         ggml_tensor *& c,
         ggml_context * ctx) {

    c = ggml_add(ctx, a, b);
}

ggml_tensor * RUN_ADD(int times,
             ggml_type type,
             const std::array<int64_t, 4UL>& ne,
             ggml_context * ctx)
{
    const int64_t N = ne[0]*ne[1]*ne[2]*ne[3];
    
    // 步骤1: 先创建 tensor (只是定义,还没有内存)
    ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
    ggml_tensor * b = ggml_new_tensor_1d(ctx, type, 1);
    
    // 创建 out tensor (在分配内存之前)
    ggml_tensor * out = nullptr;
    ADD(a, b, out, ctx);
    
    // 步骤2: 创建 backend 并分配内存 (这会为 ctx 中所有 tensor 分配内存,包括 out)
    ggml_backend_t be = ggml_backend_cpu_init();       
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (!buf) 
    {
        std::cout<<"ggml_backend_alloc_ctx_tensors failed!"<<std::endl;
        ggml_backend_free(be);
        return nullptr;
    }
    
    // 步骤3: 现在才能设置数据 (内存已分配)
    std::vector<float> h(N);
    
    // init a with elements = 1
    for (int64_t i = 0; i < N; ++i) h[i] = float(1);
    ggml_backend_tensor_set(a, h.data(), 0, N*sizeof(float));
    
    // init b with element = 2
    h[0] = float(2);  // b 只有1个元素
    ggml_backend_tensor_set(b, h.data(), 0, sizeof(float));

    // 步骤4: 执行计算
    for(int i=0; i<times; i++)
    {
        ggml_cgraph* gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        ggml_backend_graph_compute(be, gf);
        
        std::vector<float> y(N);
        ggml_backend_tensor_get(out, y.data(), 0, N*sizeof(float));
        std::printf("Iteration %d: y[0]=%.1f y[1]=%.1f y[last]=%.1f\n", i, y[0], y[1], y.back());
    }
    
    ggml_backend_buffer_free(buf);
    ggml_backend_free(be);
    return out;
}
