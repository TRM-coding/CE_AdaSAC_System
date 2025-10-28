#include "ops.h"

void CPY(
    ggml_tensor * src,
    ggml_tensor * dst,
    ggml_tensor * & out,
    ggml_context * ctx)
{
    out = ggml_cpy(ctx, src, dst);
    return;
}

void RUN_CPY(
    int times,
    ggml_type type_src,
    ggml_type type_dst,
    const std::array<int64_t,4UL>& ne)
{
    ggml_init_params params{};
    params.mem_size = 16 * 1024 * 1024;
    params.no_alloc = true;
    ggml_context * ctx = ggml_init(params);
    if(!ctx)
    {
        std::cout<<"ggml_init failed!"<<std::endl;
        return;
    }

    
    ggml_tensor * src = ggml_new_tensor(ctx, type_src, 4, ne.data());
    ggml_tensor * dst = ggml_new_tensor(ctx, type_dst, 4, src->ne);
    ggml_tensor * out = nullptr;

    auto N =1;
    for(auto x: ne)N*=x;

    CPY(src,dst,out,ctx);

    ggml_backend_t be = ggml_backend_cpu_init();       
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (!buf) 
    {
        std::cout<<"ggml_backend_alloc_ctx_tensors failed!"<<std::endl;
        ggml_backend_free(be);
        return ;
    }

    std::vector<float>h(N);
    for(int i=0;i<N;i++)h[i]=float(i);
    ggml_backend_tensor_set(src, h.data(), 0, sizeof(float)*N);
    for(int i=0;i<times;i++)
    {
        ggml_cgraph* gf = ggml_new_graph(ctx);
        
        ggml_build_forward_expand(gf, out);
        ggml_graph_print(gf);
        ggml_backend_graph_compute(be, gf);
        for(int i=0;i<ggml_nelements(dst);i++)
        {
            auto v = ggml_get_f32_1d(dst, i);
            std::cout<<"dst["<<i<<"]="<<v<<" ";
        }
        std::cout<<std::endl;
    }
    ggml_backend_buffer_free(buf);
    ggml_backend_free(be);
    ggml_free(ctx);
    return;
}

