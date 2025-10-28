#include "ops.h"

void FLASH_ATTN_EXT(
    ggml_tensor *q,
    ggml_tensor *k,
    ggml_tensor *v,
    ggml_tensor *mask,
    int64_t nb,
    int64_t kv,
    std::array<int64_t, 2UL> nr23,
    int64_t hsk,
    float max_bias,
    float logit_softcap,
    ggml_prec prec,
    bool sinks,
    ggml_tensor *&out,
    ggml_context *ctx)
{

    ggml_tensor *m = nullptr;
    if (mask)
    {
        m = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, kv, GGML_PAD(nb, GGML_KQ_MASK_PAD), 1, nr23[1]);
        ggml_set_name(m, "m");
    }

    ggml_tensor *s = nullptr;
    if (sinks)
    {
        s = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, q->ne[2]);
        ggml_set_name(s, "s");
    }

    out = ggml_flash_attn_ext(ctx, q, k, v, m, 1.0f / sqrtf(hsk), max_bias, logit_softcap);
    ggml_flash_attn_ext_add_sinks(out, s);
    ggml_flash_attn_ext_set_prec(out, prec);
    ggml_set_name(out, "out");
    
    return;
}

void RUN_FLASH_ATTN_EXT(
    int times,
    int64_t hsk,
    int64_t hsv,
    int64_t nh,
    std::array<int64_t, 2UL> nr23,
    int64_t kv,
    int64_t nb,
    bool mask,
    bool sinks ,
    float max_bias ,
    float logit_softcap ,
    ggml_prec prec,
    ggml_type type_KV)
{
    ggml_init_params params{};
    params.mem_size = 16 * 1024 * 1024;
    params.no_alloc = true;
    ggml_context *ctx = ggml_init(params);
    if (!ctx)
    {
        std::cout << "ggml_init failed!" << std::endl;
        return;
    }

    const int64_t hsk_padded = GGML_PAD(hsk, ggml_blck_size(type_KV));
    const int64_t hsv_padded = GGML_PAD(hsv, ggml_blck_size(type_KV));
    ggml_tensor *q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hsk_padded, nb, nh * nr23[0], nr23[1]);
    ggml_tensor *k = ggml_new_tensor_4d(ctx, type_KV, hsk_padded, kv, nh, nr23[1]); // the K tensor is usually a view of the K cache
    ggml_tensor *v = ggml_new_tensor_4d(ctx, type_KV, hsv_padded, kv, nh, nr23[1]); // the V tensor is usually a view of the V cache
    ggml_tensor *out = nullptr;
    FLASH_ATTN_EXT(
        q,
        k,
        v,
        mask ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, nb * kv) : nullptr,
        nb,
        kv,
        nr23,
        hsk,
        max_bias,
        logit_softcap,
        prec,
        sinks,
        out,
        ctx);

    ggml_backend_t be = ggml_backend_cpu_init();       
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (!buf) 
    {
        std::cout<<"ggml_backend_alloc_ctx_tensors failed!"<<std::endl;
        ggml_backend_free(be);
        return ;
    }
        
    std::vector<float> q_data(ggml_nelements(q));
    std::vector<float> k_data(ggml_nelements(k));
    std::vector<float> v_data(ggml_nelements(v));
    for (size_t i = 0; i < q_data.size(); i++)
    {
        q_data[i] = static_cast<float>(i % 13) * 0.1f;
    }
    for (size_t i = 0; i < k_data.size(); i++)
    {
        k_data[i] = static_cast<float>(i % 7) * 0.1f;
    }
    for (size_t i = 0; i < v_data.size(); i++)
    {
        v_data[i] = static_cast<float>(i % 5) * 0.1f;
    }

    for (int i = 0; i < times; i++)
    {
        ggml_cgraph* gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        ggml_graph_print(gf);
        ggml_backend_graph_compute(be, gf);
        for(int i=0;i<ggml_nelements(out);i++)
        {
            auto v = ggml_get_f32_1d(out, i);
            // std::cout<<"out["<<i<<"]="<<v<<" ";
        }
        std::cout<<std::endl;
    }
}
