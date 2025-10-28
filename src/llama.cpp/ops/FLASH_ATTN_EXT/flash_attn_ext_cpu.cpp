// #include "ops.h"

// void MASKED_FLASH_ATTN_EXT(
//     ggml_tensor *q,
//     ggml_tensor *k,
//     ggml_tensor *v,
//     ggml_tensor * mask,
//     ggml_context *ctx)
// {

//     ggml_tensor *m = nullptr;
//     if (mask)
//     {
//         m = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, kv, GGML_PAD(nb, GGML_KQ_MASK_PAD), 1, nr23[1]);
//         ggml_set_name(m, "m");
//     }

//     ggml_tensor *s = nullptr;
//     if (sinks)
//     {
//         s = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, q->ne[2]);
//         ggml_set_name(s, "s");
//     }

//     ggml_tensor *out = ggml_flash_attn_ext(ctx, q, k, v, m, 1.0f / sqrtf(hsk), max_bias, logit_softcap);
//     ggml_flash_attn_ext_add_sinks(out, s);
//     ggml_flash_attn_ext_set_prec(out, prec);
//     ggml_set_name(out, "out");

//     return out;
// }

// void UNMASKED_FLASH_ATTN_EXT(
//     ggml_tensor *q,
//     ggml_tensor *k,
//     ggml_tensor *v,
//     ggml_tensor * mask,
//     ggml_context *ctx)
// {

//     ggml_tensor *m = nullptr;
//     if (mask)
//     {
//         m = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, kv, GGML_PAD(nb, GGML_KQ_MASK_PAD), 1, nr23[1]);
//         ggml_set_name(m, "m");
//     }

//     ggml_tensor *s = nullptr;
//     if (sinks)
//     {
//         s = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, q->ne[2]);
//         ggml_set_name(s, "s");
//     }

//     ggml_tensor *out = ggml_flash_attn_ext(ctx, q, k, v, m, 1.0f / sqrtf(hsk), max_bias, logit_softcap);
//     ggml_flash_attn_ext_add_sinks(out, s);
//     ggml_flash_attn_ext_set_prec(out, prec);
//     ggml_set_name(out, "out");

//     return out;
// }

// void RUN_UNMASKED_FLASH_ATTN_EXT(
//     int times,
//     ggml_context *ctx,
//     int64_t hsk,
//     int64_t hsv,
//     int64_t nh,
//     std::array<int64_t, 2UL> nr23,
//     int64_t kv,
//     int64_t nb,
//     bool mask = true,
//     bool sinks = false,
//     float max_bias = 0.0f,
//     float logit_softcap = 0.0f,
//     ggml_prec prec = GGML_PREC_F32,
//     ggml_type type_KV = GGML_TYPE_F16)
// {
//     const int64_t hsk_padded = GGML_PAD(hsk, ggml_blck_size(type_KV));
//     const int64_t hsv_padded = GGML_PAD(hsv, ggml_blck_size(type_KV));


//     ggml_tensor *q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hsk_padded, nb, nh * nr23[0], nr23[1]);
  

//     ggml_tensor *k = ggml_new_tensor_4d(ctx,type_KV, hsk_padded, kv, nh, nr23[1]); // the K tensor is usually a view of the K cache
    

//     ggml_tensor *v = ggml_new_tensor_4d(ctx,type_KV, hsv_padded, kv, nh, nr23[1]); // the V tensor is usually a view of the V cache
   
//     for (int i = 0; i < times; i++)
//     {
//         FLASH_ATTN_EXT(src, dst, ctx);
//     }
// }
