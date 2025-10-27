#include "../ops.h"

void ADD(ggml_tensor * a,
         ggml_tensor * b, 
         ggml_tensor * c,
         ggml_context * ctx) {
    
    // ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
    // ggml_set_param(a);
    // ggml_set_name(a, "a");

    // ggml_tensor * b = ggml_new_tensor_1d(ctx, type, 1);
    // ggml_set_param(b); 
    // ggml_set_name(b, "b");

    ggml_tensor * out = ggml_add1(ctx, a, b);
    // ggml_set_name(out, "out");
}

void RUN_ADD(int times,
             ggml_type type,
             const std::array<int64_t, 4UL>& ne,
             ggml_context * ctx)
{
    ggml_tensor * a = ggml_new_tensor(ctx, type, 4, ne.data());
    ggml_tensor * b = ggml_new_tensor_1d(ctx, type, 1);
    ggml_tensor * out;
    for(int i=0 ; i<times;i++)
    {
        ADD(a, b, out, ctx);
    }   
}