#include "ops.h"

void CPY(
    ggml_tensor * src,
    ggml_tensor * dst,
    ggml_context * ctx)
{
    ggml_cpy(ctx, src, dst);
}

void RUN_CPY(
    int times,
    ggml_context * ctx,
    ggml_type type_src,
    ggml_type type_dst,
    const std::array<int64_t,4UL>& ne)
{
    ggml_tensor * src = ggml_new_tensor(ctx, type_src, 4, ne.data());
    ggml_tensor * dst = ggml_new_tensor(ctx, type_dst, 4, src->ne);
    for(int i=0;i<times;i++)
    {
        CPY(src,dst,ctx);
    }
}

