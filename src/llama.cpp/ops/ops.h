#include <ggml.h>
#include<array>
#include<ggml-cpu.h>
ggml_tensor * RUN_ADD(int times,
             ggml_type type,
             const std::array<int64_t, 4UL>& ne,
             ggml_context * ctx);

void RUN_CPY(
    int times,
    ggml_context * ctx,
    ggml_type type_src,
    ggml_type type_dst,
    const std::array<int64_t,4UL>& ne);