#include <pybind11/pybind11.h>
#include <pybind11/stl.h>     // 支持 std::array/std::vector 映射
#include <ggml.h>
#include <ggml-cpu.h>
#include <array>
#include <iostream>
#include <vector>
#include <cmath>

namespace py = pybind11;

// 你的已有声明
ggml_tensor * RUN_ADD(int times,
             ggml_type type,
             const std::array<int64_t, 4UL>& ne);

void RUN_CPY(
    int times,
    ggml_type type_src,
    ggml_type type_dst,
    const std::array<int64_t,4UL>& ne);

void RUN_FLASH_ATTN_EXT(
    int times,
    int64_t hsk,
    int64_t hsv,
    int64_t nh,
    std::array<int64_t, 2UL> nr23,
    int64_t kv,
    int64_t nb,
    bool mask = true,
    bool sinks = false,
    float max_bias = 0.0f,
    float logit_softcap = 0.0f,
    ggml_prec prec = GGML_PREC_F32,
    ggml_type type_KV = GGML_TYPE_F16);


    