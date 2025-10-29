#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "ops.h"
namespace py = pybind11;

// 工具宏：用 C/C++ 枚举量本名作为 Python 名称，避免手写字符串
#define ENUM_VAL(e) .value(#e, e)

PYBIND11_MODULE(opslib, m) {
    // 1) 导出 OPS_INFO 结构体
    py::class_<OPS_INFO>(m, "OPS_INFO")
        .def(py::init<>())
        .def_readwrite("time_per_op_ms", &OPS_INFO::time_per_op_ms);

    // 2) 直接导出 ggml 的枚举类型与值（名字=头文件里的 token）
    py::class_<ggml_tensor>(m, "ggml_tensor");
    py::enum_<ggml_prec>(m, "ggml_prec")
        ENUM_VAL(GGML_PREC_DEFAULT)
        ENUM_VAL(GGML_PREC_F32)
        /* 如有更多，继续列出 */
        .export_values();

    py::enum_<ggml_type>(m, "ggml_type")
        ENUM_VAL(GGML_TYPE_F32)
        ENUM_VAL(GGML_TYPE_F16)
        /* … */
        .export_values();

    // 3) 绑定 run_add - 返回 OPS_INFO
    m.def("run_add", [](int times, ggml_type type, const std::array<int64_t, 4UL>& ne) {
        OPS_INFO info;
        RUN_ADD(times, type, ne, info);
        return info;  // 返回性能信息
    },
        py::arg("times"),
        py::arg("type"),
        py::arg("ne"),
        "Run ggml add operation");

    // 4) 绑定 run_cpy - 返回 OPS_INFO
    m.def("run_cpy", [](int times, ggml_type type_src, ggml_type type_dst, const std::array<int64_t, 4UL>& ne) {
        OPS_INFO info;
        RUN_CPY(times, type_src, type_dst, ne, info);
        return info;  // 返回性能信息
    },
        py::arg("times"),
        py::arg("type_src"),
        py::arg("type_dst"),
        py::arg("ne"),
        "Run ggml copy operation");

    // 5) 绑定 run_flash_attn_ext - 返回 OPS_INFO
    m.def("run_flash_attn_ext", [](
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
        ggml_type type_KV = GGML_TYPE_F16) {
        OPS_INFO info;
        RUN_FLASH_ATTN_EXT(times, hsk, hsv, nh, nr23, kv, nb, info, mask, sinks, max_bias, logit_softcap, prec, type_KV);
        return info;  // 返回性能信息
    },
        py::arg("times"),
        py::arg("hsk"),
        py::arg("hsv"),
        py::arg("nh"),
        py::arg("nr23"),
        py::arg("kv"),
        py::arg("nb"),
        py::arg("mask") = true,
        py::arg("sinks") = false,
        py::arg("max_bias") = 0.0f,
        py::arg("logit_softcap") = 0.0f,
        py::arg("prec") = GGML_PREC_F32,    
        py::arg("type_KV") = GGML_TYPE_F16,
        "Run Flash Attention ext operation");

    m.def("run_gelu", [](
        int times,
        ggml_type type_src,
        std::array<int64_t, 4UL> ne) {
        OPS_INFO info;
        ggml_tensor * out = RUN_GELU(times, type_src, ne, info);
        return info;  // 返回性能信息
    },
    py::arg("times"),
    py::arg("type_src"),
    py::arg("ne"),
    "Run GELU operation");

    m.def("run_mul", [](int times,
                    ggml_type type,
                    const std::array<int64_t, 4UL>& ne) {
        OPS_INFO info;
        RUN_MUL(times, type, ne, info);
        return info;  // 返回性能信息
    },
    py::arg("times"),
    py::arg("type"),
    py::arg("ne"),
    "Run ggml MUL operation");

    
    // 12) 绑定 run_scale - 返回 OPS_INFO
    m.def("run_scale", [](int times, ggml_type type, const std::array<int64_t, 4UL>& ne, float scale, float bias) {
        OPS_INFO info;
        RUN_SCALE(times, type, ne, scale, bias, info);
        return info;  // 返回性能信息
    },
        py::arg("times"),
        py::arg("type"),
        py::arg("ne"),
        py::arg("scale"),
        py::arg("bias"),
        "Run ggml SCALE operation");
    
    m.def("run_mul_mat", [](int times,
                        ggml_type type,
                        const std::array<int64_t, 4UL>& ne) {
        OPS_INFO info;
        RUN_MUL_MAT(times, type, ne, info);
        return info;  // 返回性能信息
    },
    py::arg("times"),
    py::arg("type"),
    py::arg("ne"),
    "Run ggml MUL_MAT operation");

    // 13) 绑定 run_swiglu - 返回 OPS_INFO
    m.def("run_swiglu", [](int times, ggml_type type, const std::array<int64_t, 4UL>& ne) {
        OPS_INFO info;
        RUN_SWIGLU(times, type, ne, info);
        return info;  // 返回性能信息
    },
        py::arg("times"),
        py::arg("type"),
        py::arg("ne"),
        "Run ggml MUL_MAT operation");
    
    m.def("run_permute", [](
        int times,
        ggml_type type_src,
        std::array<int64_t, 4UL> ne,
        std::array<int, 4UL> permute_axes) {
        OPS_INFO info;
        ggml_tensor * out = RUN_PERMUTE(times, type_src, ne, permute_axes, info);
        return info;  // 返回性能信息
    },
    py::arg("times"),
    py::arg("type_src"),
    py::arg("ne"),
    py::arg("permute_axes"),
    "Run Permute operation");


    m.def("run_reshape", [](
        int times,
        ggml_type type_src,
        std::array<int64_t, 4UL> ne,
        std::array<int64_t, 4UL> shape_size) {
        OPS_INFO info;
        ggml_tensor * out = RUN_RESHAPE(times, type_src, ne, shape_size, info);
        return info;  // 返回性能信息
    },
    py::arg("times"),
    py::arg("type_src"),
    py::arg("ne"),
    py::arg("shape_size"),
    "Run Reshape operation");

    m.def("run_norm", [](int times,
                     ggml_type type_src,
                     const std::array<int64_t, 4UL>& ne) {
        OPS_INFO info;
        RUN_NORM(times, type_src, ne, info);
        return info;  // 返回性能信息
    },
    py::arg("times"),
    py::arg("type_src"),
    py::arg("ne"),
    "Run ggml NORM operation");

    m.def("run_rms_norm", [](int times,
                     ggml_type type_src,
                     const std::array<int64_t, 4UL>& ne) {
        OPS_INFO info;
        RUN_RMS_NORM(times, type_src, ne, info);
        return info;  // 返回性能信息
    },
    py::arg("times"),
    py::arg("type_src"),
    py::arg("ne"),
    "Run ggml RMS operation");

    m.def("run_view_1d", [](int times,
                            ggml_type type_src,
                            const std::array<int64_t, 4UL>& ne,
                            const std::array<int, 1UL>& view_axes) {
            OPS_INFO info;
            RUN_VIEW_1D(times, type_src, ne, view_axes, info);
            return info;
        },
        py::arg("times"),
        py::arg("type_src"),
        py::arg("ne"),
        py::arg("view_axes"),
        "Run ggml VIEW_1D operation");


    // VIEW_2D
    m.def("run_view_2d", [](int times,
                            ggml_type type_src,
                            const std::array<int64_t, 4UL>& ne,
                            const std::array<int, 2UL>& view_axes) {
            OPS_INFO info;
            RUN_VIEW_2D(times, type_src, ne, view_axes, info);
            return info;
        },
        py::arg("times"),
        py::arg("type_src"),
        py::arg("ne"),
        py::arg("view_axes"),
        "Run ggml VIEW_2D operation");


    // VIEW_3D
    m.def("run_view_3d", [](int times,
                            ggml_type type_src,
                            const std::array<int64_t, 4UL>& ne,
                            const std::array<int, 3UL>& view_axes) {
            OPS_INFO info;
            RUN_VIEW_3D(times, type_src, ne, view_axes, info);
            return info;
        },
        py::arg("times"),
        py::arg("type_src"),
        py::arg("ne"),
        py::arg("view_axes"),
        "Run ggml VIEW_3D operation");


    // VIEW_4D
    m.def("run_view_4d", [](int times,
                            ggml_type type_src,
                            const std::array<int64_t, 4UL>& ne,
                            const std::array<int, 4UL>& view_axes) {
            OPS_INFO info;
            RUN_VIEW_4D(times, type_src, ne, view_axes, info);
            return info;
        },
        py::arg("times"),
        py::arg("type_src"),
        py::arg("ne"),
        py::arg("view_axes"),
        "Run ggml VIEW_4D operation");

    // 14) 绑定 run_swiglu_oai - 返回 OPS_INFO
    m.def("run_swiglu_oai", [](int times, ggml_type type, const std::array<int64_t, 4UL>& ne_a, float alpha, float limit) {
        OPS_INFO info;
        RUN_SWIGLU_OAI(times, type, ne_a, alpha, limit, info);
        return info;  // 返回性能信息
    },
        py::arg("times"),
        py::arg("type"),
        py::arg("ne_a"),
        py::arg("alpha"),
        py::arg("limit"),
        "Run ggml siwglu oai operation");


    // 可选：把枚举量同时曝为模块常量（同名，便于旧代码）
    m.attr("GGML_PREC_F32") = py::int_(static_cast<int>(GGML_PREC_F32));
    m.attr("GGML_TYPE_F16") = py::int_(static_cast<int>(GGML_TYPE_F16));
}
#undef ENUM_VAL