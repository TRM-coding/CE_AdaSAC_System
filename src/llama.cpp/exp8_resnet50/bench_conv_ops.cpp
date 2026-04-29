#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef RESNET50_USE_ONEDNN
#include <omp.h>
#include "oneapi/dnnl/dnnl.hpp"
#endif

namespace {

struct bench_case {
    std::string name;
    int ic;
    int oc;
    int ih;
    int iw;
    int kh;
    int kw;
    int stride;
    int padding;
};

std::vector<float> make_input_nchw(const bench_case & c) {
    std::vector<float> out(static_cast<size_t>(c.ic * c.ih * c.iw));
    for (size_t i = 0; i < out.size(); ++i) {
        out[i] = static_cast<float>((i % 251) - 125) / 125.0f;
    }
    return out;
}

std::vector<float> make_weight_oihw(const bench_case & c) {
    std::vector<float> out(static_cast<size_t>(c.oc * c.ic * c.kh * c.kw));
    for (size_t i = 0; i < out.size(); ++i) {
        out[i] = static_cast<float>((i % 127) - 63) / 127.0f;
    }
    return out;
}

std::vector<float> make_bias(const bench_case & c) {
    std::vector<float> out(static_cast<size_t>(c.oc), 0.01f);
    return out;
}

std::vector<float> make_svd_v(const bench_case & c, int rank) {
    const int64_t input_len = static_cast<int64_t>(c.ic) * c.kh * c.kw;
    std::vector<float> out(static_cast<size_t>(input_len * rank));
    for (size_t i = 0; i < out.size(); ++i) {
        out[i] = static_cast<float>((i % 193) - 96) / 193.0f;
    }
    return out;
}

std::vector<float> make_svd_u(const bench_case & c, int rank) {
    std::vector<float> out(static_cast<size_t>(rank * c.oc));
    for (size_t i = 0; i < out.size(); ++i) {
        out[i] = static_cast<float>((i % 157) - 78) / 157.0f;
    }
    return out;
}

double median_ms(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

std::vector<float> unfold_input_im2col(
        const bench_case & c,
        const std::vector<float> & input,
        int oh,
        int ow) {
    const int64_t input_len = static_cast<int64_t>(c.ic) * c.kh * c.kw;
    const int64_t n_cols = static_cast<int64_t>(oh) * ow;
    std::vector<float> cols(static_cast<size_t>(input_len * n_cols), 0.0f);

    for (int oy = 0; oy < oh; ++oy) {
        const int iy_base = oy * c.stride - c.padding;
        for (int ox = 0; ox < ow; ++ox) {
            const int ix_base = ox * c.stride - c.padding;
            const int64_t col = ox + static_cast<int64_t>(ow) * oy;
            for (int ic = 0; ic < c.ic; ++ic) {
                const int64_t channel_base = static_cast<int64_t>(c.kw) * c.kh * ic;
                for (int ky = 0; ky < c.kh; ++ky) {
                    const int iy = iy_base + ky;
                    if (iy < 0 || iy >= c.ih) {
                        continue;
                    }
                    const int64_t row_base = channel_base + static_cast<int64_t>(c.kw) * ky;
                    for (int kx = 0; kx < c.kw; ++kx) {
                        const int ix = ix_base + kx;
                        if (ix < 0 || ix >= c.iw) {
                            continue;
                        }
                        const int64_t row = row_base + kx;
                        cols[static_cast<size_t>(row + input_len * col)] =
                                input[static_cast<size_t>(ix + c.iw * (iy + c.ih * ic))];
                    }
                }
            }
        }
    }

    return cols;
}

double bench_ggml_conv(const bench_case & c, int threads, int warmup, int repeat) {
    const int oh = (c.ih + 2 * c.padding - c.kh) / c.stride + 1;
    const int ow = (c.iw + 2 * c.padding - c.kw) / c.stride + 1;
    const auto input = make_input_nchw(c);
    const auto weight = make_weight_oihw(c);

    const size_t bytes =
            384 * 1024 * 1024 +
            sizeof(float) * (input.size() + weight.size() + static_cast<size_t>(c.oc * oh * ow));

    ggml_init_params params = {
        /*.mem_size   =*/ bytes,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };

    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        throw std::runtime_error("ggml_init failed");
    }

    ggml_tensor * input_t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.iw, c.ih, c.ic, 1);
    ggml_tensor * weight_t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.kw, c.kh, c.ic, c.oc);
    std::memcpy(input_t->data, input.data(), sizeof(float) * input.size());
    std::memcpy(weight_t->data, weight.data(), sizeof(float) * weight.size());

    std::vector<double> times;
    times.reserve(repeat);
    for (int i = 0; i < warmup + repeat; ++i) {
        ggml_tensor * out = ggml_conv_2d(ctx, weight_t, input_t, c.stride, c.stride, c.padding, c.padding, 1, 1);
        ggml_cgraph * graph = ggml_new_graph(ctx);
        const auto t0 = std::chrono::steady_clock::now();
        ggml_build_forward_expand(graph, out);
        ggml_graph_compute_with_ctx(ctx, graph, threads);
        const auto t1 = std::chrono::steady_clock::now();
        if (i >= warmup) {
            times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
    }

    ggml_free(ctx);
    return median_ms(times);
}

double bench_im2col_svd_conv(const bench_case & c, int rank, int threads, int warmup, int repeat) {
    const int oh = (c.ih + 2 * c.padding - c.kh) / c.stride + 1;
    const int ow = (c.iw + 2 * c.padding - c.kw) / c.stride + 1;
    const int64_t input_len = static_cast<int64_t>(c.ic) * c.kh * c.kw;
    const int64_t n_cols = static_cast<int64_t>(oh) * ow;
    const auto input = make_input_nchw(c);
    const auto v_data = make_svd_v(c, rank);
    const auto u_data = make_svd_u(c, rank);

    std::vector<double> times;
    times.reserve(repeat);
    for (int i = 0; i < warmup + repeat; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        const std::vector<float> input_cols = unfold_input_im2col(c, input, oh, ow);

        const size_t bytes =
                128 * 1024 * 1024 +
                sizeof(float) * (
                        input_cols.size() +
                        v_data.size() +
                        u_data.size() +
                        static_cast<size_t>(input_len * c.oc) +
                        static_cast<size_t>(c.oc * n_cols));

        ggml_init_params params = {
            /*.mem_size   =*/ bytes,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ false,
        };

        ggml_context * ctx = ggml_init(params);
        if (!ctx) {
            throw std::runtime_error("ggml_init failed for im2col SVD conv");
        }

        ggml_tensor * cols_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, input_len, n_cols);
        ggml_tensor * w_shape = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, input_len, c.oc);
        ggml_tensor * v_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, input_len, rank);
        ggml_tensor * u_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, rank, c.oc);
        if (cols_t == nullptr || w_shape == nullptr || v_t == nullptr || u_t == nullptr) {
            throw std::runtime_error("failed to allocate im2col SVD conv tensors");
        }
        std::memcpy(cols_t->data, input_cols.data(), sizeof(float) * input_cols.size());
        std::memcpy(v_t->data, v_data.data(), sizeof(float) * v_data.size());
        std::memcpy(u_t->data, u_data.data(), sizeof(float) * u_data.size());

        ggml_tensor * out = ggml_mul_mat_svd(ctx, w_shape, v_t, u_t, cols_t, 0);
        ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, out);
        ggml_graph_compute_with_ctx(ctx, graph, threads);
        ggml_free(ctx);

        const auto t1 = std::chrono::steady_clock::now();
        if (i >= warmup) {
            times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
    }

    return median_ms(times);
}

#ifdef RESNET50_USE_ONEDNN
double bench_onednn_conv(const bench_case & c, int threads, int warmup, int repeat) {
    omp_set_dynamic(0);
    omp_set_num_threads(threads);

    using namespace dnnl;
    engine eng(engine::kind::cpu, 0);
    stream s(eng);

    const int oh = (c.ih + 2 * c.padding - c.kh) / c.stride + 1;
    const int ow = (c.iw + 2 * c.padding - c.kw) / c.stride + 1;
    const auto input = make_input_nchw(c);
    const auto weight = make_weight_oihw(c);
    const auto bias = make_bias(c);

    memory::dims src_dims = {1, c.ic, c.ih, c.iw};
    memory::dims wei_dims = {c.oc, c.ic, c.kh, c.kw};
    memory::dims bias_dims = {c.oc};
    memory::dims dst_dims = {1, c.oc, oh, ow};
    memory::dims strides = {c.stride, c.stride};
    memory::dims pads = {c.padding, c.padding};

    auto src_md = memory::desc(src_dims, memory::data_type::f32, memory::format_tag::nchw);
    auto wei_md = memory::desc(wei_dims, memory::data_type::f32, memory::format_tag::oihw);
    auto bias_md = memory::desc(bias_dims, memory::data_type::f32, memory::format_tag::x);
    auto dst_md = memory::desc(dst_dims, memory::data_type::f32, memory::format_tag::any);

    auto pd = convolution_forward::primitive_desc(
            eng,
            prop_kind::forward_inference,
            algorithm::convolution_direct,
            src_md,
            memory::desc(wei_dims, memory::data_type::f32, memory::format_tag::any),
            bias_md,
            dst_md,
            strides,
            pads,
            pads);

    memory src_mem(src_md, eng, const_cast<float *>(input.data()));
    memory wei_user_mem(wei_md, eng, const_cast<float *>(weight.data()));
    memory wei_mem(pd.weights_desc(), eng);
    if (pd.weights_desc() != wei_md) {
        reorder(wei_user_mem, wei_mem).execute(s, wei_user_mem, wei_mem);
        s.wait();
    } else {
        wei_mem = wei_user_mem;
    }
    memory bias_mem(bias_md, eng, const_cast<float *>(bias.data()));
    std::vector<float> dst(static_cast<size_t>(c.oc * oh * ow), 0.0f);
    memory dst_mem(memory::desc(dst_dims, memory::data_type::f32, memory::format_tag::nchw), eng, dst.data());
    convolution_forward conv(pd);

    std::vector<double> times;
    times.reserve(repeat);
    for (int i = 0; i < warmup + repeat; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        conv.execute(s, {{DNNL_ARG_SRC, src_mem}, {DNNL_ARG_WEIGHTS, wei_mem}, {DNNL_ARG_BIAS, bias_mem}, {DNNL_ARG_DST, dst_mem}});
        s.wait();
        const auto t1 = std::chrono::steady_clock::now();
        if (i >= warmup) {
            times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
    }

    return median_ms(times);
}

double bench_fold_svd_conv(const bench_case & c, int rank, int threads, int warmup, int repeat) {
    omp_set_dynamic(0);
    omp_set_num_threads(threads);

    using namespace dnnl;
    engine eng(engine::kind::cpu, 0);
    stream s(eng);

    const int oh = (c.ih + 2 * c.padding - c.kh) / c.stride + 1;
    const int ow = (c.iw + 2 * c.padding - c.kw) / c.stride + 1;
    const auto input = make_input_nchw(c);
    const auto v_data = make_svd_v(c, rank);
    const auto u_data = make_svd_u(c, rank);
    const auto bias = make_bias(c);

    std::vector<float> v_oihw(static_cast<size_t>(rank * c.ic * c.kh * c.kw));
    for (int r = 0; r < rank; ++r) {
        for (int ic = 0; ic < c.ic; ++ic) {
            for (int ky = 0; ky < c.kh; ++ky) {
                for (int kx = 0; kx < c.kw; ++kx) {
                    const int64_t row = kx + static_cast<int64_t>(c.kw) * (ky + c.kh * ic);
                    v_oihw[static_cast<size_t>(kx + c.kw * (ky + c.kh * (ic + c.ic * r)))] =
                            v_data[static_cast<size_t>(row + static_cast<int64_t>(c.ic) * c.kh * c.kw * r)];
                }
            }
        }
    }

    std::vector<float> u_oihw(static_cast<size_t>(c.oc * rank));
    for (int oc = 0; oc < c.oc; ++oc) {
        for (int r = 0; r < rank; ++r) {
            u_oihw[static_cast<size_t>(r + rank * oc)] = u_data[static_cast<size_t>(r + rank * oc)];
        }
    }

    memory::dims src_dims = {1, c.ic, c.ih, c.iw};
    memory::dims v_dims = {rank, c.ic, c.kh, c.kw};
    memory::dims rank_dims = {1, rank, oh, ow};
    memory::dims u_dims = {c.oc, rank, 1, 1};
    memory::dims bias_dims = {c.oc};
    memory::dims dst_dims = {1, c.oc, oh, ow};
    memory::dims v_strides = {c.stride, c.stride};
    memory::dims v_pads = {c.padding, c.padding};
    memory::dims u_strides = {1, 1};
    memory::dims u_pads = {0, 0};

    auto src_md = memory::desc(src_dims, memory::data_type::f32, memory::format_tag::nchw);
    auto v_user_md = memory::desc(v_dims, memory::data_type::f32, memory::format_tag::oihw);
    auto rank_md = memory::desc(rank_dims, memory::data_type::f32, memory::format_tag::any);
    auto u_user_md = memory::desc(u_dims, memory::data_type::f32, memory::format_tag::oihw);
    auto bias_md = memory::desc(bias_dims, memory::data_type::f32, memory::format_tag::x);
    auto dst_md = memory::desc(dst_dims, memory::data_type::f32, memory::format_tag::any);

    auto v_pd = convolution_forward::primitive_desc(
            eng,
            prop_kind::forward_inference,
            algorithm::convolution_direct,
            src_md,
            memory::desc(v_dims, memory::data_type::f32, memory::format_tag::any),
            rank_md,
            v_strides,
            v_pads,
            v_pads);
    auto u_pd = convolution_forward::primitive_desc(
            eng,
            prop_kind::forward_inference,
            algorithm::convolution_direct,
            v_pd.dst_desc(),
            memory::desc(u_dims, memory::data_type::f32, memory::format_tag::any),
            bias_md,
            dst_md,
            u_strides,
            u_pads,
            u_pads);

    memory src_mem(src_md, eng, const_cast<float *>(input.data()));
    memory v_user_mem(v_user_md, eng, v_oihw.data());
    memory v_mem(v_pd.weights_desc(), eng);
    if (v_pd.weights_desc() != v_user_md) {
        reorder(v_user_mem, v_mem).execute(s, v_user_mem, v_mem);
        s.wait();
    } else {
        v_mem = v_user_mem;
    }
    memory u_user_mem(u_user_md, eng, u_oihw.data());
    memory u_mem(u_pd.weights_desc(), eng);
    if (u_pd.weights_desc() != u_user_md) {
        reorder(u_user_mem, u_mem).execute(s, u_user_mem, u_mem);
        s.wait();
    } else {
        u_mem = u_user_mem;
    }
    memory bias_mem(bias_md, eng, const_cast<float *>(bias.data()));

    convolution_forward v_conv(v_pd);
    convolution_forward u_conv(u_pd);

    std::vector<uint8_t> rank_buffer(v_pd.dst_desc().get_size());
    std::vector<uint8_t> dst_buffer(u_pd.dst_desc().get_size());
    memory rank_mem(v_pd.dst_desc(), eng, rank_buffer.data());
    memory dst_mem(u_pd.dst_desc(), eng, dst_buffer.data());

    std::vector<double> times;
    times.reserve(repeat);
    for (int i = 0; i < warmup + repeat; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        v_conv.execute(s, {{DNNL_ARG_SRC, src_mem}, {DNNL_ARG_WEIGHTS, v_mem}, {DNNL_ARG_DST, rank_mem}});
        u_conv.execute(s, {{DNNL_ARG_SRC, rank_mem}, {DNNL_ARG_WEIGHTS, u_mem}, {DNNL_ARG_BIAS, bias_mem}, {DNNL_ARG_DST, dst_mem}});
        s.wait();
        const auto t1 = std::chrono::steady_clock::now();
        if (i >= warmup) {
            times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
    }

    return median_ms(times);
}
#endif

} // namespace

int main() {
    ggml_cpu_init();
    const int threads = 8;
    const int warmup = 5;
    const int repeat = 20;
    const std::vector<bench_case> cases = {
            {"7x7 stem", 3, 64, 224, 224, 7, 7, 2, 3},
            {"3x3 block", 128, 128, 28, 28, 3, 3, 1, 1},
            {"1x1 bottleneck", 256, 1024, 14, 14, 1, 1, 1, 0},
    };
    const std::vector<bench_case> svd_cases = {
            {"1x1 kernel", 128, 256, 28, 28, 1, 1, 1, 0},
            {"3x3 kernel", 128, 256, 28, 28, 3, 3, 1, 1},
            {"5x5 kernel", 128, 256, 28, 28, 5, 5, 1, 2},
            {"7x7 kernel", 128, 256, 28, 28, 7, 7, 1, 3},
    };

    for (const auto & c : cases) {
        const double ggml_ms = bench_ggml_conv(c, threads, warmup, repeat);
#ifdef RESNET50_USE_ONEDNN
        const double onednn_ms = bench_onednn_conv(c, threads, warmup, repeat);
#else
        const double onednn_ms = -1.0;
#endif
        std::cout << c.name
                  << "\tggml_ms=" << ggml_ms
                  << "\tonednn_ms=" << onednn_ms
                  << "\n";
    }
    std::cout << std::fixed << std::setprecision(6);
    for (const auto & c : svd_cases) {
        const int input_len = c.ic * c.kh * c.kw;
        const int rank = std::max(1, std::min(c.oc, input_len) / 2);
        const double im2col_svd_ms = bench_im2col_svd_conv(c, rank, threads, warmup, repeat);
#ifdef RESNET50_USE_ONEDNN
        const double fold_svd_ms = bench_fold_svd_conv(c, rank, threads, warmup, repeat);
#else
        const double fold_svd_ms = -1.0;
#endif
        std::cout << "svd_" << c.name
                  << "\trank=" << rank
                  << "\tim2col_svd_ms=" << im2col_svd_ms
                  << "\tfold_svd_ms=" << fold_svd_ms
                  << "\tfold_speedup=" << (fold_svd_ms > 0.0 ? im2col_svd_ms / fold_svd_ms : -1.0)
                  << "\n";
    }
    return 0;
}
