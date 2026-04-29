#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Args {
    int threads = 8;
    int warmup = 5;
    int repeat = 30;
    int n_cols = 1;
    ggml_type type = GGML_TYPE_F32;
};

struct CaseSpec {
    std::string name;
    int64_t in;
    int64_t out;
    int64_t mid;
};

struct BenchResult {
    double avg_ms = 0.0;
    double median_ms = 0.0;
    std::vector<float> output;
};

double now_ms() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

ggml_type parse_type(const std::string & value) {
    if (value == "f32") {
        return GGML_TYPE_F32;
    }
    if (value == "f16") {
        return GGML_TYPE_F16;
    }
    if (value == "q4_0") {
        return GGML_TYPE_Q4_0;
    }
    throw std::runtime_error("unsupported --type: " + value + " (expected f32, f16, or q4_0)");
}

std::string type_name(ggml_type type) {
    return ggml_get_type_traits(type)->type_name;
}

Args parse_args(int argc, char ** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        auto need_value = [&](const char * name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + name);
            }
            return argv[++i];
        };

        if (key == "--threads") {
            args.threads = std::stoi(need_value("--threads"));
        } else if (key == "--warmup") {
            args.warmup = std::stoi(need_value("--warmup"));
        } else if (key == "--repeat") {
            args.repeat = std::stoi(need_value("--repeat"));
        } else if (key == "--n-cols") {
            args.n_cols = std::stoi(need_value("--n-cols"));
        } else if (key == "--type") {
            args.type = parse_type(need_value("--type"));
        } else if (key == "--help" || key == "-h") {
            std::cout
                << "usage: bench_svd_mul_mat_transformer [--threads N] [--warmup N] [--repeat N]\n"
                << "                                     [--n-cols N] [--type f32|f16|q4_0]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + key);
        }
    }
    if (args.threads <= 0 || args.warmup < 0 || args.repeat <= 0 || args.n_cols <= 0) {
        throw std::runtime_error("invalid numeric argument");
    }
    return args;
}

std::vector<float> random_vector(size_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 0.02f);
    std::vector<float> data(n);
    for (float & v : data) {
        v = dist(rng);
    }
    return data;
}

void fill_tensor_from_float(ggml_tensor * tensor, const std::vector<float> & src) {
    const int64_t rows = tensor->ne[1];
    const int64_t cols = tensor->ne[0];
    if (static_cast<int64_t>(src.size()) != rows * cols) {
        throw std::runtime_error("source size does not match tensor shape");
    }

    if (tensor->type == GGML_TYPE_F32) {
        std::memcpy(tensor->data, src.data(), src.size() * sizeof(float));
        return;
    }
    if (tensor->type == GGML_TYPE_F16) {
        ggml_fp32_to_fp16_row(src.data(), static_cast<ggml_fp16_t *>(tensor->data), src.size());
        return;
    }

    const ggml_type_traits * traits = ggml_get_type_traits(tensor->type);
    if (traits == nullptr || traits->from_float_ref == nullptr) {
        throw std::runtime_error("tensor type cannot be filled from float");
    }
    if (cols % traits->blck_size != 0) {
        throw std::runtime_error("quantized tensor row length is not block-aligned");
    }
    ggml_quantize_init(tensor->type);
    const size_t row_size = ggml_row_size(tensor->type, cols);
    for (int64_t r = 0; r < rows; ++r) {
        uint8_t * dst = static_cast<uint8_t *>(tensor->data) + static_cast<size_t>(r) * row_size;
        traits->from_float_ref(src.data() + static_cast<size_t>(r) * cols, dst, cols);
    }
}

size_t context_bytes(const CaseSpec & spec, int64_t n_cols, ggml_type type) {
    const size_t v_bytes = ggml_row_size(type, spec.in) * static_cast<size_t>(spec.mid);
    const size_t u_bytes = ggml_row_size(type, spec.mid) * static_cast<size_t>(spec.out);
    const size_t input_bytes = sizeof(float) * static_cast<size_t>(spec.in * n_cols);
    const size_t output_bytes = sizeof(float) * static_cast<size_t>(spec.out * n_cols);
    const size_t tmp_bytes = sizeof(float) * static_cast<size_t>(spec.mid * n_cols);
    const size_t shape_bytes = sizeof(float) * static_cast<size_t>(spec.in * spec.out);
    return 64 * 1024 * 1024 + 2 * (v_bytes + u_bytes + input_bytes + output_bytes + tmp_bytes + shape_bytes);
}

BenchResult run_case(
        const CaseSpec & spec,
        const Args & args,
        bool fused,
        const std::vector<float> & v_data,
        const std::vector<float> & u_data,
        const std::vector<float> & input_data) {
    ggml_init_params params {
        /*.mem_size   =*/ context_bytes(spec, args.n_cols, args.type),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };

    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        throw std::runtime_error("ggml_init failed");
    }

    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, spec.in, args.n_cols);
    ggml_tensor * v = ggml_new_tensor_2d(ctx, args.type, spec.in, spec.mid);
    ggml_tensor * u = ggml_new_tensor_2d(ctx, args.type, spec.mid, spec.out);
    ggml_tensor * w_shape = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, spec.in, spec.out);
    if (input == nullptr || v == nullptr || u == nullptr || w_shape == nullptr) {
        ggml_free(ctx);
        throw std::runtime_error("ggml tensor allocation failed");
    }

    fill_tensor_from_float(input, input_data);
    fill_tensor_from_float(v, v_data);
    fill_tensor_from_float(u, u_data);

    ggml_tensor * output = nullptr;
    if (fused) {
        output = ggml_mul_mat_svd(ctx, w_shape, v, u, input, 0);
    } else {
        ggml_tensor * tmp = ggml_mul_mat(ctx, v, input);
        output = ggml_mul_mat(ctx, u, tmp);
    }

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);

    auto compute = [&]() {
        const ggml_status status = ggml_graph_compute_with_ctx(ctx, graph, args.threads);
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("ggml_graph_compute_with_ctx failed");
        }
    };

    for (int i = 0; i < args.warmup; ++i) {
        compute();
    }

    std::vector<double> times;
    times.reserve(args.repeat);
    for (int i = 0; i < args.repeat; ++i) {
        const double t0 = now_ms();
        compute();
        times.push_back(now_ms() - t0);
    }

    std::vector<double> sorted = times;
    std::sort(sorted.begin(), sorted.end());
    double sum = 0.0;
    for (double t : times) {
        sum += t;
    }

    const size_t output_count = static_cast<size_t>(spec.out * args.n_cols);
    const float * output_ptr = static_cast<const float *>(output->data);
    BenchResult result;
    result.avg_ms = sum / static_cast<double>(times.size());
    result.median_ms = sorted[sorted.size() / 2];
    result.output.assign(output_ptr, output_ptr + output_count);

    ggml_free(ctx);
    return result;
}

double max_abs_diff(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) {
        throw std::runtime_error("output size mismatch");
    }
    double max_diff = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        max_diff = std::max(max_diff, static_cast<double>(std::fabs(a[i] - b[i])));
    }
    return max_diff;
}

} // namespace

int main(int argc, char ** argv) {
    try {
        const Args args = parse_args(argc, argv);
        const std::vector<CaseSpec> cases = {
            {"square_512",              512,  512,  512},
            {"square_1024",            1024, 1024, 1024},
            {"square_1536",            1536, 1536, 1536},
            {"qwen_ffn_up_gate_full",  1536, 8960, 1536},
            {"qwen_ffn_down_full",     8960, 1536, 1536},
        };

        std::cout
            << "# threads=" << args.threads
            << " warmup=" << args.warmup
            << " repeat=" << args.repeat
            << " n_cols=" << args.n_cols
            << " type=" << type_name(args.type) << "\n";
        std::cout
            << "case,in,out,mid_dim,type,n_cols,fused_median_ms,two_mul_mat_median_ms,"
            << "speedup_vs_two_mul_mat,fused_avg_ms,two_mul_mat_avg_ms,max_abs_diff\n";

        for (size_t i = 0; i < cases.size(); ++i) {
            const CaseSpec & spec = cases[i];
            const auto v_data = random_vector(static_cast<size_t>(spec.in * spec.mid), 1000 + static_cast<uint32_t>(i));
            const auto u_data = random_vector(static_cast<size_t>(spec.mid * spec.out), 2000 + static_cast<uint32_t>(i));
            const auto input_data = random_vector(static_cast<size_t>(spec.in * args.n_cols), 3000 + static_cast<uint32_t>(i));

            const BenchResult fused = run_case(spec, args, true, v_data, u_data, input_data);
            const BenchResult two = run_case(spec, args, false, v_data, u_data, input_data);
            const double speedup = two.median_ms / fused.median_ms;
            const double diff = max_abs_diff(fused.output, two.output);

            std::cout << std::fixed << std::setprecision(6)
                << spec.name << ','
                << spec.in << ','
                << spec.out << ','
                << spec.mid << ','
                << type_name(args.type) << ','
                << args.n_cols << ','
                << fused.median_ms << ','
                << two.median_ms << ','
                << speedup << ','
                << fused.avg_ms << ','
                << two.avg_ms << ','
                << diff << '\n';
        }

        ggml_quantize_free();
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}
