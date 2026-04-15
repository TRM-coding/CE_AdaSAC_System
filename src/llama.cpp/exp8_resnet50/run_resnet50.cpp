#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

extern "C" void ggml_svd_local_profile_print_and_reset(void) {}

using gguf_ptr = std::unique_ptr<gguf_context, decltype(&gguf_free)>;
using ggml_ctx_ptr = std::unique_ptr<ggml_context, decltype(&ggml_free)>;

struct image_u8 {
    int width = 0;
    int height = 0;
    int channels = 0;
    std::vector<uint8_t> data;
};

struct preproc_config {
    uint32_t image_size = 224;
    uint32_t resize_shorter_edge = 256;
    std::array<float, 3> mean = {0.485f, 0.456f, 0.406f};
    std::array<float, 3> std = {0.229f, 0.224f, 0.225f};
};

struct resnet50_model {
    gguf_ptr meta{nullptr, gguf_free};
    ggml_ctx_ptr weights{nullptr, ggml_free};
    preproc_config preproc;
    std::array<uint32_t, 4> stage_block_count = {3, 4, 6, 3};
    std::vector<std::string> labels;
};

struct args {
    std::string model_path;
    std::string image_path;
    int threads = 8;
    int top_k = 5;
};

int64_t require_key(const gguf_context * meta, const char * key) {
    const int64_t idx = gguf_find_key(meta, key);
    if (idx < 0) {
        throw std::runtime_error(std::string("missing GGUF key: ") + key);
    }
    return idx;
}

std::string get_string_key(const gguf_context * meta, const char * key) {
    return gguf_get_val_str(meta, require_key(meta, key));
}

uint32_t get_u32_key(const gguf_context * meta, const char * key, uint32_t fallback) {
    const int64_t idx = gguf_find_key(meta, key);
    if (idx < 0) {
        return fallback;
    }
    const gguf_type type = gguf_get_kv_type(meta, idx);
    if (type == GGUF_TYPE_UINT32) {
        return gguf_get_val_u32(meta, idx);
    }
    if (type == GGUF_TYPE_INT32) {
        return static_cast<uint32_t>(gguf_get_val_i32(meta, idx));
    }
    if (type == GGUF_TYPE_UINT64) {
        return static_cast<uint32_t>(gguf_get_val_u64(meta, idx));
    }
    if (type == GGUF_TYPE_INT64) {
        return static_cast<uint32_t>(gguf_get_val_i64(meta, idx));
    }
    throw std::runtime_error(std::string("unexpected scalar type for key: ") + key);
}

std::vector<float> get_float_array(const gguf_context * meta, const char * key) {
    const int64_t idx = require_key(meta, key);
    if (gguf_get_kv_type(meta, idx) != GGUF_TYPE_ARRAY) {
        throw std::runtime_error(std::string("key is not an array: ") + key);
    }
    const gguf_type arr_type = gguf_get_arr_type(meta, idx);
    const size_t n = gguf_get_arr_n(meta, idx);
    const void * raw = gguf_get_arr_data(meta, idx);
    std::vector<float> out(n);

    switch (arr_type) {
        case GGUF_TYPE_FLOAT32: {
            const auto * p = static_cast<const float *>(raw);
            std::copy(p, p + n, out.begin());
        } break;
        case GGUF_TYPE_FLOAT64: {
            const auto * p = static_cast<const double *>(raw);
            for (size_t i = 0; i < n; ++i) {
                out[i] = static_cast<float>(p[i]);
            }
        } break;
        default:
            throw std::runtime_error(std::string("unexpected float array type for key: ") + key);
    }

    return out;
}

std::vector<uint32_t> get_u32_array(const gguf_context * meta, const char * key) {
    const int64_t idx = require_key(meta, key);
    if (gguf_get_kv_type(meta, idx) != GGUF_TYPE_ARRAY) {
        throw std::runtime_error(std::string("key is not an array: ") + key);
    }
    const gguf_type arr_type = gguf_get_arr_type(meta, idx);
    const size_t n = gguf_get_arr_n(meta, idx);
    const void * raw = gguf_get_arr_data(meta, idx);
    std::vector<uint32_t> out(n);

    switch (arr_type) {
        case GGUF_TYPE_UINT32: {
            const auto * p = static_cast<const uint32_t *>(raw);
            std::copy(p, p + n, out.begin());
        } break;
        case GGUF_TYPE_INT32: {
            const auto * p = static_cast<const int32_t *>(raw);
            for (size_t i = 0; i < n; ++i) {
                out[i] = static_cast<uint32_t>(p[i]);
            }
        } break;
        case GGUF_TYPE_UINT64: {
            const auto * p = static_cast<const uint64_t *>(raw);
            for (size_t i = 0; i < n; ++i) {
                out[i] = static_cast<uint32_t>(p[i]);
            }
        } break;
        case GGUF_TYPE_INT64: {
            const auto * p = static_cast<const int64_t *>(raw);
            for (size_t i = 0; i < n; ++i) {
                out[i] = static_cast<uint32_t>(p[i]);
            }
        } break;
        default:
            throw std::runtime_error(std::string("unexpected integer array type for key: ") + key);
    }

    return out;
}

std::vector<std::string> get_string_array(const gguf_context * meta, const char * key) {
    const int64_t idx = require_key(meta, key);
    if (gguf_get_kv_type(meta, idx) != GGUF_TYPE_ARRAY || gguf_get_arr_type(meta, idx) != GGUF_TYPE_STRING) {
        throw std::runtime_error(std::string("key is not a string array: ") + key);
    }
    const size_t n = gguf_get_arr_n(meta, idx);
    std::vector<std::string> out;
    out.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        out.emplace_back(gguf_get_arr_str(meta, idx, i));
    }
    return out;
}

ggml_tensor * require_tensor(ggml_context * ctx, const std::string & name) {
    ggml_tensor * tensor = ggml_get_tensor(ctx, name.c_str());
    if (tensor == nullptr) {
        throw std::runtime_error("missing tensor: " + name);
    }
    return tensor;
}

ggml_tensor * optional_tensor(ggml_context * ctx, const std::string & name) {
    return ggml_get_tensor(ctx, name.c_str());
}

resnet50_model load_model(const std::string & model_path) {
    ggml_context * weights_ctx_raw = nullptr;
    gguf_init_params params = {
        /*.no_alloc =*/ false,
        /*.ctx      =*/ &weights_ctx_raw,
    };

    resnet50_model model;
    model.meta.reset(gguf_init_from_file(model_path.c_str(), params));
    if (!model.meta) {
        throw std::runtime_error("failed to load GGUF file: " + model_path);
    }
    model.weights.reset(weights_ctx_raw);
    if (!model.weights) {
        throw std::runtime_error("failed to materialize GGUF tensor context");
    }

    const std::string arch = get_string_key(model.meta.get(), "general.architecture");
    if (arch != "resnet50") {
        throw std::runtime_error("unsupported architecture: " + arch);
    }

    model.preproc.image_size = get_u32_key(model.meta.get(), "resnet50.image_size", model.preproc.image_size);
    model.preproc.resize_shorter_edge = get_u32_key(model.meta.get(), "resnet50.resize_shorter_edge", model.preproc.resize_shorter_edge);

    const auto mean = get_float_array(model.meta.get(), "resnet50.input_mean");
    const auto stdv = get_float_array(model.meta.get(), "resnet50.input_std");
    if (mean.size() != 3 || stdv.size() != 3) {
        throw std::runtime_error("expected 3-channel normalization metadata");
    }
    for (int i = 0; i < 3; ++i) {
        model.preproc.mean[i] = mean[i];
        model.preproc.std[i] = stdv[i];
    }

    const auto stages = get_u32_array(model.meta.get(), "resnet50.stage_block_count");
    if (stages.size() != 4) {
        throw std::runtime_error("expected 4 stage block counts");
    }
    for (int i = 0; i < 4; ++i) {
        model.stage_block_count[i] = stages[i];
    }

    model.labels = get_string_array(model.meta.get(), "resnet50.labels");

    return model;
}

image_u8 load_image_rgb(const std::string & image_path) {
    int w = 0;
    int h = 0;
    int c = 0;
    unsigned char * raw = stbi_load(image_path.c_str(), &w, &h, &c, 3);
    if (raw == nullptr) {
        throw std::runtime_error("failed to load image: " + image_path);
    }

    image_u8 image;
    image.width = w;
    image.height = h;
    image.channels = 3;
    image.data.assign(raw, raw + (w * h * 3));
    stbi_image_free(raw);
    return image;
}

std::vector<uint8_t> resize_bilinear_rgb(const image_u8 & image, int dst_w, int dst_h) {
    std::vector<uint8_t> out(dst_w * dst_h * 3);
    const float scale_x = static_cast<float>(image.width) / static_cast<float>(dst_w);
    const float scale_y = static_cast<float>(image.height) / static_cast<float>(dst_h);

    for (int y = 0; y < dst_h; ++y) {
        const float src_y = (static_cast<float>(y) + 0.5f) * scale_y - 0.5f;
        const int y0 = std::clamp(static_cast<int>(std::floor(src_y)), 0, image.height - 1);
        const int y1 = std::min(y0 + 1, image.height - 1);
        const float ly = src_y - static_cast<float>(y0);

        for (int x = 0; x < dst_w; ++x) {
            const float src_x = (static_cast<float>(x) + 0.5f) * scale_x - 0.5f;
            const int x0 = std::clamp(static_cast<int>(std::floor(src_x)), 0, image.width - 1);
            const int x1 = std::min(x0 + 1, image.width - 1);
            const float lx = src_x - static_cast<float>(x0);

            for (int ch = 0; ch < 3; ++ch) {
                const float p00 = image.data[(y0 * image.width + x0) * 3 + ch];
                const float p01 = image.data[(y0 * image.width + x1) * 3 + ch];
                const float p10 = image.data[(y1 * image.width + x0) * 3 + ch];
                const float p11 = image.data[(y1 * image.width + x1) * 3 + ch];

                const float top = p00 + (p01 - p00) * lx;
                const float bottom = p10 + (p11 - p10) * lx;
                const float value = top + (bottom - top) * ly;

                out[(y * dst_w + x) * 3 + ch] = static_cast<uint8_t>(std::clamp(std::lround(value), 0l, 255l));
            }
        }
    }

    return out;
}

std::vector<float> preprocess_image(const image_u8 & image, const preproc_config & cfg) {
    const int resize_short = static_cast<int>(cfg.resize_shorter_edge);
    const int crop = static_cast<int>(cfg.image_size);

    int resized_w = 0;
    int resized_h = 0;
    if (image.width < image.height) {
        resized_w = resize_short;
        resized_h = static_cast<int>(std::round(static_cast<float>(image.height) * resize_short / image.width));
    } else {
        resized_h = resize_short;
        resized_w = static_cast<int>(std::round(static_cast<float>(image.width) * resize_short / image.height));
    }

    const std::vector<uint8_t> resized = resize_bilinear_rgb(image, resized_w, resized_h);
    const int x0 = std::max(0, (resized_w - crop) / 2);
    const int y0 = std::max(0, (resized_h - crop) / 2);

    std::vector<float> out(crop * crop * 3);
    for (int y = 0; y < crop; ++y) {
        for (int x = 0; x < crop; ++x) {
            const int src_x = x0 + x;
            const int src_y = y0 + y;
            for (int ch = 0; ch < 3; ++ch) {
                const float pixel = resized[(src_y * resized_w + src_x) * 3 + ch] / 255.0f;
                const float normalized = (pixel - cfg.mean[ch]) / cfg.std[ch];
                out[x + crop * (y + crop * ch)] = normalized;
            }
        }
    }

    return out;
}

struct feature_map {
    int width = 0;
    int height = 0;
    int channels = 0;
    std::vector<float> data;

    float & at(int c, int y, int x) {
        return data[static_cast<size_t>(x + width * (y + height * c))];
    }

    const float & at(int c, int y, int x) const {
        return data[static_cast<size_t>(x + width * (y + height * c))];
    }
};

template <typename Fn>
void parallel_for_chunks(int total, int n_threads, Fn fn) {
    const int nth = std::max(1, std::min(total, n_threads));
    if (nth == 1) {
        fn(0, total);
        return;
    }

    std::vector<std::thread> workers;
    workers.reserve(nth);
    for (int tid = 0; tid < nth; ++tid) {
        const int begin = total * tid / nth;
        const int end = total * (tid + 1) / nth;
        workers.emplace_back([=, &fn]() {
            fn(begin, end);
        });
    }
    for (auto & worker : workers) {
        worker.join();
    }
}

float tensor_scalar(const ggml_tensor * tensor, int64_t idx) {
    if (tensor->type == GGML_TYPE_F32) {
        return static_cast<const float *>(tensor->data)[idx];
    }
    if (tensor->type == GGML_TYPE_F16) {
        return ggml_fp16_to_fp32(static_cast<const ggml_fp16_t *>(tensor->data)[idx]);
    }
    throw std::runtime_error("unsupported tensor type");
}

float bias_value(const ggml_tensor * bias, int channel) {
    return tensor_scalar(bias, channel);
}

feature_map conv2d(
        const ggml_tensor * weight,
        const ggml_tensor * bias,
        const feature_map & input,
        int stride,
        int padding,
        int n_threads) {
    const int kw = static_cast<int>(weight->ne[0]);
    const int kh = static_cast<int>(weight->ne[1]);
    const int in_channels = static_cast<int>(weight->ne[2]);
    const int out_channels = static_cast<int>(weight->ne[3]);
    if (in_channels != input.channels) {
        throw std::runtime_error("conv input channel mismatch");
    }

    feature_map out;
    out.width = (input.width + 2 * padding - kw) / stride + 1;
    out.height = (input.height + 2 * padding - kh) / stride + 1;
    out.channels = out_channels;
    out.data.assign(static_cast<size_t>(out.width * out.height * out.channels), 0.0f);

    const int in_plane = input.width * input.height;
    const int out_plane = out.width * out.height;

    parallel_for_chunks(out_channels, n_threads, [&](int oc_begin, int oc_end) {
        for (int oc = oc_begin; oc < oc_end; ++oc) {
            const float b = bias ? bias_value(bias, oc) : 0.0f;
            for (int oy = 0; oy < out.height; ++oy) {
                const int iy_base = oy * stride - padding;
                for (int ox = 0; ox < out.width; ++ox) {
                    const int ix_base = ox * stride - padding;
                    float sum = b;
                    for (int ic = 0; ic < in_channels; ++ic) {
                        const int64_t w_base = static_cast<int64_t>(kw) * kh * (ic + static_cast<int64_t>(in_channels) * oc);
                        const size_t in_base = static_cast<size_t>(ic * in_plane);
                        for (int ky = 0; ky < kh; ++ky) {
                            const int iy = iy_base + ky;
                            if (iy < 0 || iy >= input.height) {
                                continue;
                            }
                            const int64_t w_row = w_base + static_cast<int64_t>(kw) * ky;
                            const size_t in_row = in_base + static_cast<size_t>(iy * input.width);
                            for (int kx = 0; kx < kw; ++kx) {
                                const int ix = ix_base + kx;
                                if (ix < 0 || ix >= input.width) {
                                    continue;
                                }
                                sum += input.data[in_row + ix] * tensor_scalar(weight, w_row + kx);
                            }
                        }
                    }
                    out.data[static_cast<size_t>(ox + out.width * oy + out_plane * oc)] = sum;
                }
            }
        }
    });

    return out;
}

void relu_inplace(feature_map & fm) {
    for (float & v : fm.data) {
        v = std::max(0.0f, v);
    }
}

feature_map max_pool_2d(const feature_map & input, int kernel, int stride, int padding) {
    feature_map out;
    out.width = (input.width + 2 * padding - kernel) / stride + 1;
    out.height = (input.height + 2 * padding - kernel) / stride + 1;
    out.channels = input.channels;
    out.data.assign(static_cast<size_t>(out.width * out.height * out.channels), 0.0f);

    const int out_plane = out.width * out.height;
    for (int c = 0; c < out.channels; ++c) {
        for (int oy = 0; oy < out.height; ++oy) {
            const int iy_base = oy * stride - padding;
            for (int ox = 0; ox < out.width; ++ox) {
                const int ix_base = ox * stride - padding;
                float best = -std::numeric_limits<float>::infinity();
                for (int ky = 0; ky < kernel; ++ky) {
                    const int iy = iy_base + ky;
                    if (iy < 0 || iy >= input.height) {
                        continue;
                    }
                    for (int kx = 0; kx < kernel; ++kx) {
                        const int ix = ix_base + kx;
                        if (ix < 0 || ix >= input.width) {
                            continue;
                        }
                        best = std::max(best, input.at(c, iy, ix));
                    }
                }
                out.data[static_cast<size_t>(ox + out.width * oy + out_plane * c)] = best;
            }
        }
    }
    return out;
}

void add_inplace(feature_map & dst, const feature_map & src) {
    if (dst.width != src.width || dst.height != src.height || dst.channels != src.channels) {
        throw std::runtime_error("residual shape mismatch");
    }
    for (size_t i = 0; i < dst.data.size(); ++i) {
        dst.data[i] += src.data[i];
    }
}

feature_map bottleneck_block(
        const resnet50_model & model,
        const feature_map & input,
        int stage_idx,
        int block_idx,
        int n_threads) {
    const std::string base = "resnet.stage." + std::to_string(stage_idx) + ".block." + std::to_string(block_idx);

    feature_map identity = input;
    if (const ggml_tensor * downsample = optional_tensor(model.weights.get(), base + ".downsample.weight")) {
        identity = conv2d(
                downsample,
                require_tensor(model.weights.get(), base + ".downsample.bias"),
                identity,
                block_idx == 0 && stage_idx > 0 ? 2 : 1,
                0,
                n_threads);
    }

    feature_map out = conv2d(
            require_tensor(model.weights.get(), base + ".conv1.weight"),
            require_tensor(model.weights.get(), base + ".conv1.bias"),
            input,
            1,
            0,
            n_threads);
    relu_inplace(out);

    out = conv2d(
            require_tensor(model.weights.get(), base + ".conv2.weight"),
            require_tensor(model.weights.get(), base + ".conv2.bias"),
            out,
            block_idx == 0 && stage_idx > 0 ? 2 : 1,
            1,
            n_threads);
    relu_inplace(out);

    out = conv2d(
            require_tensor(model.weights.get(), base + ".conv3.weight"),
            require_tensor(model.weights.get(), base + ".conv3.bias"),
            out,
            1,
            0,
            n_threads);

    add_inplace(out, identity);
    relu_inplace(out);
    return out;
}

std::vector<float> run_inference(const resnet50_model & model, const std::vector<float> & input, int n_threads) {
    const int image_size = static_cast<int>(model.preproc.image_size);
    feature_map cur;
    cur.width = image_size;
    cur.height = image_size;
    cur.channels = 3;
    cur.data = input;

    cur = conv2d(
            require_tensor(model.weights.get(), "resnet.stem.conv.weight"),
            require_tensor(model.weights.get(), "resnet.stem.conv.bias"),
            cur,
            2,
            3,
            n_threads);
    relu_inplace(cur);
    cur = max_pool_2d(cur, 3, 2, 1);

    for (int stage_idx = 0; stage_idx < 4; ++stage_idx) {
        for (uint32_t block_idx = 0; block_idx < model.stage_block_count[stage_idx]; ++block_idx) {
            cur = bottleneck_block(model, cur, stage_idx, static_cast<int>(block_idx), n_threads);
        }
    }

    std::vector<float> pooled(static_cast<size_t>(cur.channels), 0.0f);
    const int spatial = cur.width * cur.height;
    for (int c = 0; c < cur.channels; ++c) {
        double sum = 0.0;
        for (int y = 0; y < cur.height; ++y) {
            for (int x = 0; x < cur.width; ++x) {
                sum += cur.at(c, y, x);
            }
        }
        pooled[c] = static_cast<float>(sum / spatial);
    }

    const ggml_tensor * classifier_w = require_tensor(model.weights.get(), "resnet.classifier.weight");
    const ggml_tensor * classifier_b = require_tensor(model.weights.get(), "resnet.classifier.bias");
    const int in_features = static_cast<int>(classifier_w->ne[0]);
    const int out_features = static_cast<int>(classifier_w->ne[1]);
    if (in_features != cur.channels) {
        throw std::runtime_error("classifier input size mismatch");
    }

    std::vector<float> logits(static_cast<size_t>(out_features), 0.0f);
    parallel_for_chunks(out_features, n_threads, [&](int begin, int end) {
        for (int oc = begin; oc < end; ++oc) {
            float sum = tensor_scalar(classifier_b, oc);
            const int64_t w_base = static_cast<int64_t>(in_features) * oc;
            for (int ic = 0; ic < in_features; ++ic) {
                sum += pooled[ic] * tensor_scalar(classifier_w, w_base + ic);
            }
            logits[oc] = sum;
        }
    });

    return logits;
}

std::vector<std::pair<int, float>> top_k(const std::vector<float> & logits, int k) {
    std::vector<int> ids(logits.size());
    std::iota(ids.begin(), ids.end(), 0);

    std::partial_sort(
            ids.begin(),
            ids.begin() + std::min<int>(k, ids.size()),
            ids.end(),
            [&](int a, int b) {
                return logits[a] > logits[b];
            });

    std::vector<std::pair<int, float>> out;
    for (int i = 0; i < std::min<int>(k, ids.size()); ++i) {
        out.emplace_back(ids[i], logits[ids[i]]);
    }
    return out;
}

args parse_args(int argc, char ** argv) {
    args out;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            out.model_path = argv[++i];
        } else if (arg == "--image" && i + 1 < argc) {
            out.image_path = argv[++i];
        } else if (arg == "--threads" && i + 1 < argc) {
            out.threads = std::stoi(argv[++i]);
        } else if (arg == "--top-k" && i + 1 < argc) {
            out.top_k = std::stoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "usage: run_resnet50 --model <model.gguf> --image <image> [--threads N] [--top-k K]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown or incomplete argument: " + arg);
        }
    }

    if (out.model_path.empty() || out.image_path.empty()) {
        throw std::runtime_error("both --model and --image are required");
    }
    return out;
}

} // namespace

int main(int argc, char ** argv) {
    try {
        const args cli = parse_args(argc, argv);
        const resnet50_model model = load_model(cli.model_path);
        const image_u8 image = load_image_rgb(cli.image_path);
        const std::vector<float> input = preprocess_image(image, model.preproc);
        const std::vector<float> logits = run_inference(model, input, cli.threads);
        const auto best = top_k(logits, cli.top_k);

        for (size_t i = 0; i < best.size(); ++i) {
            const int idx = best[i].first;
            const float score = best[i].second;
            std::string label = idx < static_cast<int>(model.labels.size()) ? model.labels[idx] : "<unknown>";
            std::cout << i + 1 << "\tclass_id=" << idx << "\tlogit=" << score << "\tlabel=" << label << "\n";
        }
        return 0;
    } catch (const std::exception & ex) {
        std::cerr << "error: " << ex.what() << "\n";
        return 1;
    }
}
