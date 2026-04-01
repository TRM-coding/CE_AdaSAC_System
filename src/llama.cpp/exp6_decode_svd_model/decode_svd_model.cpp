#include "llama-context.h"

#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-io.h"
#include "llama-memory.h"
#include "llama-mmap.h"
#include "llama-model.h"

#include <cinttypes>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <thread>

int main(int argc, char ** argv)
{
    int ngl = 99;
    uint32_t n_ctx = 2048;
    std::string model_path = argc > 1
        ? argv[1]
        : "/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/gguf_models/qwen7b.gguf.sort_svd.gguf";
    const int32_t max_new_tokens = argc > 2 ? std::stoi(argv[2]) : 64;
    const int32_t n_threads = argc > 3
        ? std::stoi(argv[3])
        : std::max(1u, std::thread::hardware_concurrency() / 2);
    const bool verbose_tokens = argc > 4 ? std::stoi(argv[4]) != 0 : false;

    // 1. 加载动态后端
    ggml_backend_load_all();

    // 2. 加载模型
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = ngl;

    std::cout << "loading model ..." << std::endl;
    llama_model *model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model)
    {
        std::cerr << "error: unable to load model" << std::endl;
        return 1;
    }
    std::cout << "model_name:" << model->name << std::endl;

    const llama_vocab *vocab = llama_model_get_vocab(model);

    // 3. 创建 context
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_ctx;
    ctx_params.n_batch = n_ctx;
    ctx_params.n_threads = n_threads;
    ctx_params.n_threads_batch = n_threads;

    std::cout << "creating context ..." << std::endl;
    llama_context *ctx = llama_init_from_model(model, ctx_params);
    if (!ctx)
    {
        std::cerr << "error: failed to create the llama_context" << std::endl;
        llama_model_free(model);
        return 1;
    }
    std::cout << "threads: decode=" << llama_n_threads(ctx)
              << " batch=" << llama_n_threads_batch(ctx) << std::endl;

    // ========== 下面是你要的一次前向传播 ==========

    // 4. 准备一个简单的 prompt
    std::string prompt = "Once upon a time";

    // 5. tokenize
    //    注意：llama_tokenize 的签名以你本地头文件为准，这里假设是新的 API：
    //    int32_t llama_tokenize(const llama_vocab * vocab,
    //                           const char * text, int32_t text_len,
    //                           llama_token * tokens, int32_t n_max_tokens,
    //                           bool add_special, bool parse_special);
    std::vector<llama_token> tokens;
    tokens.resize(prompt.size() + 8);

    int32_t n_tokens = llama_tokenize(
        vocab,
        prompt.c_str(),
        (int32_t)prompt.size(),
        tokens.data(),
        (int32_t)tokens.size(),
        /*add_special*/ true,
        /*parse_special*/ true);

    if (n_tokens <= 0)
    {
        std::cerr << "tokenize failed, n_tokens = " << n_tokens << std::endl;
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    tokens.resize(n_tokens);

    std::cout << "prompt tokens: " << n_tokens << std::endl;

    // 6. 清空 KV cache
    llama_kv_cache_clear(ctx);

    // 7. 构造 batch 并做一次 forward
    //    注意：llama_batch_init 的参数以你本地头文件为准，这里用常见形式：
    //    llama_batch llama_batch_init(int32_t n_tokens, int32_t embd, int32_t n_seq_max);
    llama_batch batch = llama_batch_init(/*n_tokens*/ n_tokens,
                                         /*embd*/ 0,
                                         /*n_seq_max*/ 1);

    batch.n_tokens = n_tokens;

    for (int32_t i = 0; i < n_tokens; ++i)
    {
        batch.token[i] = tokens[i]; // 第 i 个输入 token
        batch.pos[i] = i;           // 在序列中的位置
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;                // 只有一条 seq，id = 0
        batch.logits[i] = (i == n_tokens - 1); // 只在最后一个 token 上要 logits
    }

    std::cout << "running one forward pass (llama_decode) ..." << std::endl;
    int32_t ret = llama_decode(ctx, batch);
    if (ret != 0)
    {
        std::cerr << "llama_decode failed, ret = " << ret << std::endl;
        llama_batch_free(batch);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    // 8. 取出最后一个 token 的 logits
    const float *logits = llama_get_logits_ith(ctx, n_tokens - 1);
    if (!logits)
    {
        std::cerr << "llama_get_logits_ith returned nullptr" << std::endl;
        llama_batch_free(batch);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    int32_t vocab_size = llama_vocab_n_tokens(vocab);

    auto greedy_sample = [&](const float * logits) -> llama_token {
        int32_t best_id = 0;
        float best_logit = logits[0];
        for (int32_t i = 1; i < vocab_size; ++i) {
            if (logits[i] > best_logit) {
                best_logit = logits[i];
                best_id = i;
            }
        }
        return (llama_token) best_id;
    };

    // 9. 简单取 top-5，看一下 forward 是否正常
    struct Candidate
    {
        int32_t id;
        float logit;
    };

    std::vector<Candidate> cands;
    cands.reserve(vocab_size);
    for (int32_t i = 0; i < vocab_size; ++i)
    {
        cands.push_back({i, logits[i]});
    }

    std::partial_sort(
        cands.begin(), cands.begin() + 5, cands.end(),
        [](const Candidate &a, const Candidate &b)
        {
            return a.logit > b.logit;
        });

    std::cout << "Top-5 next-token candidates:" << std::endl;
    for (int i = 0; i < 5; ++i)
    {
        char buf[256];
        int32_t len = llama_token_to_piece(
            vocab,
            cands[i].id,
            buf,
            (int32_t)sizeof(buf),
            /*lstrip*/ 0,
            /*special*/ true);
        std::string piece = (len > 0 ? std::string(buf, buf + len) : std::string("[ERR]"));

        std::cout << "  id=" << cands[i].id
                  << " logit=" << cands[i].logit
                  << " piece=\"" << piece << "\""
                  << std::endl;
    }

    std::cout << "Autoregressive generation:" << std::endl;
    std::vector<llama_token> generated_token_ids;
    generated_token_ids.reserve(max_new_tokens);
    int32_t total_tokens = n_tokens;
    logits = llama_get_logits_ith(ctx, total_tokens - 1);
    bool generation_failed = false;
    int32_t generated_tokens = 0;
    const auto generation_start = std::chrono::steady_clock::now();
    double decode_only_sec = 0.0;

    if (!logits) {
        std::cerr << "llama_get_logits_ith returned nullptr before generation" << std::endl;
        generation_failed = true;
    } else {
        for (int32_t gen = 0; gen < max_new_tokens; ++gen) {
            llama_token next_token = greedy_sample(logits);

            if (verbose_tokens) {
                char buf[256];
                int32_t len = llama_token_to_piece(
                    vocab,
                    next_token,
                    buf,
                    (int32_t) sizeof(buf),
                    /*lstrip*/ 0,
                    /*special*/ true
                );
                std::string piece = (len > 0 ? std::string(buf, buf + len) : std::string("[ERR]"));
                std::cout << "  token #" << gen
                          << " id=" << next_token
                          << " piece=\"" << piece << "\""
                          << std::endl;
            }
            generated_token_ids.push_back(next_token);

            llama_batch gen_batch = llama_batch_init(1, 0, 1);
            gen_batch.n_tokens = 1;
            gen_batch.token[0] = next_token;
            gen_batch.pos[0] = total_tokens;
            gen_batch.n_seq_id[0] = 1;
            gen_batch.seq_id[0][0] = 0;
            gen_batch.logits[0] = 1;

            const auto decode_start = std::chrono::steady_clock::now();
            ret = llama_decode(ctx, gen_batch);
            const auto decode_end = std::chrono::steady_clock::now();
            llama_batch_free(gen_batch);
            decode_only_sec += std::chrono::duration<double>(decode_end - decode_start).count();
            if (ret != 0) {
                std::cerr << "llama_decode failed during generation, ret = " << ret << std::endl;
                generation_failed = true;
                break;
            }

            total_tokens += 1;
            generated_tokens += 1;
            logits = llama_get_logits_ith(ctx, 0); // fetch logits for the only token in the last batch
            if (!logits) {
                std::cerr << "llama_get_logits_ith returned nullptr during generation" << std::endl;
                generation_failed = true;
                break;
            }
        }
    }

    if (!generated_token_ids.empty()) {
        std::string generated_text;
        for (llama_token token : generated_token_ids) {
            char buf[256];
            int32_t len = llama_token_to_piece(
                vocab,
                token,
                buf,
                (int32_t) sizeof(buf),
                /*lstrip*/ 0,
                /*special*/ true
            );
            generated_text += (len > 0 ? std::string(buf, buf + len) : std::string("[ERR]"));
        }
        std::cout << "Generated text: " << generated_text << std::endl;
    }

    const auto generation_end = std::chrono::steady_clock::now();
    const double generation_sec = std::chrono::duration<double>(generation_end - generation_start).count();
    if (generated_tokens > 0 && decode_only_sec > 0.0) {
        std::cout << "Decode-only throughput: " << (generated_tokens / decode_only_sec)
                  << " tokens/s (" << generated_tokens << " tokens in "
                  << decode_only_sec << " s of llama_decode)" << std::endl;
    }
    if (generated_tokens > 0 && generation_sec > 0.0) {
        std::cout << "End-to-end throughput: " << (generated_tokens / generation_sec)
                  << " tokens/s (" << generated_tokens << " tokens in "
                  << generation_sec << " s)" << std::endl;
    }

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);

    if (!generation_failed) {
        std::cout << "SUCCEED (one forward pass done)" << std::endl;
    } else {
        std::cout << "FAILED (generation aborted)" << std::endl;
    }

    return generation_failed ? 1 : 0;
}
