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

int main()
{
    int ngl = 99;
    uint32_t n_ctx = 2048;
    std::string model_path="/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/gguf_models/qwen.gguf.svd.gguf";
    // std::string model_path="/home/tianruiming/CE_ADA_LLAMA/src/llama.cpp/gguf_models/qwen.gguf";

    // load dynamic backends
    ggml_backend_load_all();

    // initialize the model
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = ngl;

        std::cout<<"---------------------------------------------"<<std::endl;


    llama_model *model = llama_model_load_from_file(model_path.c_str(), model_params);

        std::cout<<"---------------------------------------------"<<std::endl;

    if (!model)
    {
        fprintf(stderr, "%s: error: unable to load model\n", __func__);
        return 1;
    }

    const llama_vocab *vocab = llama_model_get_vocab(model);


    std::cout<<"---------------------------------------------"<<std::endl;

    // initialize the context
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_ctx;
    ctx_params.n_batch = n_ctx;
    std::cout<<"---------------------------------------------"<<std::endl;
    std::cout<<"HHHHH"<<std::endl;
    llama_context *ctx = llama_init_from_model(model, ctx_params);
    if (!ctx)
    {
        fprintf(stderr, "%s: error: failed to create the llama_context\n", __func__);
        return 1;
    }
   
    std::cout<<"---------------------------------------------"<<std::endl;

    llama_free(ctx);
    llama_model_free(model);

    std::cout<<"SUCEED"<<std::endl;

    return 0;
}


