#include "ops.h"
#include<ggml.h>
#include<array>
#include<iostream>
int main()
{
    int times=10;
    ggml_type type = GGML_TYPE_F32;
    std::array<int64_t, 4UL> ne = {10, 5, 4, 3};

    struct OPS_INFO info_add{};
    RUN_ADD(times, type, ne,info_add);
    std::cout<<"------------------------"<<std::endl;
    std::cout<<"ADD operation finished! AVG Excute Time:"<<info_add.time_per_op_ms<<"ms" <<std::endl;
    std::cout<<"------------------------"<<std::endl;
    struct OPS_INFO info_cpy{};
    RUN_CPY(times,type,type,ne,info_cpy);
    std::cout<<"------------------------"<<std::endl;
    std::cout<<"CPY operation finished! AVG Excute Time:"<<info_cpy.time_per_op_ms<<"ms" <<std::endl;
    std::cout<<"------------------------"<<std::endl;
    // 参数含义：
    // times: 执行次数
    // hsk, hsv: head_size_k, head_size_v
    // nh: head 数
    // nr23: 序列长度 (n_rows, n_cols)
    // kv: KV缓存长度
    // nb: batch 大小
    // mask: 是否启用mask（用于自注意力）
    // sinks: 是否启用sinks attention（如 Llama Sink Tokens）
    // max_bias: logits偏置上限
    // logit_softcap: logits截断阈值
    // prec: 精度（GGML_PREC_F32/F16）
    // type_KV: KV tensor 数据类型（GGML_TYPE_F16/F32）

    int64_t hsk = 64;        // 每个 head 的 key 维度
    int64_t hsv = 64;        // 每个 head 的 value 维度
    int64_t nh  = 16;         // head 数量
    std::array<int64_t, 2> nr23 = {64, 64}; // 输入序列长度、输出序列长度
    int64_t kv = 128;         // KV 缓存长度（上下文窗口）
    int64_t nb = 1;           // batch size

    bool mask = true;         // 启用 causal mask
    bool sinks = false;       // 不启用 sink token
    float max_bias = 32.0f;   // 限制 bias 幅度
    float logit_softcap = 30.0f; // 限制 logits 范围
    ggml_prec prec = GGML_PREC_F32;
    ggml_type type_KV = GGML_TYPE_F16;

    struct OPS_INFO info_flash_attn_ext{};
    RUN_FLASH_ATTN_EXT(
        times,
        hsk,
        hsv,
        nh,
        nr23,
        kv,
        nb,
        info_flash_attn_ext,
        mask,
        sinks,
        max_bias,
        logit_softcap,
        prec,
        type_KV
    );
    std::cout<<"------------------------"<<std::endl;
    std::cout<<"FLASH_ATTN_EXT operation finished! AVG Excute Time:"<<info_flash_attn_ext.time_per_op_ms<<"ms" <<std::endl;
    std::cout<<"------------------------"<<std::endl;
    struct OPS_INFO info_gelu{};
    RUN_GELU(times,type,ne,info_gelu);
    std::cout<<"------------------------"<<std::endl;
    std::cout<<"GELU operation finished! AVG Excute Time:"<<info_gelu.time_per_op_ms<<"ms" <<std::endl;
    std::cout<<"------------------------"<<std::endl;
    struct OPS_INFO info_mul{};
    RUN_MUL(times,type,ne,info_mul);
    std::cout<<"------------------------"<<std::endl;
    std::cout<<"MUL operation finished! AVG Excute Time:"<<info_mul.time_per_op_ms<<"ms" <<std::endl;
    std::cout<<"------------------------"<<std::endl;
    struct OPS_INFO info_mul_mat{};
    RUN_MUL_MAT(times,type,ne,info_mul_mat);
    std::cout<<"------------------------"<<std::endl;
    std::cout<<"MUL_MAT operation finished! AVG Excute Time:"<<info_mul_mat.time_per_op_ms<<"ms" <<std::endl;
    std::cout<<"------------------------"<<std::endl;
    struct OPS_INFO info_norm{};
    RUN_NORM(times,type,ne,info_norm);
    std::cout<<"------------------------"<<std::endl;
    std::cout<<"NORM operation finished! AVG Excute Time:"<<info_norm.time_per_op_ms<<"ms" <<std::endl;
    std::cout<<"------------------------"<<std::endl;
    struct OPS_INFO info_permute{};
    RUN_PERMUTE(times,type,ne,std::array<int,4>{2,0,1,3},info_permute);
    std::cout<<"------------------------"<<std::endl;
    std::cout<<"PERMUTE operation finished! AVG Excute Time:"<<info_permute.time_per_op_ms<<"ms" <<std::endl;
    std::cout<<"------------------------"<<std::endl;
    struct OPS_INFO info_reshape{};
    RUN_RESHAPE(
        times,
        type,
        ne,
        std::array<int64_t,4UL>{4,3,5,10},
        info_reshape
    );
    std::cout<<"------------------------"<<std::endl;
    std::cout<<"RESHAPE operation finished! AVG Excute Time:"<<info_reshape.time_per_op_ms<<"ms" <<std::endl;
    std::cout<<"------------------------"<<std::endl;
    struct OPS_INFO info_rms_norm{};

    RUN_RMS_NORM(
        times,
        type,
        ne,
        info_rms_norm
    );
    std::cout<<"------------------------"<<std::endl;
    std::cout<<"RMS_NORM operation finished! AVG Excute Time:"<<info_rms_norm.time_per_op_ms<<"ms" <<std::endl;
    std::cout<<"------------------------"<<std::endl;

    struct OPS_INFO info_view{};
    RUN_VIEW_1D(
        times,
        type,
        ne,
        std::array<int,1>{10*5*4*3},
        info_view
    );
    std::cout<<"------------------------"<<std::endl;
    std::cout<<"VIEW_1D operation finished! AVG Excute Time:"<<info_view.time_per_op_ms<<"ms" <<std::endl;
    std::cout<<"------------------------"<<std::endl;

    struct OPS_INFO info_view2d{};
    RUN_VIEW_2D(
        times,
        type,
        ne,
        std::array<int,2>{10*5,4*3},
        info_view2d
    );
    std::cout<<"------------------------"<<std::endl;
    std::cout<<"VIEW_2D operation finished! AVG Excute Time:"<<info_view2d.time_per_op_ms<<"ms" <<std::endl;
    std::cout<<"------------------------"<<std::endl;

    struct OPS_INFO info_view3d{};
    RUN_VIEW_3D(
        times,
        type,
        ne,
        std::array<int,3>{10,5,4*3},
        info_view3d
    );
    std::cout<<"------------------------"<<std::endl;   
    std::cout<<"VIEW_3D operation finished! AVG Excute Time:"<<info_view3d.time_per_op_ms<<"ms" <<std::endl;
    std::cout<<"------------------------"<<std::endl;

    struct OPS_INFO info_view4d{};
    RUN_VIEW_4D(
        times,
        type,
        ne,
        std::array<int,4>{10,5,4,3},
        info_view4d
    );
    std::cout<<"------------------------"<<std::endl;   
    std::cout<<"VIEW_4D operation finished! AVG Excute Time:"<<info_view4d.time_per_op_ms<<"ms" <<std::endl;
    std::cout<<"------------------------"<<std::endl;
    // scale 参数
    std::array<int64_t, 4UL> ne_scale = {10, 10, 10, 10};
    float scale = 2.0f;
    float bias = 0.0f;
    
    // 调用 scale
    struct OPS_INFO info_scale{};
    RUN_SCALE(times,type,ne_scale,scale,bias,info_scale);
    std::cout<<"------------------------"<<std::endl;
    std::cout<<"SCALE operation finished! AVG Excute Time:"<<info_scale.time_per_op_ms<<"ms" <<std::endl;
    std::cout<<"------------------------"<<std::endl;

    // swiglu 参数
    // ne_siwglu的第一个参数必须为偶数
    std::array<int64_t, 4UL> ne_siwglu = {2 * 64, 2, 4, 1};
    
    // 调用 swiglu
    struct OPS_INFO info_swiglu{};
    RUN_SWIGLU(times,type,ne_siwglu,info_swiglu);
    std::cout<<"------------------------"<<std::endl;
    std::cout<<"SWIGLU operation finished! AVG Excute Time:"<<info_swiglu.time_per_op_ms<<"ms" <<std::endl;
    std::cout<<"------------------------"<<std::endl;

    // swiglu_oai 参数
    // ne_siwglu_oai的第一个参数必须为偶数
    std::array<int64_t, 4UL> ne_siwglu_oai = {2 * 64, 2, 2, 2};
    float alpha = 1.702f;
    float limit = 7.0f;
    
    // 调用 swiglu_oai
    struct OPS_INFO info_swiglu_oai{};
    RUN_SWIGLU_OAI(times,type,ne_siwglu_oai,alpha,limit,info_swiglu_oai);
    std::cout<<"------------------------"<<std::endl;
    std::cout<<"SWIGLU_OAI operation finished! AVG Excute Time:"<<info_swiglu_oai.time_per_op_ms<<"ms" <<std::endl;
    std::cout<<"------------------------"<<std::endl;

    return 0;

}