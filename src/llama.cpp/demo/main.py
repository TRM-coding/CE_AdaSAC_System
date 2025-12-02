import sys

sys.path.append('../build/ops')
import opslib as ops

def main():
    times = 10
    type_f32 = ops.ggml_type.GGML_TYPE_F32
    type_f16 = ops.ggml_type.GGML_TYPE_F16
    ne = [10, 5, 4, 3]

    # 调用 run_add 并接收性能信息
    info = ops.run_add(times, type_f32, ne)
    print("------------------------")
    print(f"ADD operation finished! Time per op: {info.time_per_op_ms:.6f} ms")
    print("------------------------")

    # 调用 run_cpy
    info = ops.run_cpy(times, type_f32, type_f32, ne)
    print("------------------------")
    print(f"CPY operation finished! Time per op: {info.time_per_op_ms:.6f} ms")
    print("------------------------")

    info = ops.run_gelu(times, type_f32, ne)
    print("------------------------")
    print(f"GELU operation finished! Time per op: {info.time_per_op_ms:.6f} ms")
    print("------------------------")

    info = ops.run_mul(times, type_f32, ne)
    print("------------------------")
    print(f"MUL operation finished! Time per op: {info.time_per_op_ms:.6f} ms")
    print("------------------------")

    info = ops.run_mul_mat(times, type_f32, ne)
    print("------------------------")
    print(f"MUL_MAT operation finished! Time per op: {info.time_per_op_ms:.6f} ms")
    print("------------------------")   

    info = ops.run_norm(times, type_f32, ne)
    print("------------------------")
    print(f"NORM operation finished! Time per op: {info.time_per_op_ms:.6f} ms")
    print("------------------------")



    info = ops.run_permute(
        times,
        type_f32,
        ne,
        [2, 3, 0, 1]
    )
    print("------------------------")
    print(f"PERMUTE operation finished! Time per op: {info.time_per_op_ms:.6f} ms")
    print("------------------------")

    info = ops.run_reshape(
        times,
        type_f32,
        ne,
        [1,ne[0]*ne[1], ne[2], ne[3]]
    )
    print("------------------------")
    print(f"RESHAPE operation finished! Time per op: {info.time_per_op_ms:.6f} ms")
    print("------------------------")

    info = ops.run_norm(times, type_f32, ne)
    print("------------------------")
    print(f"NORM operation finished! Time per op: {info.time_per_op_ms:.6f} ms")
    print("------------------------")   

    info = ops.run_rms_norm(times, type_f32, ne)
    print("------------------------")
    print(f"RMS_NORM operation finished! Time per op: {info.time_per_op_ms:.6f} ms")
    print("------------------------")

    info = ops.run_view_1d(
        times,
        type_f32,
        ne,
        [10*5*4*3]
    )
    print("------------------------")
    print(f"VIEW_1D operation finished! Time per op: {info.time_per_op_ms:.6f} ms")
    print("------------------------")   

    info = ops.run_view_2d(
        times,
        type_f32,
        ne,
        [10*5, 4*3]
    )
    print("------------------------")
    print(f"VIEW_2D operation finished! Time per op: {info.time_per_op_ms:.6f} ms")
    print("------------------------")       
    info = ops.run_view_3d(
        times,
        type_f32,
        ne,
        [10, 5, 4*3]
    )
    print("------------------------")
    print(f"VIEW_3D operation finished! Time per op: {info.time_per_op_ms:.6f} ms")
    print("------------------------")   

    info = ops.run_view_4d(
        times,
        type_f32,
        ne,
        [10, 5, 4, 3]
    )
    print("------------------------")
    print(f"VIEW_4D operation finished! Time per op: {info.time_per_op_ms:.6f} ms")
    print("------------------------")
    

    # Flash Attention 参数
    hsk = 64
    hsv = 64
    nh = 16
    nr23 = [64, 64]
    kv = 128
    nb = 1

    mask = True
    sinks = False
    max_bias = 32.0
    logit_softcap = 30.0
    prec = ops.ggml_prec.GGML_PREC_F32
    type_kv = ops.ggml_type.GGML_TYPE_F16

    # 调用 run_flash_attn_ext
    info = ops.run_flash_attn_ext(
        times,
        hsk,
        hsv,
        nh,
        nr23,
        kv,
        nb,
        mask,
        sinks,
        max_bias,
        logit_softcap,
        prec,
        type_kv
    )
    print("------------------------")
    print(f"FLASH_ATTN_EXT operation finished! Time per op: {info.time_per_op_ms:.6f} ms")
    print("------------------------")

    # scale 参数
    ne_scale = [10, 10, 10, 10]
    scale = 2.0
    bias = 0.0

    # 调用 run_scale
    info = ops.run_scale(times, type_f32, ne_scale, scale, bias)
    print("------------------------")
    print(f"SCALE operation finished! Time per op: {info.time_per_op_ms:.6f} ms")
    print("------------------------")


    # swiglu 参数
    # ne_siwglu的第一个参数必须为偶数
    ne_siwglu = [2 * 64, 2, 4, 1]

    # 调用 swiglu_oai
    info = ops.run_swiglu(times, type_f32, ne_siwglu)
    print("------------------------")
    print(f"SWIGLU operation finished! Time per op: {info.time_per_op_ms:.6f} ms")

    # swiglu_oai 参数
    # ne_siwglu_oai的第一个参数必须为偶数
    ne_siwglu_oai = [2 * 64, 2, 2, 2]
    alpha = 1.702
    limit = 7.0

    # 调用 swiglu_oai
    info = ops.run_swiglu_oai(times, type_f32, ne_siwglu_oai, alpha, limit)
    print("------------------------")
    print(f"SWIGLU_OAI operation finished! Time per op: {info.time_per_op_ms:.6f} ms")
    print("------------------------")

if __name__ == "__main__":
    main()