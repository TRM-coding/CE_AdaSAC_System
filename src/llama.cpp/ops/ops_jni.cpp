#include <jni.h>
#include "ops.h"
#include <cstring>

// Helper function to convert jlongArray to std::array
template<size_t N>
std::array<int64_t, N> jlongArrayToArray(JNIEnv* env, jlongArray jarray) {
    std::array<int64_t, N> result;
    jlong* elements = env->GetLongArrayElements(jarray, nullptr);
    for (size_t i = 0; i < N; i++) {
        result[i] = elements[i];
    }
    env->ReleaseLongArrayElements(jarray, elements, JNI_ABORT);
    return result;
}

template<size_t N>
std::array<int, N> jintArrayToArray(JNIEnv* env, jintArray jarray) {
    std::array<int, N> result;
    jint* elements = env->GetIntArrayElements(jarray, nullptr);
    for (size_t i = 0; i < N; i++) {
        result[i] = elements[i];
    }
    env->ReleaseIntArrayElements(jarray, elements, JNI_ABORT);
    return result;
}

// RUN_ADD wrapper
static jdoubleArray add(JNIEnv* env, jclass, jint times, jint type, jlongArray ne) {
    auto ne_array = jlongArrayToArray<4>(env, ne);
    OPS_INFO info;
    RUN_ADD(times, static_cast<ggml_type>(type), ne_array, info);
    
    jdoubleArray result = env->NewDoubleArray(1);
    jdouble timing = info.time_per_op_ms;
    env->SetDoubleArrayRegion(result, 0, 1, &timing);
    return result;
}

// RUN_CPY wrapper
static jdoubleArray cpy(JNIEnv* env, jclass, jint times, jint typeSrc, jint typeDst, jlongArray ne) {
    auto ne_array = jlongArrayToArray<4>(env, ne);
    OPS_INFO info;
    RUN_CPY(times, static_cast<ggml_type>(typeSrc), static_cast<ggml_type>(typeDst), ne_array, info);
    
    jdoubleArray result = env->NewDoubleArray(1);
    jdouble timing = info.time_per_op_ms;
    env->SetDoubleArrayRegion(result, 0, 1, &timing);
    return result;
}

// RUN_FLASH_ATTN_EXT wrapper
static jdoubleArray flashAttnExt(JNIEnv* env, jclass, jint times, jlong hsk, jlong hsv, jlong nh, 
    jlongArray nr23, jlong kv, jlong nb, jboolean mask, jboolean sinks,
    jfloat maxBias, jfloat logitSoftcap, jint prec, jint typeKV) {
    
    auto nr23_array = jlongArrayToArray<2>(env, nr23);
    OPS_INFO info;
    RUN_FLASH_ATTN_EXT(times, hsk, hsv, nh, nr23_array, kv, nb, info,
                       mask, sinks, maxBias, logitSoftcap,
                       static_cast<ggml_prec>(prec), static_cast<ggml_type>(typeKV));
    
    jdoubleArray result = env->NewDoubleArray(1);
    jdouble timing = info.time_per_op_ms;
    env->SetDoubleArrayRegion(result, 0, 1, &timing);
    return result;
}

// RUN_GELU wrapper
static jdoubleArray gelu(JNIEnv* env, jclass, jint times, jint typeSrc, jlongArray ne) {
    auto ne_array = jlongArrayToArray<4>(env, ne);
    OPS_INFO info;
    RUN_GELU(times, static_cast<ggml_type>(typeSrc), ne_array, info);
    
    jdoubleArray result = env->NewDoubleArray(1);
    jdouble timing = info.time_per_op_ms;
    env->SetDoubleArrayRegion(result, 0, 1, &timing);
    return result;
}

// RUN_MUL wrapper
static jdoubleArray mul(JNIEnv* env, jclass, jint times, jint type, jlongArray ne) {
    auto ne_array = jlongArrayToArray<4>(env, ne);
    OPS_INFO info;
    RUN_MUL(times, static_cast<ggml_type>(type), ne_array, info);
    
    jdoubleArray result = env->NewDoubleArray(1);
    jdouble timing = info.time_per_op_ms;
    env->SetDoubleArrayRegion(result, 0, 1, &timing);
    return result;
}

// RUN_SCALE wrapper
static jdoubleArray scale(JNIEnv* env, jclass, jint times, jint type, jlongArray ne, jfloat scale, jfloat bias) {
    auto ne_array = jlongArrayToArray<4>(env, ne);
    OPS_INFO info;
    RUN_SCALE(times, static_cast<ggml_type>(type), ne_array, scale, bias, info);
    
    jdoubleArray result = env->NewDoubleArray(1);
    jdouble timing = info.time_per_op_ms;
    env->SetDoubleArrayRegion(result, 0, 1, &timing);
    return result;
}

// RUN_SWIGLU wrapper
static jdoubleArray swiglu(JNIEnv* env, jclass, jint times, jint type, jlongArray ne) {
    auto ne_array = jlongArrayToArray<4>(env, ne);
    OPS_INFO info;
    RUN_SWIGLU(times, static_cast<ggml_type>(type), ne_array, info);
    
    jdoubleArray result = env->NewDoubleArray(1);
    jdouble timing = info.time_per_op_ms;
    env->SetDoubleArrayRegion(result, 0, 1, &timing);
    return result;
}

// RUN_SWIGLU_OAI wrapper
static jdoubleArray swigluOai(JNIEnv* env, jclass, jint times, jint type, jlongArray neA, jfloat alpha, jfloat limit) {
    auto ne_array = jlongArrayToArray<4>(env, neA);
    OPS_INFO info;
    RUN_SWIGLU_OAI(times, static_cast<ggml_type>(type), ne_array, alpha, limit, info);
    
    jdoubleArray result = env->NewDoubleArray(1);
    jdouble timing = info.time_per_op_ms;
    env->SetDoubleArrayRegion(result, 0, 1, &timing);
    return result;
}

// RUN_MUL_MAT wrapper
static jdoubleArray mulMat(JNIEnv* env, jclass, jint times, jint type, jlongArray ne) {
    auto ne_array = jlongArrayToArray<4>(env, ne);
    OPS_INFO info;
    RUN_MUL_MAT(times, static_cast<ggml_type>(type), ne_array, info);
    
    jdoubleArray result = env->NewDoubleArray(1);
    jdouble timing = info.time_per_op_ms;
    env->SetDoubleArrayRegion(result, 0, 1, &timing);
    return result;
}

// RUN_NORM wrapper
static jdoubleArray norm(JNIEnv* env, jclass, jint times, jint typeSrc, jlongArray ne) {
    auto ne_array = jlongArrayToArray<4>(env, ne);
    OPS_INFO info;
    RUN_NORM(times, static_cast<ggml_type>(typeSrc), ne_array, info);
    
    jdoubleArray result = env->NewDoubleArray(1);
    jdouble timing = info.time_per_op_ms;
    env->SetDoubleArrayRegion(result, 0, 1, &timing);
    return result;
}

// RUN_PERMUTE wrapper
static jdoubleArray permute(JNIEnv* env, jclass, jint times, jint typeSrc, jlongArray ne, jintArray permuteAxes) {
    auto ne_array = jlongArrayToArray<4>(env, ne);
    auto permute_array = jintArrayToArray<4>(env, permuteAxes);
    OPS_INFO info;
    RUN_PERMUTE(times, static_cast<ggml_type>(typeSrc), ne_array, permute_array, info);
    
    jdoubleArray result = env->NewDoubleArray(1);
    jdouble timing = info.time_per_op_ms;
    env->SetDoubleArrayRegion(result, 0, 1, &timing);
    return result;
}

// RUN_RESHAPE wrapper
static jdoubleArray reshape(JNIEnv* env, jclass, jint times, jint typeSrc, jlongArray ne, jlongArray shapeSize) {
    auto ne_array = jlongArrayToArray<4>(env, ne);
    auto shape_array = jlongArrayToArray<4>(env, shapeSize);
    OPS_INFO info;
    RUN_RESHAPE(times, static_cast<ggml_type>(typeSrc), ne_array, shape_array, info);
    
    jdoubleArray result = env->NewDoubleArray(1);
    jdouble timing = info.time_per_op_ms;
    env->SetDoubleArrayRegion(result, 0, 1, &timing);
    return result;
}

// RUN_RMS_NORM wrapper
static jdoubleArray rmsNorm(JNIEnv* env, jclass, jint times, jint typeSrc, jlongArray ne) {
    auto ne_array = jlongArrayToArray<4>(env, ne);
    OPS_INFO info;
    RUN_RMS_NORM(times, static_cast<ggml_type>(typeSrc), ne_array, info);
    
    jdoubleArray result = env->NewDoubleArray(1);
    jdouble timing = info.time_per_op_ms;
    env->SetDoubleArrayRegion(result, 0, 1, &timing);
    return result;
}

// RUN_VIEW_1D wrapper
static jdoubleArray view1D(JNIEnv* env, jclass, jint times, jint typeSrc, jlongArray ne, jintArray viewAxes) {
    auto ne_array = jlongArrayToArray<4>(env, ne);
    auto view_array = jintArrayToArray<1>(env, viewAxes);
    OPS_INFO info;
    RUN_VIEW_1D(times, static_cast<ggml_type>(typeSrc), ne_array, view_array, info);
    
    jdoubleArray result = env->NewDoubleArray(1);
    jdouble timing = info.time_per_op_ms;
    env->SetDoubleArrayRegion(result, 0, 1, &timing);
    return result;
}

// RUN_VIEW_2D wrapper
static jdoubleArray view2D(JNIEnv* env, jclass, jint times, jint typeSrc, jlongArray ne, jintArray viewAxes) {
    auto ne_array = jlongArrayToArray<4>(env, ne);
    auto view_array = jintArrayToArray<2>(env, viewAxes);
    OPS_INFO info;
    RUN_VIEW_2D(times, static_cast<ggml_type>(typeSrc), ne_array, view_array, info);
    
    jdoubleArray result = env->NewDoubleArray(1);
    jdouble timing = info.time_per_op_ms;
    env->SetDoubleArrayRegion(result, 0, 1, &timing);
    return result;
}

// RUN_VIEW_3D wrapper
static jdoubleArray view3D(JNIEnv* env, jclass, jint times, jint typeSrc, jlongArray ne, jintArray viewAxes) {
    auto ne_array = jlongArrayToArray<4>(env, ne);
    auto view_array = jintArrayToArray<3>(env, viewAxes);
    OPS_INFO info;
    RUN_VIEW_3D(times, static_cast<ggml_type>(typeSrc), ne_array, view_array, info);
    
    jdoubleArray result = env->NewDoubleArray(1);
    jdouble timing = info.time_per_op_ms;
    env->SetDoubleArrayRegion(result, 0, 1, &timing);
    return result;
}

// RUN_VIEW_4D wrapper
static jdoubleArray view4D(JNIEnv* env, jclass, jint times, jint typeSrc, jlongArray ne, jintArray viewAxes) {
    auto ne_array = jlongArrayToArray<4>(env, ne);
    auto view_array = jintArrayToArray<4>(env, viewAxes);
    OPS_INFO info;
    RUN_VIEW_4D(times, static_cast<ggml_type>(typeSrc), ne_array, view_array, info);
    
    jdoubleArray result = env->NewDoubleArray(1);
    jdouble timing = info.time_per_op_ms;
    env->SetDoubleArrayRegion(result, 0, 1, &timing);
    return result;
}

// 动态注册的方法表
static const JNINativeMethod methods[] = {
    {const_cast<char*>("add"), const_cast<char*>("(II[J)[D"), (void*)add},
    {const_cast<char*>("cpy"), const_cast<char*>("(III[J)[D"), (void*)cpy},
    {const_cast<char*>("flashAttnExt"), const_cast<char*>("(IJJJ[JJJZZFFII)[D"), (void*)flashAttnExt},
    {const_cast<char*>("gelu"), const_cast<char*>("(II[J)[D"), (void*)gelu},
    {const_cast<char*>("mul"), const_cast<char*>("(II[J)[D"), (void*)mul},
    {const_cast<char*>("scale"), const_cast<char*>("(II[JFF)[D"), (void*)scale},
    {const_cast<char*>("swiglu"), const_cast<char*>("(II[J)[D"), (void*)swiglu},
    {const_cast<char*>("swigluOai"), const_cast<char*>("(II[JFF)[D"), (void*)swigluOai},
    {const_cast<char*>("mulMat"), const_cast<char*>("(II[J)[D"), (void*)mulMat},
    {const_cast<char*>("norm"), const_cast<char*>("(II[J)[D"), (void*)norm},
    {const_cast<char*>("permute"), const_cast<char*>("(II[J[I)[D"), (void*)permute},
    {const_cast<char*>("reshape"), const_cast<char*>("(II[J[J)[D"), (void*)reshape},
    {const_cast<char*>("rmsNorm"), const_cast<char*>("(II[J)[D"), (void*)rmsNorm},
    {const_cast<char*>("view1D"), const_cast<char*>("(II[J[I)[D"), (void*)view1D},
    {const_cast<char*>("view2D"), const_cast<char*>("(II[J[I)[D"), (void*)view2D},
    {const_cast<char*>("view3D"), const_cast<char*>("(II[J[I)[D"), (void*)view3D},
    {const_cast<char*>("view4D"), const_cast<char*>("(II[J[I)[D"), (void*)view4D}
};

// JNI_OnLoad函数 - 动态注册
JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved) {
    JNIEnv* env;
    if (vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) != JNI_OK) {
        return JNI_ERR;
    }

    // 查找Java类
    jclass clazz = env->FindClass("com/infrapower/OpsJNI");
    if (clazz == nullptr) {
        return JNI_ERR;
    }

    // 注册native方法
    if (env->RegisterNatives(clazz, methods, sizeof(methods) / sizeof(methods[0])) < 0) {
        return JNI_ERR;
    }

    return JNI_VERSION_1_6;
}
