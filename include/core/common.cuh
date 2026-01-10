#pragma once
#include <cuda_runtime.h>
#include <cstdint>

// 引入 Phantom 的数学库
// 假设 common.cuh 在 include/core/，uintmodmath.cuh 在 extern/phantom-fhe/include/
#include "../../extern/phantom-fhe/include/uintmodmath.cuh"

namespace matrix_fhe {

// 使用 Phantom 的高效算子
static __device__ __forceinline__ uint64_t add_mod(uint64_t a, uint64_t b, uint64_t q) {
    return phantom::arith::add_uint64_uint64_mod(a, b, q);
}

static __device__ __forceinline__ uint64_t sub_mod(uint64_t a, uint64_t b, uint64_t q) {
    return phantom::arith::sub_uint64_uint64_mod(a, b, q);
}

static __device__ __forceinline__ uint64_t mul_mod(uint64_t a, uint64_t b, uint64_t q) {
    // 使用简单的 __int128 版本，避免依赖 mu 参数，方便集成到现有 kernel
    unsigned __int128 res = (unsigned __int128)a * b;
    return (uint64_t)(res % q);
}

static __device__ __forceinline__ uint64_t pow_mod(uint64_t base, uint64_t exp, uint64_t q) {
    uint64_t res = 1;
    base %= q;
    while (exp > 0) {
        if (exp % 2 == 1) res = mul_mod(res, base, q);
        base = mul_mod(base, base, q);
        exp /= 2;
    }
    return res;
}

static __device__ __forceinline__ uint64_t inv_mod(uint64_t n, uint64_t q) {
    return pow_mod(n, q - 2, q);
}

} // namespace matrix_fhe