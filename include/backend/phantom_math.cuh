#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace phantom_math {

// Phantom 的 Shoup 预计算常数生成器 (Host/Device)
// w = 2^64, p = modulus. return floor(w * w / p)
__host__ __device__ inline uint64_t compute_shoup(uint64_t operand, uint64_t modulus) {
    unsigned __int128 w = (unsigned __int128)1 << 64;
    unsigned __int128 p = (unsigned __int128)operand * w; // p = operand * 2^64
    return (uint64_t)(p / modulus);
}

// 核心：Shoup 模乘 (a * b) % q
// 利用预计算的 quotient (b_shoup) 避免除法，速度极快
__device__ inline uint64_t multiply_and_reduce_shoup(uint64_t a, uint64_t b, uint64_t b_shoup, uint64_t q) {
    unsigned __int128 q_hi = (unsigned __int128)a * b_shoup; 
    uint64_t q_est = (uint64_t)(q_hi >> 64); // 估算的商
    
    // result = a*b - q_est*q
    // 利用无符号溢出特性，只计算低 64 位即可
    uint64_t res = a * b - q_est * q;
    
    // 修正误差 (最多减一次 q)
    if (res >= q) res -= q;
    return res;
}

} 
