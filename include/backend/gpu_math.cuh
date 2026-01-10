#pragma once
#include <cuda_runtime.h>
#include "../core/config.h"

namespace matrix_fhe {

// =========================================================
// RNS-Compatible Gaussian Integer
// Represents a value in Z_Q[i] decomposed into k components
// =========================================================
struct __align__(16) GaussianIntRNS {
    // 布局：我们可以用 Structure of Arrays (SoA) 或 Array of Structures (AoS)
    // 这里为了直观，用 AoS: 每个元素包含 3 个 limb 的实部和 3 个 limb 的虚部
    // x[0] mod q0, x[1] mod q1, x[2] mod q2
    uint64_t x[RNS_NUM_LIMBS]; 
    uint64_t y[RNS_NUM_LIMBS];

    // --- RNS Modular Arithmetic (Single Component) ---
    
    // (a + b) mod q
    __device__ static inline uint64_t add_mod(uint64_t a, uint64_t b, uint64_t q) {
        uint64_t res = a + b;
        if (res >= q) res -= q;
        return res;
    }

    // (a - b) mod q
    __device__ static inline uint64_t sub_mod(uint64_t a, uint64_t b, uint64_t q) {
        if (a >= b) return a - b;
        else return a + q - b;
    }

    // (a * b) mod q (using 128-bit intermediate)
    __device__ static inline uint64_t mul_mod(uint64_t a, uint64_t b, uint64_t q) {
        unsigned __int128 res = (unsigned __int128)a * b;
        return (uint64_t)(res % q); 
        // 性能提示：在深度优化时，这里应该用 Barrett Reduction 替换 %
    }

    // --- RNS Complex Operations (Vectorized) ---
    
    // 加法：每个 limb 独立相加
    __device__ static GaussianIntRNS add(const GaussianIntRNS& A, const GaussianIntRNS& B, const uint64_t* moduli) {
        GaussianIntRNS res;
        #pragma unroll
        for(int i=0; i<RNS_NUM_LIMBS; ++i) {
            res.x[i] = add_mod(A.x[i], B.x[i], moduli[i]);
            res.y[i] = add_mod(A.y[i], B.y[i], moduli[i]);
        }
        return res;
    }

    // 乘法：(a+bi)(c+di) = (ac-bd) + (ad+bc)i
    __device__ static GaussianIntRNS mul(const GaussianIntRNS& A, const GaussianIntRNS& B, const uint64_t* moduli) {
        GaussianIntRNS res;
        #pragma unroll
        for(int i=0; i<RNS_NUM_LIMBS; ++i) {
            uint64_t q = moduli[i];
            uint64_t ac = mul_mod(A.x[i], B.x[i], q);
            uint64_t bd = mul_mod(A.y[i], B.y[i], q);
            uint64_t ad = mul_mod(A.x[i], B.y[i], q);
            uint64_t bc = mul_mod(A.y[i], B.x[i], q);
            
            res.x[i] = sub_mod(ac, bd, q);
            res.y[i] = add_mod(ad, bc, q);
        }
        return res;
    }

    // 共轭
    __device__ static GaussianIntRNS conj(const GaussianIntRNS& A, const uint64_t* moduli) {
        GaussianIntRNS res;
        #pragma unroll
        for(int i=0; i<RNS_NUM_LIMBS; ++i) {
            res.x[i] = A.x[i];
            res.y[i] = (A.y[i] == 0) ? 0 : moduli[i] - A.y[i];
        }
        return res;
    }

    // 乘以 -i: (b, -a)
    __device__ static GaussianIntRNS mul_minus_i(const GaussianIntRNS& A, const uint64_t* moduli) {
        GaussianIntRNS res;
        #pragma unroll
        for(int i=0; i<RNS_NUM_LIMBS; ++i) {
            res.x[i] = A.y[i]; // Real part becomes b
            res.y[i] = (A.x[i] == 0) ? 0 : moduli[i] - A.x[i]; // Imag part becomes -a
        }
        return res;
    }
};

} // namespace matrix_fhe