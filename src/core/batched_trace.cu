// src/core/batched_trace.cu

#include "core/batched_trace.cuh"
#include "core/config.h"
#include <cuda_runtime.h>
#include <cstdint>

namespace matrix_fhe {

// ------------------------------------------------------------------
// Modular Arithmetic Helpers
// ------------------------------------------------------------------
// 使用 config.h 中的 RNS_MODULI
static constexpr uint64_t Q0 = RNS_MODULI[0];
static constexpr uint64_t Q1 = RNS_MODULI[1];
static constexpr uint64_t Q2 = RNS_MODULI[2];

static __device__ __forceinline__ uint64_t add_mod(uint64_t a, uint64_t b, uint64_t q) {
    uint64_t s = a + b;
    return (s >= q || s < a) ? (s - q) : s;
}

static __device__ __forceinline__ uint64_t sub_mod(uint64_t a, uint64_t b, uint64_t q) {
    return (a >= b) ? (a - b) : (q - (b - a));
}

static __device__ __forceinline__ uint64_t mul_mod_u128(uint64_t a, uint64_t b, uint64_t q) {
    unsigned __int128 p = (unsigned __int128)a * (unsigned __int128)b;
    return (uint64_t)(p % (unsigned __int128)q);
}

static __device__ __forceinline__ uint64_t neg_mod(uint64_t a, uint64_t q) {
    return (a == 0) ? 0 : (q - a);
}

// ------------------------------------------------------------------
// Kernel 1: Map B -> B' (Batched with W^-1 Permutation)
// ------------------------------------------------------------------
__global__ void map_Bprime_batched_kernel(
    const uint64_t* B_real, const uint64_t* B_imag,
    uint64_t* Bp_real, uint64_t* Bp_imag,
    int n, int rns_limbs, int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Pixel index inside one matrix
    int dst_batch_idx = blockIdx.z;                  // Output Batch index (l_out)
    
    // === 关键逻辑: W^-1 Permutation ===
    // 在 Evaluation 域中，W^-1 对应输入索引的逆序: src = 15 - dst
    int src_batch_idx = (batch_size - 1) - dst_batch_idx;

    int n2 = n * n;
    int matrix_size = n2 * rns_limbs;
    
    // 计算读写偏移量
    long long dst_offset = (long long)dst_batch_idx * matrix_size;
    long long src_offset = (long long)src_batch_idx * matrix_size;

    if (idx >= n2) return;

    int j = idx / n;
    int k = idx - j * n;

    // Logic for X^{-1} twist with X^n = i
    int j_dst = (n - j) & (n - 1);  
    int dst = j_dst * n + k;

    // Process Limb 0
    {
        uint64_t a = B_real[src_offset + 0 * n2 + idx];
        uint64_t b = B_imag[src_offset + 0 * n2 + idx];

        // Conjugate: a + ib -> a - ib
        uint64_t b_conj = neg_mod(b, Q0);

        if (j == 0) {
            Bp_real[dst_offset + 0 * n2 + dst] = a;
            Bp_imag[dst_offset + 0 * n2 + dst] = b_conj;
        } else {
            // mult by -i: (-i)*(a - ib) = -b - ia
            Bp_real[dst_offset + 0 * n2 + dst] = neg_mod(b, Q0); 
            Bp_imag[dst_offset + 0 * n2 + dst] = neg_mod(a, Q0); 
        }
    }

    // Process Limb 1
    {
        uint64_t a = B_real[src_offset + 1 * n2 + idx];
        uint64_t b = B_imag[src_offset + 1 * n2 + idx];
        uint64_t b_conj = neg_mod(b, Q1);

        if (j == 0) {
            Bp_real[dst_offset + 1 * n2 + dst] = a;
            Bp_imag[dst_offset + 1 * n2 + dst] = b_conj;
        } else {
            Bp_real[dst_offset + 1 * n2 + dst] = neg_mod(b, Q1);
            Bp_imag[dst_offset + 1 * n2 + dst] = neg_mod(a, Q1);
        }
    }

    // Process Limb 2
    {
        uint64_t a = B_real[src_offset + 2 * n2 + idx];
        uint64_t b = B_imag[src_offset + 2 * n2 + idx];
        uint64_t b_conj = neg_mod(b, Q2);

        if (j == 0) {
            Bp_real[dst_offset + 2 * n2 + dst] = a;
            Bp_imag[dst_offset + 2 * n2 + dst] = b_conj;
        } else {
            Bp_real[dst_offset + 2 * n2 + dst] = neg_mod(b, Q2);
            Bp_imag[dst_offset + 2 * n2 + dst] = neg_mod(a, Q2);
        }
    }
}

// Host Wrapper for Map
void map_B_to_Bprime_batched(
    const uint64_t* B_real, const uint64_t* B_imag,
    uint64_t* Bp_real, uint64_t* Bp_imag,
    int n, int rns_limbs, int batch_size
) {
    int n2 = n * n;
    int threads = 256;
    int blocks_x = (n2 + threads - 1) / threads;
    
    dim3 grid(blocks_x, 1, batch_size);
    map_Bprime_batched_kernel<<<grid, threads>>>(B_real, B_imag, Bp_real, Bp_imag, n, rns_limbs, batch_size);
}

// ------------------------------------------------------------------
// Kernel 2: Batched Trace GEMM
// C = A * (B')^T for each batch independently
// ------------------------------------------------------------------
__global__ void trace_gemm_batched_kernel(
    const uint64_t* A_real, const uint64_t* A_imag,
    const uint64_t* Bp_real, const uint64_t* Bp_imag,
    uint64_t* C_real, uint64_t* C_imag,
    int n
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x; 
    int batch_idx = blockIdx.z;

    if (row >= n || col >= n) return;

    int n2 = n * n;
    // Offset for this specific matrix in the batch
    long long batch_offset = (long long)batch_idx * (3 * n2);
    int out_idx = row * n + col;

    // --- Limb 0 ---
    {
        uint64_t acc_r = 0, acc_i = 0;
        const uint64_t* Ar = A_real  + batch_offset + 0 * n2;
        const uint64_t* Ai = A_imag  + batch_offset + 0 * n2;
        const uint64_t* Br = Bp_real + batch_offset + 0 * n2;
        const uint64_t* Bi = Bp_imag + batch_offset + 0 * n2;

        for (int t = 0; t < n; t++) {
            int a_idx = row * n + t;
            int b_idx = col * n + t; // Transposed access to B'

            uint64_t ar = Ar[a_idx], ai = Ai[a_idx];
            uint64_t br = Br[b_idx], bi = Bi[b_idx];

            uint64_t arbr = mul_mod_u128(ar, br, Q0);
            uint64_t aibi = mul_mod_u128(ai, bi, Q0);
            uint64_t arbi = mul_mod_u128(ar, bi, Q0);
            uint64_t aibr = mul_mod_u128(ai, br, Q0);

            uint64_t prod_r = sub_mod(arbr, aibi, Q0);
            uint64_t prod_i = add_mod(arbi, aibr, Q0);

            acc_r = add_mod(acc_r, prod_r, Q0);
            acc_i = add_mod(acc_i, prod_i, Q0);
        }
        C_real[batch_offset + 0 * n2 + out_idx] = acc_r;
        C_imag[batch_offset + 0 * n2 + out_idx] = acc_i;
    }

    // --- Limb 1 ---
    {
        uint64_t acc_r = 0, acc_i = 0;
        const uint64_t* Ar = A_real  + batch_offset + 1 * n2;
        const uint64_t* Ai = A_imag  + batch_offset + 1 * n2;
        const uint64_t* Br = Bp_real + batch_offset + 1 * n2;
        const uint64_t* Bi = Bp_imag + batch_offset + 1 * n2;

        for (int t = 0; t < n; t++) {
            int a_idx = row * n + t;
            int b_idx = col * n + t;

            uint64_t ar = Ar[a_idx], ai = Ai[a_idx];
            uint64_t br = Br[b_idx], bi = Bi[b_idx];

            uint64_t arbr = mul_mod_u128(ar, br, Q1);
            uint64_t aibi = mul_mod_u128(ai, bi, Q1);
            uint64_t arbi = mul_mod_u128(ar, bi, Q1);
            uint64_t aibr = mul_mod_u128(ai, br, Q1);

            uint64_t prod_r = sub_mod(arbr, aibi, Q1);
            uint64_t prod_i = add_mod(arbi, aibr, Q1);

            acc_r = add_mod(acc_r, prod_r, Q1);
            acc_i = add_mod(acc_i, prod_i, Q1);
        }
        C_real[batch_offset + 1 * n2 + out_idx] = acc_r;
        C_imag[batch_offset + 1 * n2 + out_idx] = acc_i;
    }

    // --- Limb 2 ---
    {
        uint64_t acc_r = 0, acc_i = 0;
        const uint64_t* Ar = A_real  + batch_offset + 2 * n2;
        const uint64_t* Ai = A_imag  + batch_offset + 2 * n2;
        const uint64_t* Br = Bp_real + batch_offset + 2 * n2;
        const uint64_t* Bi = Bp_imag + batch_offset + 2 * n2;

        for (int t = 0; t < n; t++) {
            int a_idx = row * n + t;
            int b_idx = col * n + t;

            uint64_t ar = Ar[a_idx], ai = Ai[a_idx];
            uint64_t br = Br[b_idx], bi = Bi[b_idx];

            uint64_t arbr = mul_mod_u128(ar, br, Q2);
            uint64_t aibi = mul_mod_u128(ai, bi, Q2);
            uint64_t arbi = mul_mod_u128(ar, bi, Q2);
            uint64_t aibr = mul_mod_u128(ai, br, Q2);

            uint64_t prod_r = sub_mod(arbr, aibi, Q2);
            uint64_t prod_i = add_mod(arbi, aibr, Q2);

            acc_r = add_mod(acc_r, prod_r, Q2);
            acc_i = add_mod(acc_i, prod_i, Q2);
        }
        C_real[batch_offset + 2 * n2 + out_idx] = acc_r;
        C_imag[batch_offset + 2 * n2 + out_idx] = acc_i;
    }
}

// Host Wrapper for Trace GEMM
void trace_gemm_batched(
    const uint64_t* A_real, const uint64_t* A_imag,
    const uint64_t* Bp_real, const uint64_t* Bp_imag,
    uint64_t* C_real, uint64_t* C_imag,
    int n, int rns_limbs, int batch_size
) {
    dim3 thr(16, 16);
    dim3 blk((n + 15) / 16, (n + 15) / 16, batch_size);
    trace_gemm_batched_kernel<<<blk, thr>>>(A_real, A_imag, Bp_real, Bp_imag, C_real, C_imag, n);
}

// ------------------------------------------------------------------
// Kernel 3: Batched Rescaling
// ------------------------------------------------------------------
__global__ void rescale_by_delta_batched_kernel(
    uint64_t* Cre, uint64_t* Cim, int n2,
    uint64_t inv0, uint64_t inv1, uint64_t inv2
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.z;
    
    // Offset calculation
    long long batch_offset = (long long)batch_idx * (3 * n2);

    if (idx >= n2) return;

    // Limb 0
    {
        uint64_t r = Cre[batch_offset + 0*n2 + idx];
        uint64_t i = Cim[batch_offset + 0*n2 + idx];
        Cre[batch_offset + 0*n2 + idx] = mul_mod_u128(r, inv0, Q0);
        Cim[batch_offset + 0*n2 + idx] = mul_mod_u128(i, inv0, Q0);
    }
    // Limb 1
    {
        uint64_t r = Cre[batch_offset + 1*n2 + idx];
        uint64_t i = Cim[batch_offset + 1*n2 + idx];
        Cre[batch_offset + 1*n2 + idx] = mul_mod_u128(r, inv1, Q1);
        Cim[batch_offset + 1*n2 + idx] = mul_mod_u128(i, inv1, Q1);
    }
    // Limb 2
    {
        uint64_t r = Cre[batch_offset + 2*n2 + idx];
        uint64_t i = Cim[batch_offset + 2*n2 + idx];
        Cre[batch_offset + 2*n2 + idx] = mul_mod_u128(r, inv2, Q2);
        Cim[batch_offset + 2*n2 + idx] = mul_mod_u128(i, inv2, Q2);
    }
}

// Host Wrapper for Rescale
void rescale_by_delta_batched(
    uint64_t* C_real, uint64_t* C_imag, 
    int n, int /*rns_limbs*/, int batch_size,
    uint64_t inv0, uint64_t inv1, uint64_t inv2
) {
    int n2 = n * n;
    int threads = 256;
    int blocks_x = (n2 + threads - 1) / threads;
    
    dim3 grid(blocks_x, 1, batch_size);
    rescale_by_delta_batched_kernel<<<grid, threads>>>(C_real, C_imag, n2, inv0, inv1, inv2);
}

} // namespace matrix_fhe