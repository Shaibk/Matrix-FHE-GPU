// src/core/batched_trace.cu

#include "core/batched_trace.cuh"
#include "core/config.h"
#include <cuda_runtime.h>
#include <cstdint>

namespace matrix_fhe {

// ------------------------------------------------------------------
// Modular Arithmetic Helpers
// ------------------------------------------------------------------
// Use HE moduli from HE.cu constant memory
extern __constant__ uint64_t d_he_moduli[RNS_NUM_LIMBS];

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
    
    // W-CRT domain: no permutation across W slots
    int src_batch_idx = dst_batch_idx;

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

    for (int limb = 0; limb < rns_limbs; ++limb) {
        uint64_t q = d_he_moduli[limb];
        uint64_t a = B_real[src_offset + limb * n2 + idx];
        uint64_t b = B_imag[src_offset + limb * n2 + idx];

        uint64_t b_conj = neg_mod(b, q);

        if (j == 0) {
            Bp_real[dst_offset + limb * n2 + dst] = a;
            Bp_imag[dst_offset + limb * n2 + dst] = b_conj;
        } else {
            Bp_real[dst_offset + limb * n2 + dst] = neg_mod(b, q); 
            Bp_imag[dst_offset + limb * n2 + dst] = neg_mod(a, q); 
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
    long long batch_offset = (long long)batch_idx * ((long long)RNS_NUM_LIMBS * n2);
    int out_idx = row * n + col;

    for (int limb = 0; limb < RNS_NUM_LIMBS; ++limb) {
        uint64_t q = d_he_moduli[limb];
        uint64_t acc_r = 0, acc_i = 0;
        const uint64_t* Ar = A_real  + batch_offset + limb * n2;
        const uint64_t* Ai = A_imag  + batch_offset + limb * n2;
        const uint64_t* Br = Bp_real + batch_offset + limb * n2;
        const uint64_t* Bi = Bp_imag + batch_offset + limb * n2;

        for (int t = 0; t < n; t++) {
            int a_idx = row * n + t;
            int b_idx = col * n + t;

            uint64_t ar = Ar[a_idx], ai = Ai[a_idx];
            uint64_t br = Br[b_idx], bi = Bi[b_idx];

            uint64_t arbr = mul_mod_u128(ar, br, q);
            uint64_t aibi = mul_mod_u128(ai, bi, q);
            uint64_t arbi = mul_mod_u128(ar, bi, q);
            uint64_t aibr = mul_mod_u128(ai, br, q);

            uint64_t prod_r = sub_mod(arbr, aibi, q);
            uint64_t prod_i = add_mod(arbi, aibr, q);

            acc_r = add_mod(acc_r, prod_r, q);
            acc_i = add_mod(acc_i, prod_i, q);
        }
        uint64_t n_mod = (uint64_t)n % q;
        C_real[batch_offset + limb * n2 + out_idx] = mul_mod_u128(acc_r, n_mod, q);
        C_imag[batch_offset + limb * n2 + out_idx] = mul_mod_u128(acc_i, n_mod, q);
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
    long long batch_offset = (long long)batch_idx * ((long long)RNS_NUM_LIMBS * n2);

    if (idx >= n2) return;

    for (int limb = 0; limb < RNS_NUM_LIMBS; ++limb) {
        uint64_t q = d_he_moduli[limb];
        uint64_t r = Cre[batch_offset + limb*n2 + idx];
        uint64_t i = Cim[batch_offset + limb*n2 + idx];
        uint64_t inv = (limb == 0) ? inv0 : (limb == 1 ? inv1 : (limb == 2 ? inv2 : 0));
        Cre[batch_offset + limb*n2 + idx] = mul_mod_u128(r, inv, q);
        Cim[batch_offset + limb*n2 + idx] = mul_mod_u128(i, inv, q);
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
