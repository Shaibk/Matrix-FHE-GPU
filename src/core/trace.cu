#include "core/trace.cuh"
#include "core/config.h"
#include <cuda_runtime.h>
#include <cstdint>

namespace matrix_fhe {

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

// ------------------- map B -> B' (X^{-1} with X^n=i twist) + conjugate -------------------
__global__ void map_Bprime_Xinv_twist_kernel(
    const uint64_t* B_real, const uint64_t* B_imag,
    uint64_t* Bp_real, uint64_t* Bp_imag,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n2 = n * n;
    if (idx >= n2) return;

    int j = idx / n;
    int k = idx - j * n;

    int j_dst = (n - j) & (n - 1);  // (-j mod n), n power-of-two
    int dst = j_dst * n + k;

    for (int limb = 0; limb < RNS_NUM_LIMBS; ++limb) {
        uint64_t q = d_he_moduli[limb];
        uint64_t a = B_real[limb * n2 + idx];
        uint64_t b = B_imag[limb * n2 + idx];

        // conj: a + i b  ->  a - i b
        uint64_t b_conj = neg_mod(b, q);

        if (j == 0) {
            Bp_real[limb * n2 + dst] = a;
            Bp_imag[limb * n2 + dst] = b_conj;
        } else {
            // scalar = -i: (-i) * (a - i b) = -b - i a
            Bp_real[limb * n2 + dst] = neg_mod(b, q);
            Bp_imag[limb * n2 + dst] = neg_mod(a, q);
        }
    }
}

void map_B_to_Bprime_Xinv_twist(
    const uint64_t* B_real, const uint64_t* B_imag,
    uint64_t* Bp_real, uint64_t* Bp_imag,
    int n, int /*rns_limbs*/
) {
    int n2 = n * n;
    int threads = 256;
    int blocks = (n2 + threads - 1) / threads;
    map_Bprime_Xinv_twist_kernel<<<blocks, threads>>>(B_real, B_imag, Bp_real, Bp_imag, n);
}


// ------------------- trace GEMM: C = A * (B')^T -------------------
__global__ void trace_gemm_ABpT_rns_kernel(
    const uint64_t* A_real, const uint64_t* A_imag,
    const uint64_t* Bp_real, const uint64_t* Bp_imag,
    uint64_t* C_real, uint64_t* C_imag,
    int n
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // j
    int col = blockIdx.x * blockDim.x + threadIdx.x; // k
    if (row >= n || col >= n) return;

    int n2 = n * n;
    int out_idx = row * n + col;

    for (int limb = 0; limb < RNS_NUM_LIMBS; ++limb) {
        uint64_t q = d_he_moduli[limb];
        uint64_t acc_r = 0, acc_i = 0;
        const uint64_t* Ar = A_real  + limb * n2;
        const uint64_t* Ai = A_imag  + limb * n2;
        const uint64_t* Br = Bp_real + limb * n2;
        const uint64_t* Bi = Bp_imag + limb * n2;

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
        C_real[limb * n2 + out_idx] = mul_mod_u128(acc_r, n_mod, q);
        C_imag[limb * n2 + out_idx] = mul_mod_u128(acc_i, n_mod, q);
    }
}

void trace_gemm_ABpT_rns(
    const uint64_t* A_real, const uint64_t* A_imag,
    const uint64_t* Bp_real, const uint64_t* Bp_imag,
    uint64_t* C_real, uint64_t* C_imag,
    int n, int /*rns_limbs*/
) {
    dim3 thr(16, 16);
    dim3 blk((n + 15) / 16, (n + 15) / 16);
    trace_gemm_ABpT_rns_kernel<<<blk, thr>>>(A_real, A_imag, Bp_real, Bp_imag, C_real, C_imag, n);
}
__global__ void rescale_by_delta_kernel(uint64_t* Cre, uint64_t* Cim, int n2,
                                        const uint64_t* inv) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n2) return;

    for (int limb = 0; limb < RNS_NUM_LIMBS; ++limb) {
        uint64_t q = d_he_moduli[limb];
        uint64_t r = Cre[limb * n2 + idx];
        uint64_t i = Cim[limb * n2 + idx];
        Cre[limb * n2 + idx] = mul_mod_u128(r, inv[limb], q);
        Cim[limb * n2 + idx] = mul_mod_u128(i, inv[limb], q);
    }
}

void rescale_by_delta_rns(uint64_t* C_real, uint64_t* C_imag, int n, int /*rns_limbs*/,
                          uint64_t inv0, uint64_t inv1, uint64_t inv2) {
    int n2 = n*n;
    int threads = 256;
    int blocks  = (n2 + threads - 1) / threads;
    uint64_t inv[RNS_NUM_LIMBS] = {0};
    inv[0] = inv0;
    if (RNS_NUM_LIMBS > 1) inv[1] = inv1;
    if (RNS_NUM_LIMBS > 2) inv[2] = inv2;

    uint64_t* d_inv = nullptr;
    cudaMalloc(&d_inv, RNS_NUM_LIMBS * sizeof(uint64_t));
    cudaMemcpy(d_inv, inv, RNS_NUM_LIMBS * sizeof(uint64_t), cudaMemcpyHostToDevice);
    rescale_by_delta_kernel<<<blocks, threads>>>(C_real, C_imag, n2, d_inv);
    cudaFree(d_inv);
}


} // namespace matrix_fhe
