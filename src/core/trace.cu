#include "core/trace.cuh"
#include "core/config.h"
#include <cuda_runtime.h>
#include <cstdint>

namespace matrix_fhe {

// compile-time moduli to avoid device indexing into RNS_MODULI[]
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

    // limb 0
    {
        uint64_t a = B_real[0 * n2 + idx];
        uint64_t b = B_imag[0 * n2 + idx];

        // conj: a + i b  ->  a - i b
        uint64_t b_conj = neg_mod(b, Q0);

        if (j == 0) {
            // scalar = 1
            Bp_real[0 * n2 + dst] = a;
            Bp_imag[0 * n2 + dst] = b_conj;
        } else {
            // scalar = -i: (-i) * (a - i b) = -b - i a
            Bp_real[0 * n2 + dst] = neg_mod(b, Q0);  // -b
            Bp_imag[0 * n2 + dst] = neg_mod(a, Q0);  // -a
        }
    }

    // limb 1
    {
        uint64_t a = B_real[1 * n2 + idx];
        uint64_t b = B_imag[1 * n2 + idx];

        uint64_t b_conj = neg_mod(b, Q1);

        if (j == 0) {
            Bp_real[1 * n2 + dst] = a;
            Bp_imag[1 * n2 + dst] = b_conj;
        } else {
            Bp_real[1 * n2 + dst] = neg_mod(b, Q1);
            Bp_imag[1 * n2 + dst] = neg_mod(a, Q1);
        }
    }

    // limb 2
    {
        uint64_t a = B_real[2 * n2 + idx];
        uint64_t b = B_imag[2 * n2 + idx];

        uint64_t b_conj = neg_mod(b, Q2);

        if (j == 0) {
            Bp_real[2 * n2 + dst] = a;
            Bp_imag[2 * n2 + dst] = b_conj;
        } else {
            Bp_real[2 * n2 + dst] = neg_mod(b, Q2);
            Bp_imag[2 * n2 + dst] = neg_mod(a, Q2);
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

    // limb 0
    {
        uint64_t acc_r = 0, acc_i = 0;
        const uint64_t* Ar = A_real  + 0 * n2;
        const uint64_t* Ai = A_imag  + 0 * n2;
        const uint64_t* Br = Bp_real + 0 * n2;
        const uint64_t* Bi = Bp_imag + 0 * n2;

        for (int t = 0; t < n; t++) {
            int a_idx = row * n + t;
            int b_idx = col * n + t; // B'[col,t]

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
        C_real[0 * n2 + out_idx] = acc_r;
        C_imag[0 * n2 + out_idx] = acc_i;
    }

    // limb 1
    {
        uint64_t acc_r = 0, acc_i = 0;
        const uint64_t* Ar = A_real  + 1 * n2;
        const uint64_t* Ai = A_imag  + 1 * n2;
        const uint64_t* Br = Bp_real + 1 * n2;
        const uint64_t* Bi = Bp_imag + 1 * n2;

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
        C_real[1 * n2 + out_idx] = acc_r;
        C_imag[1 * n2 + out_idx] = acc_i;
    }

    // limb 2
    {
        uint64_t acc_r = 0, acc_i = 0;
        const uint64_t* Ar = A_real  + 2 * n2;
        const uint64_t* Ai = A_imag  + 2 * n2;
        const uint64_t* Br = Bp_real + 2 * n2;
        const uint64_t* Bi = Bp_imag + 2 * n2;

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
        C_real[2 * n2 + out_idx] = acc_r;
        C_imag[2 * n2 + out_idx] = acc_i;
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
static constexpr uint64_t INV_DELTA_Q0 = 80764681003797ULL;
static constexpr uint64_t INV_DELTA_Q1 = 1098975018896ULL;
static constexpr uint64_t INV_DELTA_Q2 = 14066253496755ULL;

__global__ void rescale_by_delta_kernel(uint64_t* Cre, uint64_t* Cim, int n2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n2) return;

    // limb 0
    {
        uint64_t r = Cre[0 * n2 + idx];
        uint64_t i = Cim[0 * n2 + idx];
        Cre[0 * n2 + idx] = mul_mod_u128(r, INV_DELTA_Q0, Q0);
        Cim[0 * n2 + idx] = mul_mod_u128(i, INV_DELTA_Q0, Q0);
    }
    // limb 1
    {
        uint64_t r = Cre[1 * n2 + idx];
        uint64_t i = Cim[1 * n2 + idx];
        Cre[1 * n2 + idx] = mul_mod_u128(r, INV_DELTA_Q1, Q1);
        Cim[1 * n2 + idx] = mul_mod_u128(i, INV_DELTA_Q1, Q1);
    }
    // limb 2
    {
        uint64_t r = Cre[2 * n2 + idx];
        uint64_t i = Cim[2 * n2 + idx];
        Cre[2 * n2 + idx] = mul_mod_u128(r, INV_DELTA_Q2, Q2);
        Cim[2 * n2 + idx] = mul_mod_u128(i, INV_DELTA_Q2, Q2);
    }
}

__global__ void rescale_by_delta_kernel(uint64_t* Cre, uint64_t* Cim, int n2,
                                        uint64_t inv0, uint64_t inv1, uint64_t inv2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n2) return;

    // limb0
    {
        uint64_t r = Cre[0*n2 + idx], i = Cim[0*n2 + idx];
        Cre[0*n2 + idx] = mul_mod_u128(r, inv0, Q0);
        Cim[0*n2 + idx] = mul_mod_u128(i, inv0, Q0);
    }
    // limb1
    {
        uint64_t r = Cre[1*n2 + idx], i = Cim[1*n2 + idx];
        Cre[1*n2 + idx] = mul_mod_u128(r, inv1, Q1);
        Cim[1*n2 + idx] = mul_mod_u128(i, inv1, Q1);
    }
    // limb2
    {
        uint64_t r = Cre[2*n2 + idx], i = Cim[2*n2 + idx];
        Cre[2*n2 + idx] = mul_mod_u128(r, inv2, Q2);
        Cim[2*n2 + idx] = mul_mod_u128(i, inv2, Q2);
    }
}

void rescale_by_delta_rns(uint64_t* C_real, uint64_t* C_imag, int n, int /*rns_limbs*/,
                          uint64_t inv0, uint64_t inv1, uint64_t inv2) {
    int n2 = n*n;
    int threads = 256;
    int blocks  = (n2 + threads - 1) / threads;
    rescale_by_delta_kernel<<<blocks, threads>>>(C_real, C_imag, n2, inv0, inv1, inv2);
}


} // namespace matrix_fhe
