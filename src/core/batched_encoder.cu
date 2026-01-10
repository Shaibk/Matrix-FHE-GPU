#include "../../include/core/batched_encoder.cuh"
#include "../../include/core/encoder.cuh"
#include "../../include/core/config.h"

#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <algorithm>

namespace matrix_fhe {

// ===================== Device helpers =====================
__device__ __forceinline__ uint64_t d_add_mod(uint64_t a, uint64_t b, uint64_t mod) {
    uint64_t s = a + b;
    if (s >= mod || s < a) s -= mod;
    return s;
}

__device__ __forceinline__ uint64_t d_mul_mod_u128(uint64_t a, uint64_t b, uint64_t mod) {
#if defined(__CUDA_ARCH__)
    unsigned __int128 p = (unsigned __int128)a * (unsigned __int128)b;
    return (uint64_t)(p % mod);
#else
    return 0;
#endif
}

// ===================== Host helpers =====================
static inline uint64_t h_add(uint64_t a, uint64_t b, uint64_t mod) {
    uint64_t s = a + b;
    if (s >= mod || s < a) s -= mod;
    return s;
}
static inline uint64_t h_sub(uint64_t a, uint64_t b, uint64_t mod) {
    return (a >= b) ? (a - b) : (mod - (b - a));
}
static inline uint64_t h_mul_u128(uint64_t a, uint64_t b, uint64_t mod) {
    unsigned __int128 p = (unsigned __int128)a * (unsigned __int128)b;
    return (uint64_t)(p % mod);
}
static uint64_t h_pow(uint64_t a, uint64_t e, uint64_t mod) {
    uint64_t r = 1;
    while (e) {
        if (e & 1) r = h_mul_u128(r, a, mod);
        a = h_mul_u128(a, a, mod);
        e >>= 1;
    }
    return r;
}
static uint64_t h_inv(uint64_t a, uint64_t mod) {
    return h_pow(a, mod - 2, mod);
}

// ===================== Device constants =====================
__constant__ uint64_t d_q[3];
__constant__ uint64_t d_V_q[3][16][16];    // V[ell][r]
__constant__ uint64_t d_Vinv_q[3][16][16];  // Vinv[r][ell]

// ===================== Kernels (Revised for Poly-Major Layout) =====================

/**
 * Packing Kernel: W-Inverse Transform + Transpose
 * * Input Layout:  [Batch=16][Limb=3][Space=N^2] (Linear chunks from SingleEncoder)
 * Output Layout: [Y=n][Limb=3][Pack(X, W)]   (Poly-Major for HE.cu)
 * * Mapping:
 * - idx covers Space (y, x). idx = y * n + x.
 * - r covers W-dim (0..15).
 * - Output Index = y * (3 * RLWE_N) + limb * RLWE_N + (x * 16 + r)
 * Assuming RLWE_N = n * 16 (e.g. 256 * 16 = 4096).
 */
__global__ void pack_w_phi16_kernel(
    const uint64_t* __restrict__ in_re,
    const uint64_t* __restrict__ in_im,
    uint64_t* __restrict__ out_re,
    uint64_t* __restrict__ out_im,
    int n,      // Matrix dimension (e.g., 256)
    int n2      // n * n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n2) return;

    // Decode spatial indices (y, x) from idx
    int y = idx / n;
    int x = idx % n;
    
    // Poly-Major Stride Calculation
    // Total coefficients per limb per polynomial = n * 16
    int rlwe_n = n * 16; 
    
    // Size of one full polynomial block (all limbs)
    int single_poly_size = 3 * rlwe_n;

    #pragma unroll
    for (int limb = 0; limb < 3; ++limb) {
        const uint64_t Q = d_q[limb];

        uint64_t v_re[16];
        uint64_t v_im[16];

        // Gather from input: [Batch][Limb][Space]
        #pragma unroll
        for (int ell = 0; ell < 16; ++ell) {
            size_t off = (size_t)ell * (size_t)3 * (size_t)n2 + (size_t)limb * (size_t)n2 + (size_t)idx;
            v_re[ell] = in_re[off];
            v_im[ell] = in_im[off];
        }

        // W-Inverse Transform: c_r = sum_ell Vinv[r][ell] * v_ell
        #pragma unroll
        for (int r = 0; r < 16; ++r) {
            uint64_t acc_re = 0;
            uint64_t acc_im = 0;

            #pragma unroll
            for (int ell = 0; ell < 16; ++ell) {
                uint64_t w = d_Vinv_q[limb][r][ell];
                uint64_t t_re = d_mul_mod_u128(v_re[ell], w, Q);
                uint64_t t_im = d_mul_mod_u128(v_im[ell], w, Q);
                acc_re = d_add_mod(acc_re, t_re, Q);
                acc_im = d_add_mod(acc_im, t_im, Q);
            }

            // Scatter to output: [Y][Limb][X_W_Packed]
            // We pack W (r) into the fine-grained dimension of X
            int packed_offset_in_poly = x * 16 + r;
            
            size_t out_off = (size_t)y * single_poly_size + 
                             (size_t)limb * rlwe_n + 
                             (size_t)packed_offset_in_poly;

            out_re[out_off] = acc_re;
            out_im[out_off] = acc_im;
        }
    }
}

/**
 * Evaluation Kernel: W-Transform + Transpose Back
 * * Input Layout:  [Y][Limb][Pack(X, W)]
 * Output Layout: [Batch][Limb][Space]
 */
__global__ void eval_w_phi16_kernel(
    const uint64_t* __restrict__ in_re,
    const uint64_t* __restrict__ in_im,
    uint64_t* __restrict__ out_re,
    uint64_t* __restrict__ out_im,
    int n,
    int n2
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n2) return;

    int y = idx / n;
    int x = idx % n;
    int rlwe_n = n * 16;
    int single_poly_size = 3 * rlwe_n;

    #pragma unroll
    for (int limb = 0; limb < 3; ++limb) {
        const uint64_t Q = d_q[limb];

        uint64_t c_re[16];
        uint64_t c_im[16];

        // Gather from Poly-Major Input
        #pragma unroll
        for (int r = 0; r < 16; ++r) {
            int packed_offset_in_poly = x * 16 + r;
            size_t in_off = (size_t)y * single_poly_size + 
                            (size_t)limb * rlwe_n + 
                            (size_t)packed_offset_in_poly;
            
            c_re[r] = in_re[in_off];
            c_im[r] = in_im[in_off];
        }

        // W-Transform: v_ell = sum_r V[ell][r] * c_r
        #pragma unroll
        for (int ell = 0; ell < 16; ++ell) {
            uint64_t acc_re = 0;
            uint64_t acc_im = 0;

            #pragma unroll
            for (int r = 0; r < 16; ++r) {
                uint64_t w = d_V_q[limb][ell][r];
                uint64_t t_re = d_mul_mod_u128(c_re[r], w, Q);
                uint64_t t_im = d_mul_mod_u128(c_im[r], w, Q);
                acc_re = d_add_mod(acc_re, t_re, Q);
                acc_im = d_add_mod(acc_im, t_im, Q);
            }

            // Scatter to Output: [Batch][Limb][Space]
            size_t out_off = (size_t)ell * (size_t)3 * (size_t)n2 + (size_t)limb * (size_t)n2 + (size_t)idx;
            out_re[out_off] = acc_re;
            out_im[out_off] = acc_im;
        }
    }
}

// ===================== Class Implementation =====================

BatchedEncoder::BatchedEncoder(int n)
    : n_(n), n2_(n * n) {
    init_tables_p17();
}

void BatchedEncoder::encode_packed_p17(
    const cuDoubleComplex* d_msg_batch,
    uint64_t* d_out_re,
    uint64_t* d_out_im,
    cudaStream_t stream
) {
    // 16 batches, each is [3 limbs][n*n coeffs]
    const size_t blk_words = (size_t)3 * (size_t)n2_;
    const size_t tmp_words = (size_t)16 * blk_words;

    uint64_t* d_tmp_re = nullptr;
    uint64_t* d_tmp_im = nullptr;
    cudaMalloc(&d_tmp_re, tmp_words * sizeof(uint64_t));
    cudaMalloc(&d_tmp_im, tmp_words * sizeof(uint64_t));

    Encoder encoder(n_);

    // 1. Encode each matrix in the batch (Single Encoder)
    // d_tmp Layout: [Batch][Limb][Space]
    for (int ell = 0; ell < 16; ++ell) {
        const cuDoubleComplex* d_m_ell = d_msg_batch + (size_t)ell * (size_t)n2_;
        uint64_t* d_re_ell = d_tmp_re + (size_t)ell * blk_words;
        uint64_t* d_im_ell = d_tmp_im + (size_t)ell * blk_words;
        encoder.encode(d_m_ell, d_re_ell, d_im_ell);
    }

    // 2. Pack and Transpose to Poly-Major
    int threads = 256;
    int blocks  = (n2_ + threads - 1) / threads;

    pack_w_phi16_kernel<<<blocks, threads, 0, stream>>>(
        d_tmp_re, d_tmp_im, d_out_re, d_out_im, n_, n2_
    );

    cudaFree(d_tmp_re);
    cudaFree(d_tmp_im);
}

void BatchedEncoder::unpack_eval_p17(
    const uint64_t* d_in_re,
    const uint64_t* d_in_im,
    uint64_t* d_eval_re,
    uint64_t* d_eval_im,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks  = (n2_ + threads - 1) / threads;

    eval_w_phi16_kernel<<<blocks, threads, 0, stream>>>(
        d_in_re, d_in_im, d_eval_re, d_eval_im, n_, n2_
    );
}

// ===================== Wrapper Implementation (Fix for Root Cause) =====================
// These function calls are used by HE.cu to invoke the kernels defined in this translation unit
void gpu_pack_w_phi16(const uint64_t* in_re, const uint64_t* in_im, 
                      uint64_t* out_re, uint64_t* out_im, 
                      int n2, cudaStream_t stream) {
    // Assuming n2 passed from HE.cu is total coefficients?
    // Actually, for consistency, let's derive n_ from MATRIX_N logic or assume n2 is n*n
    // If n2 == 65536 (256*256), then n = 256.
    int n = (int)sqrt((double)n2); 
    
    int threads = 256;
    int blocks = (n2 + threads - 1) / threads;
    pack_w_phi16_kernel<<<blocks, threads, 0, stream>>>(in_re, in_im, out_re, out_im, n, n2);
}

void gpu_eval_w_phi16(const uint64_t* in_re, const uint64_t* in_im, 
                      uint64_t* out_re, uint64_t* out_im, 
                      int n2, cudaStream_t stream) {
    int n = (int)sqrt((double)n2); 
    int threads = 256;
    int blocks = (n2 + threads - 1) / threads;
    eval_w_phi16_kernel<<<blocks, threads, 0, stream>>>(in_re, in_im, out_re, out_im, n, n2);
}

void BatchedEncoder::init_tables_p17() {
    uint64_t moduli[3] = {
        140737488252929ULL,
        140737488218113ULL,
        140737488061441ULL 
    };

    cudaMemcpyToSymbol(d_q, moduli, sizeof(moduli));

    uint64_t V_host[3][16][16]    = {};
    uint64_t Vinv_host[3][16][16] = {};

    for (int limb = 0; limb < 3; ++limb) {
        uint64_t q = moduli[limb];

        // Find primitive 17th root omega
        uint64_t omega = 0;
        uint64_t e = (q - 1) / 17;
        const uint64_t MAX_TRIES = 2000;
        for (uint64_t a = 2; a < 2 + MAX_TRIES; ++a) {
            uint64_t cand = h_pow(a, e, q);
            if (cand == 1) continue;
            if (h_pow(cand, 17, q) != 1) continue;
            omega = cand;
            break;
        }

        if (omega == 0) throw std::runtime_error("Failed to find primitive 17th root.");

        uint64_t eta[16];
        for (int ell = 0; ell < 16; ++ell) {
            eta[ell] = h_pow(omega, (uint64_t)(ell + 1), q);
        }

        // V[ell][r] = eta_ell^r
        for (int ell = 0; ell < 16; ++ell) {
            uint64_t pwr = 1;
            for (int r = 0; r < 16; ++r) {
                V_host[limb][ell][r] = pwr;
                pwr = h_mul_u128(pwr, eta[ell], q);
            }
        }

        // Invert V via Gauss-Jordan
        uint64_t A[16][32];
        for (int i = 0; i < 16; ++i) {
            for (int j = 0; j < 16; ++j) A[i][j] = V_host[limb][i][j];
            for (int j = 0; j < 16; ++j) A[i][16 + j] = (i == j) ? 1 : 0;
        }

        for (int col = 0; col < 16; ++col) {
            int piv = col;
            while (piv < 16 && A[piv][col] == 0) piv++;
            if (piv == 16) throw std::runtime_error("V not invertible.");
            if (piv != col) {
                for (int j = 0; j < 32; ++j) std::swap(A[piv][j], A[col][j]);
            }
            uint64_t inv_p = h_inv(A[col][col], q);
            for (int j = 0; j < 32; ++j) A[col][j] = h_mul_u128(A[col][j], inv_p, q);
            for (int row = 0; row < 16; ++row) {
                if (row == col) continue;
                uint64_t f = A[row][col];
                if (f == 0) continue;
                for (int j = 0; j < 32; ++j) {
                    uint64_t t = h_mul_u128(f, A[col][j], q);
                    A[row][j] = h_sub(A[row][j], t, q);
                }
            }
        }

        // Extract Vinv
        for (int r = 0; r < 16; ++r)
            for (int ell = 0; ell < 16; ++ell)
                Vinv_host[limb][r][ell] = A[r][16 + ell]; // Note: A stores [I | Vinv]
    }

    cudaMemcpyToSymbol(d_V_q, V_host, sizeof(V_host));
    cudaMemcpyToSymbol(d_Vinv_q, Vinv_host, sizeof(Vinv_host));
}

} // namespace matrix_fhe