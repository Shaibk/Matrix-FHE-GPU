#include "../../include/core/batched_encoder.cuh"
#include "../../include/core/encoder.cuh"
#include "../../include/core/config.h"
#include "../../include/core/HE.cuh"

#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include <complex>

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
__constant__ uint64_t d_q[RNS_NUM_LIMBS];
static cuDoubleComplex* d_wdft_inv = nullptr; // [eval_w][coeff_w]
static bool wdft_inited = false;

// ===================== Kernels =====================

/**
 * Copy Kernel: W-batched CRT Layout (no W transform)
 * Input/Output Layout: [Batch=phi][Limb][Space=N^2]
 */
__global__ void copy_w_crt_kernel(
    const uint64_t* __restrict__ in_re,
    const uint64_t* __restrict__ in_im,
    uint64_t* __restrict__ out_re,
    uint64_t* __restrict__ out_im,
    size_t total_words
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_words) return;
    out_re[idx] = in_re[idx];
    out_im[idx] = in_im[idx];
}

// Layout convert: poly-major [poly=w*n+y][limb][x] -> matrix-major [w][limb][y][x]
__global__ void poly_to_matrix_layout_kernel(
    const uint64_t* __restrict__ in_poly,
    uint64_t* __restrict__ out_mat,
    int n, int limbs, int phi
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t n2 = (size_t)n * (size_t)n;
    size_t total = (size_t)phi * (size_t)limbs * n2;
    if (idx >= total) return;

    int x = (int)(idx % (size_t)n);
    size_t t = idx / (size_t)n;
    int y = (int)(t % (size_t)n);
    int limb = (int)((t / (size_t)n) % (size_t)limbs);
    int w = (int)(t / ((size_t)n * (size_t)limbs));

    int poly = w * n + y;
    size_t in_off = ((size_t)poly * (size_t)limbs + (size_t)limb) * (size_t)n + (size_t)x;
    out_mat[idx] = in_poly[in_off];
}

__global__ void w_idft_kernel(
    const cuDoubleComplex* __restrict__ in_eval_xy, // [eval_w][n2]
    cuDoubleComplex* __restrict__ out_coeff_xy,     // [coeff_w][n2]
    const cuDoubleComplex* __restrict__ inv_mat,    // [eval_w][coeff_w]
    int n2,
    int phi
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)phi * (size_t)n2;
    if (idx >= total) return;
    int pos = (int)(idx % (size_t)n2);
    int r = (int)(idx / (size_t)n2);
    cuDoubleComplex acc = make_cuDoubleComplex(0.0, 0.0);
    for (int w = 0; w < phi; ++w) {
        cuDoubleComplex a = in_eval_xy[(size_t)w * (size_t)n2 + (size_t)pos];
        cuDoubleComplex v = inv_mat[(size_t)w * (size_t)phi + (size_t)r];
        acc = cuCadd(acc, cuCmul(a, v));
    }
    out_coeff_xy[idx] = acc;
}

__global__ void quantize_coeff_to_rns_kernel(
    const cuDoubleComplex* __restrict__ in_coeff_xy, // [coeff_w][n2]
    uint64_t* __restrict__ out_re_rns,               // [coeff_w][limb][n2]
    uint64_t* __restrict__ out_im_rns,
    int n2,
    int limbs,
    int phi,
    double delta
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)phi * (size_t)limbs * (size_t)n2;
    if (idx >= total) return;
    int pos = (int)(idx % (size_t)n2);
    size_t t = idx / (size_t)n2;
    int limb = (int)(t % (size_t)limbs);
    int r = (int)(t / (size_t)limbs);

    cuDoubleComplex z = in_coeff_xy[(size_t)r * (size_t)n2 + (size_t)pos];
    int64_t xr = llround(cuCreal(z) * delta);
    int64_t xi = llround(cuCimag(z) * delta);
    uint64_t q = d_q[limb];
    int64_t mr = xr % (int64_t)q;
    int64_t mi = xi % (int64_t)q;
    if (mr < 0) mr += (int64_t)q;
    if (mi < 0) mi += (int64_t)q;
    out_re_rns[idx] = (uint64_t)mr;
    out_im_rns[idx] = (uint64_t)mi;
}

// ===================== Class Implementation =====================

BatchedEncoder::BatchedEncoder(int n)
    : n_(n), n2_(n * n) {
    init_tables_p17();
}

void BatchedEncoder::encode_to_wntt_eval(
    const cuDoubleComplex* d_msg_batch,
    uint64_t* d_out_re,
    uint64_t* d_out_im,
    cudaStream_t stream
) {
    const int phi = BATCH_SIZE;
    const int limbs = RNS_NUM_LIMBS;
    const size_t n2 = (size_t)n2_;
    const size_t coeff_words = (size_t)phi * (size_t)limbs * n2;

    cuDoubleComplex* d_xy_coeff = nullptr;     // [phi][n2] after XY-IDFT
    cuDoubleComplex* d_w_coeff = nullptr;      // [phi][n2] after W-IDFT
    uint64_t* d_coeff_re = nullptr;            // [phi][limb][n2] W-coeff RNS
    uint64_t* d_coeff_im = nullptr;
    uint64_t* d_eval_re = nullptr;             // poly-major [poly][limb][x]
    uint64_t* d_eval_im = nullptr;
    uint64_t* d_eval_re_mat = nullptr;         // matrix-major [w][limb][y][x]
    uint64_t* d_eval_im_mat = nullptr;
    cudaMalloc(&d_xy_coeff, (size_t)phi * n2 * sizeof(cuDoubleComplex));
    cudaMalloc(&d_w_coeff, (size_t)phi * n2 * sizeof(cuDoubleComplex));
    cudaMalloc(&d_coeff_re, coeff_words * sizeof(uint64_t));
    cudaMalloc(&d_coeff_im, coeff_words * sizeof(uint64_t));
    cudaMalloc(&d_eval_re, coeff_words * sizeof(uint64_t));
    cudaMalloc(&d_eval_im, coeff_words * sizeof(uint64_t));
    cudaMalloc(&d_eval_re_mat, coeff_words * sizeof(uint64_t));
    cudaMalloc(&d_eval_im_mat, coeff_words * sizeof(uint64_t));

    Encoder encoder(n_);

    // 1) XY-IDFT only (no scale/RNS yet)
    for (int ell = 0; ell < phi; ++ell) {
        const cuDoubleComplex* d_m_ell = d_msg_batch + (size_t)ell * (size_t)n2_;
        cuDoubleComplex* d_c_ell = d_xy_coeff + (size_t)ell * n2;
        encoder.idft2(d_m_ell, d_c_ell);
    }

    // 2) W-IDFT in complex domain (eval -> coeff)
    int threads = 256;
    int blocks  = (int)(((size_t)phi * n2 + threads - 1) / threads);
    w_idft_kernel<<<blocks, threads, 0, stream>>>(
        d_xy_coeff, d_w_coeff, d_wdft_inv, (int)n2, phi
    );

    // 3) scale+round then RNS split in W-coeff domain
    blocks = (int)((coeff_words + threads - 1) / threads);
    quantize_coeff_to_rns_kernel<<<blocks, threads, 0, stream>>>(
        d_w_coeff, d_coeff_re, d_coeff_im, (int)n2, limbs, phi, SCALING_FACTOR
    );

    // 4) explicit W-NTT to output eval layout
    wntt_forward_matrix(d_coeff_re, d_eval_re, n_, limbs, phi, stream);
    wntt_forward_matrix(d_coeff_im, d_eval_im, n_, limbs, phi, stream);
    blocks = (int)((coeff_words + threads - 1) / threads);
    poly_to_matrix_layout_kernel<<<blocks, threads, 0, stream>>>(d_eval_re, d_eval_re_mat, n_, limbs, phi);
    poly_to_matrix_layout_kernel<<<blocks, threads, 0, stream>>>(d_eval_im, d_eval_im_mat, n_, limbs, phi);
    cudaMemcpyAsync(d_out_re, d_eval_re_mat, coeff_words * sizeof(uint64_t), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(d_out_im, d_eval_im_mat, coeff_words * sizeof(uint64_t), cudaMemcpyDeviceToDevice, stream);

    cudaFree(d_xy_coeff);
    cudaFree(d_w_coeff);
    cudaFree(d_coeff_re);
    cudaFree(d_coeff_im);
    cudaFree(d_eval_re);
    cudaFree(d_eval_im);
    cudaFree(d_eval_re_mat);
    cudaFree(d_eval_im_mat);
}

void BatchedEncoder::unpack_eval_p17(
    const uint64_t* d_in_re,
    const uint64_t* d_in_im,
    uint64_t* d_eval_re,
    uint64_t* d_eval_im,
    cudaStream_t stream
) {
    const size_t total_words = (size_t)BATCH_SIZE * (size_t)RNS_NUM_LIMBS * (size_t)n2_;
    int threads = 256;
    int blocks  = (int)((total_words + threads - 1) / threads);
    copy_w_crt_kernel<<<blocks, threads, 0, stream>>>(
        d_in_re, d_in_im, d_eval_re, d_eval_im, total_words
    );
}

// ===================== Wrapper Implementation (Fix for Root Cause) =====================
// These function calls are used by HE.cu to invoke the kernels defined in this translation unit
void gpu_pack_w_phi16(const uint64_t* in_re, const uint64_t* in_im, 
                      uint64_t* out_re, uint64_t* out_im, 
                      int n2, cudaStream_t stream) {
    const size_t total_words = (size_t)BATCH_SIZE * (size_t)RNS_NUM_LIMBS * (size_t)n2;
    int threads = 256;
    int blocks = (int)((total_words + threads - 1) / threads);
    copy_w_crt_kernel<<<blocks, threads, 0, stream>>>(
        in_re, in_im, out_re, out_im, total_words
    );
}

void gpu_eval_w_phi16(const uint64_t* in_re, const uint64_t* in_im, 
                      uint64_t* out_re, uint64_t* out_im, 
                      int n2, cudaStream_t stream) {
    const size_t total_words = (size_t)BATCH_SIZE * (size_t)RNS_NUM_LIMBS * (size_t)n2;
    int threads = 256;
    int blocks = (int)((total_words + threads - 1) / threads);
    copy_w_crt_kernel<<<blocks, threads, 0, stream>>>(
        in_re, in_im, out_re, out_im, total_words
    );
}

void BatchedEncoder::init_tables_p17() {
    cudaMemcpyToSymbol(d_q, RNS_MODULI, sizeof(RNS_MODULI));
    if (wdft_inited) return;
    const int phi = BATCH_SIZE;
    const double p = (double)BATCH_PRIME_P;
    const double two_pi = 6.283185307179586476925286766559;

    std::vector<uint16_t> exp(phi);
    int idx = 0;
    for (int a = 1; a <= 2; ++a) {
        for (int b = 1; b <= 256; ++b) {
            exp[idx++] = (uint16_t)((a * 257 + b * 3) % BATCH_PRIME_P);
        }
    }

    std::vector<std::complex<double>> v((size_t)phi * (size_t)phi);
    for (int w = 0; w < phi; ++w) {
        double ang = two_pi * (double)exp[w] / p;
        std::complex<double> root(std::cos(ang), std::sin(ang));
        std::complex<double> cur(1.0, 0.0);
        for (int r = 0; r < phi; ++r) {
            v[(size_t)w * (size_t)phi + (size_t)r] = cur;
            cur *= root;
        }
    }
    // Gauss-Jordan inverse in complex
    std::vector<std::complex<double>> a = v;
    std::vector<std::complex<double>> inv((size_t)phi * (size_t)phi, std::complex<double>(0.0, 0.0));
    for (int i = 0; i < phi; ++i) inv[(size_t)i * (size_t)phi + (size_t)i] = {1.0, 0.0};
    for (int i = 0; i < phi; ++i) {
        int pivot = i;
        double best = std::abs(a[(size_t)i * (size_t)phi + (size_t)i]);
        for (int r = i + 1; r < phi; ++r) {
            double cand = std::abs(a[(size_t)r * (size_t)phi + (size_t)i]);
            if (cand > best) { best = cand; pivot = r; }
        }
        if (best < 1e-18) throw std::runtime_error("W-IDFT matrix singular");
        if (pivot != i) {
            for (int c = 0; c < phi; ++c) {
                std::swap(a[(size_t)i * (size_t)phi + (size_t)c], a[(size_t)pivot * (size_t)phi + (size_t)c]);
                std::swap(inv[(size_t)i * (size_t)phi + (size_t)c], inv[(size_t)pivot * (size_t)phi + (size_t)c]);
            }
        }
        std::complex<double> pv = a[(size_t)i * (size_t)phi + (size_t)i];
        for (int c = 0; c < phi; ++c) {
            a[(size_t)i * (size_t)phi + (size_t)c] /= pv;
            inv[(size_t)i * (size_t)phi + (size_t)c] /= pv;
        }
        for (int r = 0; r < phi; ++r) {
            if (r == i) continue;
            std::complex<double> f = a[(size_t)r * (size_t)phi + (size_t)i];
            if (std::abs(f) < 1e-18) continue;
            for (int c = 0; c < phi; ++c) {
                a[(size_t)r * (size_t)phi + (size_t)c] -= f * a[(size_t)i * (size_t)phi + (size_t)c];
                inv[(size_t)r * (size_t)phi + (size_t)c] -= f * inv[(size_t)i * (size_t)phi + (size_t)c];
            }
        }
    }

    std::vector<cuDoubleComplex> h_inv((size_t)phi * (size_t)phi);
    for (int w = 0; w < phi; ++w) {
        for (int r = 0; r < phi; ++r) {
            // Kernel reads inv_mat[w][r], so store V^{-1}[r][w] at [w][r].
            const auto& z = inv[(size_t)r * (size_t)phi + (size_t)w];
            h_inv[(size_t)w * (size_t)phi + (size_t)r] = make_cuDoubleComplex(z.real(), z.imag());
        }
    }
    cudaMalloc(&d_wdft_inv, h_inv.size() * sizeof(cuDoubleComplex));
    cudaMemcpy(d_wdft_inv, h_inv.data(), h_inv.size() * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    wdft_inited = true;
}

} // namespace matrix_fhe
