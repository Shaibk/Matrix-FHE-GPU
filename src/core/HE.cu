#include "core/HE.cuh"
#include "core/common.cuh"
#include "core/encoder.cuh"
#include <vector>
#include <iostream>
#include <random>
#include <stdexcept>
#include <complex>


// === Phantom Headers ===
#include "../../extern/phantom-fhe/include/context.cuh"
#include "../../extern/phantom-fhe/include/ntt.cuh"
#include "../../extern/phantom-fhe/include/host/encryptionparams.h"
#include "core/ntt_core.cuh"
#include "../../include/core/config.h"

namespace matrix_fhe {

// 全局单例 Context
static PhantomContext* g_phantom_ctx = nullptr;
static PhantomContext* g_phantom_xy_ctx = nullptr;
static cuDoubleComplex* d_gl_V_cx = nullptr;
static cuDoubleComplex* d_gl_V_cx_T = nullptr;
static constexpr bool kDbgUsePhantomXY = true;
static constexpr bool kDbgZeroNoise = false;
static Encoder* g_decoder_init = nullptr;
static constexpr int HE_CRT_BIGINT_LIMBS = 7;

// 常量显存
__device__ __constant__ uint64_t d_he_moduli[matrix_fhe::RNS_NUM_LIMBS];
extern __device__ __constant__ uint64_t d_rns_moduli[RNS_NUM_LIMBS];
extern __device__ __constant__ uint64_t d_crt_M[RNS_NUM_LIMBS * HE_CRT_BIGINT_LIMBS];
extern __device__ __constant__ uint64_t d_crt_Q[HE_CRT_BIGINT_LIMBS];
extern __device__ __constant__ uint64_t d_crt_Q_half[HE_CRT_BIGINT_LIMBS];
extern __device__ __constant__ uint64_t d_crt_inv[RNS_NUM_LIMBS];

// W-CRT tables (Phi_p evaluation), layout: [limb][w_out][w_coeff]
static uint64_t* d_wntt_powers = nullptr;
static uint64_t* d_wntt_inv_powers = nullptr;
static bool wntt_inited = false;
static cuDoubleComplex* d_wdft_powers = nullptr;
static cuDoubleComplex* d_wdft_inv_powers = nullptr;
static bool wdft_inited = false;

// Forward declarations
__global__ void wntt_forward_matrix_kernel(const uint64_t* in, uint64_t* out, const uint64_t* powers,
                                           int n, int limbs, int phi);
__global__ void wntt_inverse_matrix_kernel(const uint64_t* in_eval, uint64_t* out_coeff,
                                           const uint64_t* inv_powers,
                                           int n, int limbs, int phi);
__global__ void wntt_forward_centered_kernel(const int64_t* in_coeff_centered,
                                             int64_t* out_eval_centered,
                                             const uint64_t* powers,
                                             int n, int phi);
__global__ void wntt_inverse_centered_kernel(const int64_t* in_eval_centered,
                                             int64_t* out_coeff_centered,
                                             const uint64_t* inv_powers,
                                             int n, int phi);
__global__ void wdft_forward_pair_kernel(const int64_t* in_re_coeff, const int64_t* in_im_coeff,
                                         double* out_re_eval, double* out_im_eval,
                                         const cuDoubleComplex* powers, int n, int phi);
__global__ void wdft_forward_complex_kernel(const cuDoubleComplex* in_coeff,
                                            cuDoubleComplex* out_eval,
                                            const cuDoubleComplex* powers, int n, int phi);
__global__ void wdft_inverse_pair_kernel(const double* in_re_eval, const double* in_im_eval,
                                         double* out_re_coeff, double* out_im_coeff,
                                         const cuDoubleComplex* inv_powers, int n, int phi);
__global__ void count_nonzero_i64_kernel(const int64_t* in, size_t total, unsigned long long* out_nonzero);

// Fixed exp[] order: a in Z_3^* outer, b in Z_257^* inner
static constexpr uint16_t k_wntt_exp[BATCH_SIZE] = {
    260, 263, 266, 269, 272, 275, 278, 281, 284, 287, 290, 293, 296, 299, 302, 305,
    308, 311, 314, 317, 320, 323, 326, 329, 332, 335, 338, 341, 344, 347, 350, 353,
    356, 359, 362, 365, 368, 371, 374, 377, 380, 383, 386, 389, 392, 395, 398, 401,
    404, 407, 410, 413, 416, 419, 422, 425, 428, 431, 434, 437, 440, 443, 446, 449,
    452, 455, 458, 461, 464, 467, 470, 473, 476, 479, 482, 485, 488, 491, 494, 497,
    500, 503, 506, 509, 512, 515, 518, 521, 524, 527, 530, 533, 536, 539, 542, 545,
    548, 551, 554, 557, 560, 563, 566, 569, 572, 575, 578, 581, 584, 587, 590, 593,
    596, 599, 602, 605, 608, 611, 614, 617, 620, 623, 626, 629, 632, 635, 638, 641,
    644, 647, 650, 653, 656, 659, 662, 665, 668, 671, 674, 677, 680, 683, 686, 689,
    692, 695, 698, 701, 704, 707, 710, 713, 716, 719, 722, 725, 728, 731, 734, 737,
    740, 743, 746, 749, 752, 755, 758, 761, 764, 767, 770, 2, 5, 8, 11, 14,
    17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 50, 53, 56, 59, 62,
    65, 68, 71, 74, 77, 80, 83, 86, 89, 92, 95, 98, 101, 104, 107, 110,
    113, 116, 119, 122, 125, 128, 131, 134, 137, 140, 143, 146, 149, 152, 155, 158,
    161, 164, 167, 170, 173, 176, 179, 182, 185, 188, 191, 194, 197, 200, 203, 206,
    209, 212, 215, 218, 221, 224, 227, 230, 233, 236, 239, 242, 245, 248, 251, 254,
    517, 520, 523, 526, 529, 532, 535, 538, 541, 544, 547, 550, 553, 556, 559, 562,
    565, 568, 571, 574, 577, 580, 583, 586, 589, 592, 595, 598, 601, 604, 607, 610,
    613, 616, 619, 622, 625, 628, 631, 634, 637, 640, 643, 646, 649, 652, 655, 658,
    661, 664, 667, 670, 673, 676, 679, 682, 685, 688, 691, 694, 697, 700, 703, 706,
    709, 712, 715, 718, 721, 724, 727, 730, 733, 736, 739, 742, 745, 748, 751, 754,
    757, 760, 763, 766, 769, 1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31,
    34, 37, 40, 43, 46, 49, 52, 55, 58, 61, 64, 67, 70, 73, 76, 79,
    82, 85, 88, 91, 94, 97, 100, 103, 106, 109, 112, 115, 118, 121, 124, 127,
    130, 133, 136, 139, 142, 145, 148, 151, 154, 157, 160, 163, 166, 169, 172, 175,
    178, 181, 184, 187, 190, 193, 196, 199, 202, 205, 208, 211, 214, 217, 220, 223,
    226, 229, 232, 235, 238, 241, 244, 247, 250, 253, 256, 259, 262, 265, 268, 271,
    274, 277, 280, 283, 286, 289, 292, 295, 298, 301, 304, 307, 310, 313, 316, 319,
    322, 325, 328, 331, 334, 337, 340, 343, 346, 349, 352, 355, 358, 361, 364, 367,
    370, 373, 376, 379, 382, 385, 388, 391, 394, 397, 400, 403, 406, 409, 412, 415,
    418, 421, 424, 427, 430, 433, 436, 439, 442, 445, 448, 451, 454, 457, 460, 463,
    466, 469, 472, 475, 478, 481, 484, 487, 490, 493, 496, 499, 502, 505, 508, 511,
};

// Host helpers for W-CRT
static uint64_t h_pow_mod_u128(uint64_t base, uint64_t exp, uint64_t mod) {
    uint64_t res = 1;
    base %= mod;
    while (exp > 0) {
        if (exp & 1) res = (unsigned __int128)res * base % mod;
        base = (unsigned __int128)base * base % mod;
        exp >>= 1;
    }
    return res;
}

static uint64_t h_find_eta(uint64_t q) {
    const uint64_t p = BATCH_PRIME_P; // 771
    const uint64_t f1 = 3;
    const uint64_t f2 = 257;
    const uint64_t exp = (q - 1) / p;
    for (uint64_t g = 2; g < q; ++g) {
        uint64_t eta = h_pow_mod_u128(g, exp, q);
        if (eta == 1) continue;
        if (h_pow_mod_u128(eta, p, q) != 1) continue;
        if (h_pow_mod_u128(eta, p / f1, q) == 1) continue;
        if (h_pow_mod_u128(eta, p / f2, q) == 1) continue;
        return eta;
    }
    throw std::runtime_error("Failed to find eta for W-CRT");
}

static std::vector<uint64_t> matrix_inverse_mod(const std::vector<uint64_t>& matrix, int dim, uint64_t mod) {
    std::vector<uint64_t> a = matrix;
    std::vector<uint64_t> inv((size_t)dim * (size_t)dim, 0);
    for (int i = 0; i < dim; ++i) {
        inv[(size_t)i * (size_t)dim + (size_t)i] = 1;
    }

    for (int i = 0; i < dim; ++i) {
        int pivot = i;
        while (pivot < dim && a[(size_t)pivot * (size_t)dim + (size_t)i] == 0) {
            ++pivot;
        }
        if (pivot == dim) {
            throw std::runtime_error("W-CRT matrix is singular");
        }
        if (pivot != i) {
            for (int j = 0; j < dim; ++j) {
                std::swap(a[(size_t)i * (size_t)dim + (size_t)j],
                          a[(size_t)pivot * (size_t)dim + (size_t)j]);
                std::swap(inv[(size_t)i * (size_t)dim + (size_t)j],
                          inv[(size_t)pivot * (size_t)dim + (size_t)j]);
            }
        }

        uint64_t pivot_val = a[(size_t)i * (size_t)dim + (size_t)i];
        uint64_t pivot_inv = h_pow_mod_u128(pivot_val, mod - 2, mod);
        for (int j = 0; j < dim; ++j) {
            a[(size_t)i * (size_t)dim + (size_t)j] =
                (unsigned __int128)a[(size_t)i * (size_t)dim + (size_t)j] * pivot_inv % mod;
            inv[(size_t)i * (size_t)dim + (size_t)j] =
                (unsigned __int128)inv[(size_t)i * (size_t)dim + (size_t)j] * pivot_inv % mod;
        }

        for (int r = 0; r < dim; ++r) {
            if (r == i) continue;
            uint64_t factor = a[(size_t)r * (size_t)dim + (size_t)i];
            if (factor == 0) continue;
            for (int c = 0; c < dim; ++c) {
                uint64_t sub_a =
                    (unsigned __int128)factor * a[(size_t)i * (size_t)dim + (size_t)c] % mod;
                uint64_t sub_inv =
                    (unsigned __int128)factor * inv[(size_t)i * (size_t)dim + (size_t)c] % mod;
                uint64_t& a_rc = a[(size_t)r * (size_t)dim + (size_t)c];
                uint64_t& inv_rc = inv[(size_t)r * (size_t)dim + (size_t)c];
                a_rc = (a_rc >= sub_a) ? (a_rc - sub_a) : (a_rc + mod - sub_a);
                inv_rc = (inv_rc >= sub_inv) ? (inv_rc - sub_inv) : (inv_rc + mod - sub_inv);
            }
        }
    }
    return inv;
}

static std::vector<std::complex<double>> matrix_inverse_complex(
    const std::vector<std::complex<double>>& matrix, int dim
) {
    std::vector<std::complex<double>> a = matrix;
    std::vector<std::complex<double>> inv((size_t)dim * (size_t)dim, std::complex<double>(0.0, 0.0));
    for (int i = 0; i < dim; ++i) {
        inv[(size_t)i * (size_t)dim + (size_t)i] = std::complex<double>(1.0, 0.0);
    }

    for (int i = 0; i < dim; ++i) {
        int pivot = i;
        double best = std::abs(a[(size_t)i * (size_t)dim + (size_t)i]);
        for (int r = i + 1; r < dim; ++r) {
            double cand = std::abs(a[(size_t)r * (size_t)dim + (size_t)i]);
            if (cand > best) {
                best = cand;
                pivot = r;
            }
        }
        if (best < 1e-18) {
            throw std::runtime_error("W-DFT matrix is singular");
        }
        if (pivot != i) {
            for (int j = 0; j < dim; ++j) {
                std::swap(a[(size_t)i * (size_t)dim + (size_t)j],
                          a[(size_t)pivot * (size_t)dim + (size_t)j]);
                std::swap(inv[(size_t)i * (size_t)dim + (size_t)j],
                          inv[(size_t)pivot * (size_t)dim + (size_t)j]);
            }
        }

        std::complex<double> pivot_val = a[(size_t)i * (size_t)dim + (size_t)i];
        for (int j = 0; j < dim; ++j) {
            a[(size_t)i * (size_t)dim + (size_t)j] /= pivot_val;
            inv[(size_t)i * (size_t)dim + (size_t)j] /= pivot_val;
        }

        for (int r = 0; r < dim; ++r) {
            if (r == i) continue;
            std::complex<double> factor = a[(size_t)r * (size_t)dim + (size_t)i];
            if (std::abs(factor) < 1e-18) continue;
            for (int c = 0; c < dim; ++c) {
                a[(size_t)r * (size_t)dim + (size_t)c] -= factor * a[(size_t)i * (size_t)dim + (size_t)c];
                inv[(size_t)r * (size_t)dim + (size_t)c] -= factor * inv[(size_t)i * (size_t)dim + (size_t)c];
            }
        }
    }
    return inv;
}

static void init_wntt_tables() {
    if (wntt_inited) return;
    const int phi = BATCH_SIZE;       // 512
    std::vector<uint64_t> h_wntt((size_t)RNS_NUM_LIMBS * (size_t)phi * (size_t)phi, 0);
    std::vector<uint64_t> h_wntt_inv((size_t)RNS_NUM_LIMBS * (size_t)phi * (size_t)phi, 0);
    for (int l = 0; l < RNS_NUM_LIMBS; ++l) {
        uint64_t q = RNS_MODULI[l];
        uint64_t eta = h_find_eta(q);

        std::vector<uint64_t> v((size_t)phi * (size_t)phi, 0);
        for (int w = 0; w < phi; ++w) {
            uint64_t root = h_pow_mod_u128(eta, k_wntt_exp[w], q);
            uint64_t cur = 1;
            for (int r = 0; r < phi; ++r) {
                v[(size_t)w * (size_t)phi + (size_t)r] = cur;
                cur = (unsigned __int128)cur * root % q;
            }
        }

        std::vector<uint64_t> v_inv = matrix_inverse_mod(v, phi, q);

        for (int w = 0; w < phi; ++w) {
            size_t base = ((size_t)l * (size_t)phi + (size_t)w) * (size_t)phi;
            for (int r = 0; r < phi; ++r) {
                h_wntt[base + (size_t)r] = v[(size_t)w * (size_t)phi + (size_t)r];
                h_wntt_inv[base + (size_t)r] = v_inv[(size_t)r * (size_t)phi + (size_t)w];
            }
        }
    }

    size_t bytes = h_wntt.size() * sizeof(uint64_t);
    cudaMalloc(&d_wntt_powers, bytes);
    cudaMemcpy(d_wntt_powers, h_wntt.data(), bytes, cudaMemcpyHostToDevice);
    cudaMalloc(&d_wntt_inv_powers, bytes);
    cudaMemcpy(d_wntt_inv_powers, h_wntt_inv.data(), bytes, cudaMemcpyHostToDevice);
    wntt_inited = true;
}

static void init_wdft_tables() {
    if (wdft_inited) return;
    const int phi = BATCH_SIZE;
    const double p = (double)BATCH_PRIME_P;
    const double two_pi = 6.283185307179586476925286766559;

    std::vector<std::complex<double>> v((size_t)phi * (size_t)phi, std::complex<double>(0.0, 0.0));
    for (int w = 0; w < phi; ++w) {
        double ang = two_pi * (double)k_wntt_exp[w] / p;
        std::complex<double> root(std::cos(ang), std::sin(ang));
        std::complex<double> cur(1.0, 0.0);
        for (int r = 0; r < phi; ++r) {
            v[(size_t)w * (size_t)phi + (size_t)r] = cur;
            cur *= root;
        }
    }

    std::vector<std::complex<double>> v_inv = matrix_inverse_complex(v, phi);
    std::vector<cuDoubleComplex> h_wdft((size_t)phi * (size_t)phi);
    std::vector<cuDoubleComplex> h_wdft_inv((size_t)phi * (size_t)phi);
    for (int w = 0; w < phi; ++w) {
        for (int r = 0; r < phi; ++r) {
            const auto& f = v[(size_t)w * (size_t)phi + (size_t)r];
            h_wdft[(size_t)w * (size_t)phi + (size_t)r] = make_cuDoubleComplex(f.real(), f.imag());
            const auto& iv = v_inv[(size_t)r * (size_t)phi + (size_t)w];
            h_wdft_inv[(size_t)w * (size_t)phi + (size_t)r] = make_cuDoubleComplex(iv.real(), iv.imag());
        }
    }

    size_t bytes = h_wdft.size() * sizeof(cuDoubleComplex);
    cudaMalloc(&d_wdft_powers, bytes);
    cudaMemcpy(d_wdft_powers, h_wdft.data(), bytes, cudaMemcpyHostToDevice);
    cudaMalloc(&d_wdft_inv_powers, bytes);
    cudaMemcpy(d_wdft_inv_powers, h_wdft_inv.data(), bytes, cudaMemcpyHostToDevice);
    wdft_inited = true;
}



// ==========================================
// 1. Initialization
// ==========================================

void init_he_backend() {
    if (g_phantom_ctx != nullptr) return;

    // NOTE: GL path bypasses PhantomContext to avoid NTT root constraints tied to HE_N.
    cudaMemcpyToSymbol(d_he_moduli, matrix_fhe::RNS_MODULI, sizeof(matrix_fhe::RNS_MODULI));
    init_ntt_tables_manual(MATRIX_N, RNS_NUM_LIMBS);
    init_wntt_tables();
    init_wdft_tables();

    if (g_phantom_xy_ctx == nullptr) {
        phantom::EncryptionParameters parms(phantom::scheme_type::ckks);
        parms.set_poly_modulus_degree(MATRIX_N);
        std::vector<phantom::arith::Modulus> mods;
        for (int i = 0; i < RNS_NUM_LIMBS; ++i) {
            mods.emplace_back(RNS_MODULI[i]);
        }
        parms.set_coeff_modulus(mods);
        g_phantom_xy_ctx = new PhantomContext(parms);
    }
    if (d_gl_V_cx == nullptr || d_gl_V_cx_T == nullptr) {
        const double PI = 3.141592653589793;
        std::vector<cuDoubleComplex> h_V(MATRIX_N * MATRIX_N);
        std::vector<cuDoubleComplex> h_VT(MATRIX_N * MATRIX_N);
        for (int j = 0; j < MATRIX_N; ++j) {
            uint64_t exp = 1, b5 = 5;
            int p = j;
            while (p > 0) {
                if (p & 1) exp = (exp * b5) % (4 * MATRIX_N);
                b5 = (b5 * b5) % (4 * MATRIX_N);
                p >>= 1;
            }
            double ang = 2.0 * PI * (double)exp / (4.0 * MATRIX_N);
            cuDoubleComplex z = {cos(ang), sin(ang)};
            cuDoubleComplex c = {1, 0};
            for (int k = 0; k < MATRIX_N; ++k) {
                h_V[j * MATRIX_N + k] = c;
                c = cuCmul(c, z);
            }
        }
        for (int r = 0; r < MATRIX_N; ++r) {
            for (int c = 0; c < MATRIX_N; ++c) {
                h_VT[c * MATRIX_N + r] = h_V[r * MATRIX_N + c];
            }
        }
        cudaMalloc(&d_gl_V_cx, MATRIX_N * MATRIX_N * sizeof(cuDoubleComplex));
        cudaMalloc(&d_gl_V_cx_T, MATRIX_N * MATRIX_N * sizeof(cuDoubleComplex));
        cudaMemcpy(d_gl_V_cx, h_V.data(), MATRIX_N * MATRIX_N * sizeof(cuDoubleComplex),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_gl_V_cx_T, h_VT.data(), MATRIX_N * MATRIX_N * sizeof(cuDoubleComplex),
                   cudaMemcpyHostToDevice);
    }
    if (g_decoder_init == nullptr) {
        g_decoder_init = new Encoder(MATRIX_N);
    }
    if (d_gl_V_cx == nullptr || d_gl_V_cx_T == nullptr) {
        const double PI = 3.141592653589793;
        std::vector<cuDoubleComplex> h_V(MATRIX_N * MATRIX_N);
        std::vector<cuDoubleComplex> h_VT(MATRIX_N * MATRIX_N);
        for (int j = 0; j < MATRIX_N; ++j) {
            uint64_t exp = 1, b5 = 5;
            int p = j;
            while (p > 0) {
                if (p & 1) exp = (exp * b5) % (4 * MATRIX_N);
                b5 = (b5 * b5) % (4 * MATRIX_N);
                p >>= 1;
            }
            double ang = 2.0 * PI * (double)exp / (4.0 * MATRIX_N);
            cuDoubleComplex z = {cos(ang), sin(ang)};
            cuDoubleComplex c = {1, 0};
            for (int k = 0; k < MATRIX_N; ++k) {
                h_V[j * MATRIX_N + k] = c;
                c = cuCmul(c, z);
            }
        }
        for (int r = 0; r < MATRIX_N; ++r) {
            for (int c = 0; c < MATRIX_N; ++c) {
                h_VT[c * MATRIX_N + r] = h_V[r * MATRIX_N + c];
            }
        }
        cudaMalloc(&d_gl_V_cx, MATRIX_N * MATRIX_N * sizeof(cuDoubleComplex));
        cudaMalloc(&d_gl_V_cx_T, MATRIX_N * MATRIX_N * sizeof(cuDoubleComplex));
        cudaMemcpy(d_gl_V_cx, h_V.data(), MATRIX_N * MATRIX_N * sizeof(cuDoubleComplex),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_gl_V_cx_T, h_VT.data(), MATRIX_N * MATRIX_N * sizeof(cuDoubleComplex),
                   cudaMemcpyHostToDevice);
    }

    std::cout << ">>> GL HE Backend Initialized (PhantomContext bypassed)\n";
    std::cout << ">>> Matrix Config: n=" << MATRIX_N << ", PHI=" << BATCH_SIZE
              << ", logical_poly_size=" << MATRIX_N * BATCH_SIZE << "\n";
}

void copy_device_moduli(uint64_t* h_out, int count) {
    if (count < RNS_NUM_LIMBS) {
        std::cerr << "Error: copy_device_moduli count too small\n";
        exit(1);
    }
    cudaError_t e = cudaMemcpyFromSymbol(h_out, d_he_moduli,
                                         sizeof(uint64_t) * RNS_NUM_LIMBS);
    if (e != cudaSuccess) {
        std::cerr << "[CUDA ERROR] MemcpyFromSymbol d_he_moduli: "
                  << cudaGetErrorString(e) << "\n";
        exit(1);
    }
}

const DNTTTable& get_ntt_table() {
    std::cerr << "Error: PhantomContext is disabled in GL path; get_ntt_table() unavailable.\n";
    exit(1);
}

const DNTTTable& get_xy_ntt_table() {
    if (!g_phantom_xy_ctx) {
        std::cerr << "Error: XY NTT backend not initialized. Call init_he_backend() first.\n";
        exit(1);
    }
    return g_phantom_xy_ctx->gpu_rns_tables();
}

void wntt_forward_matrix(const uint64_t* in, uint64_t* out, int n, int limbs, int phi, cudaStream_t stream) {
    size_t total = (size_t)phi * (size_t)limbs * (size_t)n * (size_t)n;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    wntt_forward_matrix_kernel<<<blocks, threads, 0, stream>>>(in, out, d_wntt_powers, n, limbs, phi);
}

void wntt_inverse_matrix(const uint64_t* in_eval, uint64_t* out_coeff,
                         int n, int limbs, int phi, cudaStream_t stream) {
    size_t total = (size_t)phi * (size_t)limbs * (size_t)n * (size_t)n;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    wntt_inverse_matrix_kernel<<<blocks, threads, 0, stream>>>(in_eval, out_coeff,
                                                               d_wntt_inv_powers,
                                                               n, limbs, phi);
}

void wntt_forward_centered(const int64_t* in_coeff_centered, int64_t* out_eval_centered,
                           int n, int phi, cudaStream_t stream) {
    size_t total = (size_t)phi * (size_t)n * (size_t)n;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    wntt_forward_centered_kernel<<<blocks, threads, 0, stream>>>(
        in_coeff_centered, out_eval_centered, d_wntt_powers, n, phi);
}

void wntt_inverse_centered(const int64_t* in_eval_centered, int64_t* out_coeff_centered,
                           int n, int phi, cudaStream_t stream) {
    size_t total = (size_t)phi * (size_t)n * (size_t)n;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    wntt_inverse_centered_kernel<<<blocks, threads, 0, stream>>>(
        in_eval_centered, out_coeff_centered, d_wntt_inv_powers, n, phi);
}

void wdft_forward_centered_pair(const int64_t* in_re_centered, const int64_t* in_im_centered,
                                double* out_re_eval, double* out_im_eval,
                                int n, int phi, cudaStream_t stream) {
    if (!wdft_inited) init_wdft_tables();
    size_t total = (size_t)phi * (size_t)n * (size_t)n;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    wdft_forward_pair_kernel<<<blocks, threads, 0, stream>>>(
        in_re_centered, in_im_centered, out_re_eval, out_im_eval, d_wdft_powers, n, phi);
}

static void wdft_forward_complex(const cuDoubleComplex* in_coeff, cuDoubleComplex* out_eval,
                                 int n, int phi, cudaStream_t stream) {
    if (!wdft_inited) init_wdft_tables();
    size_t total = (size_t)phi * (size_t)n * (size_t)n;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    wdft_forward_complex_kernel<<<blocks, threads, 0, stream>>>(
        in_coeff, out_eval, d_wdft_powers, n, phi);
}

void wdft_inverse_pair(const double* in_re_eval, const double* in_im_eval,
                       double* out_re_coeff, double* out_im_coeff,
                       int n, int phi, cudaStream_t stream) {
    if (!wdft_inited) init_wdft_tables();
    size_t total = (size_t)phi * (size_t)n * (size_t)n;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    wdft_inverse_pair_kernel<<<blocks, threads, 0, stream>>>(
        in_re_eval, in_im_eval, out_re_coeff, out_im_coeff, d_wdft_inv_powers, n, phi);
}

// ==========================================
// 2. Custom Kernels (Arithmetic)
// ==========================================

// --- Encrypt Kernel (Poly-Major Layout, X-NTT domain) ---
__global__ void pointwise_mul_s_kernel(
    const uint64_t* a_ntt,
    const uint64_t* s_ntt,
    uint64_t* t_ntt,
    size_t total_len,
    int limbs,
    int n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_len) return;

    int single_poly_len = limbs * n;
    int offset_in_poly = (int)(idx % (size_t)single_poly_len);
    int l = offset_in_poly / n;
    int coeff = offset_in_poly - l * n;
    int poly = (int)(idx / (size_t)single_poly_len);
    int w = poly / n; // poly = w*n + y

    uint64_t q = d_he_moduli[l];
    size_t s_off = ((size_t)w * (size_t)limbs + (size_t)l) * (size_t)n + (size_t)coeff;

    t_ntt[idx] = mul_mod(a_ntt[idx], s_ntt[s_off], q);
}


// --- Decrypt Kernel (Poly-Major Layout) ---
__global__ void combine_b_kernel(const uint64_t* m, const uint64_t* t, const uint64_t* e,
                                 uint64_t* b, size_t total_len, int limbs, int n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_len) return;

    int single_poly_len = limbs * n;
    int offset_in_poly = (int)(idx % (size_t)single_poly_len);
    int l = offset_in_poly / n;
    
    uint64_t q = d_he_moduli[l];
    uint64_t tmp = sub_mod(m[idx], t[idx], q);
    b[idx] = add_mod(tmp, e[idx], q);
}

__global__ void add_poly_kernel(const uint64_t* a, const uint64_t* b,
                                uint64_t* out, size_t total_len, int limbs, int n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_len) return;

    int single_poly_len = limbs * n;
    int offset_in_poly = (int)(idx % (size_t)single_poly_len);
    int l = offset_in_poly / n;
    uint64_t q = d_he_moduli[l];

    out[idx] = add_mod(a[idx], b[idx], q);
}

// --- Noise Generation Kernels ---
// W-coeff layout: [W][Limb][Y][X] (flat: [W][Limb][n*n])
__global__ void uniform_random_kernel(uint64_t* data, size_t total_len, int limbs, int n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_len) return;

    size_t n2 = (size_t)n * (size_t)n;
    size_t single_w_len = (size_t)limbs * n2;
    size_t offset_in_w = idx % single_w_len;
    int limb_idx = (int)(offset_in_w / n2);

    uint64_t q = d_he_moduli[limb_idx];

    uint64_t seed = 123456789ULL + idx;
    seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
    data[idx] = seed % q;
}


__device__ __forceinline__ uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

// W-coeff layout: [W][Limb][Y][X] (flat: [W][Limb][n*n])
// Discrete Gaussian (sigma=3.2) per coefficient
__global__ void gaussian_noise_kernel(uint64_t* data, size_t total_len, int limbs, int n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_len) return;

    size_t n2 = (size_t)n * (size_t)n;
    size_t single_w_len = (size_t)limbs * n2;
    size_t w_idx = idx / single_w_len;
    size_t offset_in_w = idx % single_w_len;
    int limb_idx = (int)(offset_in_w / n2);
    size_t pos_in_w = offset_in_w % n2;

    uint64_t q = d_he_moduli[limb_idx];

    // Sample one centered integer noise per logical coefficient [w][y][x],
    // then map the same integer into each RNS limb.
    uint64_t coeff_id = (uint64_t)(w_idx * n2 + pos_in_w);
    uint64_t seed = (0xD6E8FEB86659FD93ULL ^ coeff_id);
    uint64_t r1 = splitmix64(seed);
    uint64_t r2 = splitmix64(r1);

    // Box-Muller: U(0,1] from 53-bit mantissa
    const double inv53 = 1.0 / 9007199254740992.0; // 2^53
    double u1 = ((double)(r1 >> 11) + 1.0) * inv53;
    double u2 = ((double)(r2 >> 11) + 1.0) * inv53;

    const double sigma = 3.2;
    double mag = sigma * sqrt(-2.0 * log(u1));
    double z = mag * cos(6.283185307179586 * u2); // 2*pi

    int64_t noise = llround(z);
    uint64_t out;
    if (noise >= 0) {
        out = (uint64_t)noise;
    } else {
        out = q - (uint64_t)(-noise);
    }
    data[idx] = out;
}



__global__ void add_ct_kernel(const uint64_t* b1, const uint64_t* a1,
                              const uint64_t* b2, const uint64_t* a2,
                              uint64_t* b_out, uint64_t* a_out, size_t total_len, int limbs, int n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_len) return;

    int single_poly_len = limbs * n;
    int offset_in_poly = idx % single_poly_len;
    int l = offset_in_poly / n;
    uint64_t q = d_he_moduli[l];

    b_out[idx] = add_mod(b1[idx], b2[idx], q);
    a_out[idx] = add_mod(a1[idx], a2[idx], q);
}


__global__ void mul_tensor_kernel(const uint64_t* b1, const uint64_t* a1,
                                  const uint64_t* b2, const uint64_t* a2,
                                  uint64_t* d0, uint64_t* d1, uint64_t* d2,
                                  size_t total_len, int limbs, int n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_len) return;

    int single_poly_len = limbs * n;
    int offset_in_poly = idx % single_poly_len;
    int l = offset_in_poly / n;
    uint64_t q = d_he_moduli[l];

    uint64_t vb1 = b1[idx]; uint64_t va1 = a1[idx];
    uint64_t vb2 = b2[idx]; uint64_t va2 = a2[idx];

    d0[idx] = mul_mod(vb1, vb2, q);
    
    uint64_t term1 = mul_mod(vb1, va2, q);
    uint64_t term2 = mul_mod(va1, vb2, q);
    d1[idx] = add_mod(term1, term2, q);
    
    d2[idx] = mul_mod(va1, va2, q);
}

// ==========================================
// 3. Host API Implementation
// ==========================================

void allocate_ciphertext(RLWECiphertext& ct, int limbs) {
    ct.num_limbs = limbs;
    ct.is_ntt = false;
    size_t total_coeffs = (size_t)BATCH_SIZE * (size_t)MATRIX_N * (size_t)limbs * (size_t)MATRIX_N;
    size_t size = 2 * total_coeffs * sizeof(uint64_t);
    cudaMalloc(&ct.data, size);
    cudaMemset(ct.data, 0, size);
}

void free_ciphertext(RLWECiphertext& ct) {
    if(ct.data) {
        cudaFree(ct.data);
        ct.data = nullptr;
    }
}
__global__ void ternary_secret_kernel(uint64_t* s, size_t total_len, int limbs, int n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_len) return;

    const int single_poly_len = limbs * n;

    // [poly][limb][coeff]
    int offset_in_poly = (int)(idx % (size_t)single_poly_len);
    int limb = offset_in_poly / n;
    int coeff = offset_in_poly - limb * n;

    // 让同一 coeff 在所有 limb 上一致：seed 不包含 limb
    // 同时为了让不同 poly 不同，加入 poly_idx
    int poly = (int)(idx / (size_t)single_poly_len);

    uint64_t t = (uint64_t)poly * 1315423911ULL + (uint64_t)coeff * 2654435761ULL;
    // 取三值：0,1,2 -> 映射到 0,+1,-1
    int r = (int)((t * 11400714819323198485ULL) % 3ULL);

    uint64_t q = d_he_moduli[limb];
    if (r == 0) s[idx] = 0;
    else if (r == 1) s[idx] = 1;
    else s[idx] = q - 1; // -1 mod q
}

// W-CRT forward (Phi_p evaluation), matrix layout: [W][Limb][n*n]
__global__ void wntt_forward_matrix_kernel(
    const uint64_t* in,
    uint64_t* out,
    const uint64_t* powers,
    int n, int limbs, int phi
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t n2 = (size_t)n * (size_t)n;
    size_t total = (size_t)phi * (size_t)limbs * n2;
    if (idx >= total) return;

    int x = (int)(idx % (size_t)n);
    size_t t = idx / (size_t)n;
    int y = (int)(t % (size_t)n);
    t /= (size_t)n;
    int limb = (int)(t % (size_t)limbs);
    int w_out = (int)(t / (size_t)limbs);

    uint64_t q = d_he_moduli[limb];
    uint64_t acc = 0;
    size_t pow_base = ((size_t)limb * (size_t)phi + (size_t)w_out) * (size_t)phi;
    for (int r = 0; r < phi; ++r) {
        size_t in_off = ((size_t)r * (size_t)limbs + (size_t)limb) * n2 +
                        (size_t)y * (size_t)n + (size_t)x;
        uint64_t a = in[in_off];
        uint64_t w = powers[pow_base + (size_t)r];
        acc = add_mod(acc, mul_mod(a, w, q), q);
    }
    int poly = w_out * n + y; // poly-major: [poly][limb][x]
    size_t out_off = ((size_t)poly * (size_t)limbs + (size_t)limb) * (size_t)n + (size_t)x;
    out[out_off] = acc;
}

// W-CRT inverse (eval -> coeff). Input is poly-major eval: [poly=w_out*n+y][limb][x]
// Output is matrix-major coeff: [r][limb][y][x]
__global__ void wntt_inverse_matrix_kernel(
    const uint64_t* in_eval,
    uint64_t* out_coeff,
    const uint64_t* inv_powers,
    int n, int limbs, int phi
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t n2 = (size_t)n * (size_t)n;
    size_t total = (size_t)phi * (size_t)limbs * n2;
    if (idx >= total) return;

    int x = (int)(idx % (size_t)n);
    size_t t = idx / (size_t)n;
    int y = (int)(t % (size_t)n);
    t /= (size_t)n;
    int limb = (int)(t % (size_t)limbs);
    int r = (int)(t / (size_t)limbs); // coeff index in W

    uint64_t q = d_he_moduli[limb];
    uint64_t acc = 0;
    size_t pow_base = ((size_t)limb * (size_t)phi) * (size_t)phi;
    for (int w = 0; w < phi; ++w) {
        size_t in_off = ((size_t)(w * n + y) * (size_t)limbs + (size_t)limb) * (size_t)n + (size_t)x;
        uint64_t a = in_eval[in_off];
        uint64_t wpow = inv_powers[pow_base + (size_t)w * (size_t)phi + (size_t)r];
        acc = add_mod(acc, mul_mod(a, wpow, q), q);
    }
    size_t out_off = ((size_t)r * (size_t)limbs + (size_t)limb) * n2 +
                     (size_t)y * (size_t)n + (size_t)x;
    out_coeff[out_off] = acc;
}

// Round (v / delta) in coeff domain; output in mod q
__global__ void round_by_delta_kernel(
    uint64_t* data, int n, int limbs, int phi, double delta
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t n2 = (size_t)n * (size_t)n;
    size_t total = (size_t)phi * (size_t)limbs * n2;
    if (idx >= total) return;

    int limb = (int)((idx / n2) % (size_t)limbs);
    uint64_t q = d_he_moduli[limb];

    uint64_t a = data[idx];
    int64_t centered = (a > (q >> 1)) ? (int64_t)a - (int64_t)q : (int64_t)a;
    double v = (double)centered / delta;
    int64_t r = llround(v);
    int64_t scale = (int64_t)delta;
    __int128 scaled = (r >= 0) ? (__int128)r * scale : -((__int128)(-r) * scale);
    int64_t modv = (int64_t)(scaled % (int64_t)q);
    if (modv < 0) modv += (int64_t)q;
    uint64_t out = (uint64_t)modv;
    data[idx] = out;
}

__global__ void round_centered_by_delta_kernel(
    int64_t* data, size_t total, double delta
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    data[idx] = llround((double)data[idx] / delta);
}

__global__ void centered_int_to_rns_matrix_kernel(
    const int64_t* in_centered,
    uint64_t* out_rns,
    int n, int limbs, int phi
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t n2 = (size_t)n * (size_t)n;
    size_t total = (size_t)phi * (size_t)limbs * n2;
    if (idx >= total) return;

    size_t pos = idx % n2;
    size_t t = idx / n2;
    int limb = (int)(t % (size_t)limbs);
    int w = (int)(t / (size_t)limbs);

    int64_t v = in_centered[(size_t)w * n2 + pos];
    uint64_t q = d_he_moduli[limb];
    int64_t modv = v % (int64_t)q;
    if (modv < 0) modv += (int64_t)q;
    out_rns[idx] = (uint64_t)modv;
}

__global__ void centered_pair_to_complex_kernel(
    const int64_t* in_re,
    const int64_t* in_im,
    cuDoubleComplex* out,
    int n2
) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= n2) return;
    out[idx] = make_cuDoubleComplex((double)in_re[idx], (double)in_im[idx]);
}

__global__ void real_pair_to_complex_kernel(
    const double* in_re,
    const double* in_im,
    cuDoubleComplex* out,
    int n2
) {
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= n2) return;
    out[idx] = make_cuDoubleComplex(in_re[idx], in_im[idx]);
}

__device__ __forceinline__ int he_big_cmp(const uint64_t* a, const uint64_t* b) {
    for (int i = HE_CRT_BIGINT_LIMBS - 1; i >= 0; --i) {
        if (a[i] > b[i]) return 1;
        if (a[i] < b[i]) return -1;
    }
    return 0;
}

__device__ __forceinline__ void he_big_sub_inplace(uint64_t* a, const uint64_t* b) {
    uint64_t borrow = 0;
    for (int i = 0; i < HE_CRT_BIGINT_LIMBS; ++i) {
        uint64_t bi = b[i] + borrow;
        borrow = (a[i] < bi);
        a[i] -= bi;
    }
}

__device__ __forceinline__ void he_big_sub_rev(uint64_t* out, const uint64_t* a, const uint64_t* b) {
    uint64_t borrow = 0;
    for (int i = 0; i < HE_CRT_BIGINT_LIMBS; ++i) {
        uint64_t ai = a[i];
        uint64_t bi = b[i] + borrow;
        borrow = (ai < bi);
        out[i] = ai - bi;
    }
}

__device__ __forceinline__ void he_big_add_inplace(uint64_t* a, const uint64_t* b) {
    uint64_t carry = 0;
    for (int i = 0; i < HE_CRT_BIGINT_LIMBS; ++i) {
        uint64_t old = a[i];
        a[i] = old + b[i] + carry;
        carry = (a[i] < old) || (carry && a[i] == old);
    }
}

__device__ __forceinline__ void he_big_mul_u64(const uint64_t* a, uint64_t m, uint64_t* out) {
    unsigned __int128 carry = 0;
    for (int i = 0; i < HE_CRT_BIGINT_LIMBS; ++i) {
        unsigned __int128 p = (unsigned __int128)a[i] * m + carry;
        out[i] = (uint64_t)p;
        carry = p >> 64;
    }
}

__device__ __forceinline__ int64_t he_big_to_i64_checked(const uint64_t* a, bool neg) {
    for (int i = 1; i < HE_CRT_BIGINT_LIMBS; ++i) {
        if (a[i] != 0) {
            return neg ? INT64_MIN : INT64_MAX;
        }
    }
    if (a[0] > (uint64_t)INT64_MAX) {
        return neg ? INT64_MIN : INT64_MAX;
    }
    int64_t v = (int64_t)a[0];
    return neg ? -v : v;
}

__device__ __forceinline__ double he_big_to_f64(const uint64_t* a) {
    const double two64 = 18446744073709551616.0;
    double v = 0.0;
    for (int i = HE_CRT_BIGINT_LIMBS - 1; i >= 0; --i) {
        v = v * two64 + (double)a[i];
    }
    return v;
}

__device__ __forceinline__ void he_big_divmod_u64(
    const uint64_t* a, uint64_t d, uint64_t* q, uint64_t* rem
) {
    unsigned __int128 r = 0;
    for (int i = HE_CRT_BIGINT_LIMBS - 1; i >= 0; --i) {
        unsigned __int128 cur = (r << 64) | (unsigned __int128)a[i];
        q[i] = (uint64_t)(cur / d);
        r = cur % d;
    }
    *rem = (uint64_t)r;
}

__device__ __forceinline__ void he_big_inc_inplace(uint64_t* a) {
    uint64_t carry = 1;
    for (int i = 0; i < HE_CRT_BIGINT_LIMBS && carry; ++i) {
        uint64_t old = a[i];
        a[i] = old + carry;
        carry = (a[i] < old);
    }
}

__device__ __forceinline__ void he_big_shr_bits(
    const uint64_t* in, int shift_bits, uint64_t* out
) {
    int limb_shift = shift_bits / 64;
    int bit_shift = shift_bits % 64;
    for (int i = 0; i < HE_CRT_BIGINT_LIMBS; ++i) {
        int src = i + limb_shift;
        uint64_t lo = (src < HE_CRT_BIGINT_LIMBS) ? in[src] : 0ULL;
        uint64_t hi = (bit_shift != 0 && (src + 1) < HE_CRT_BIGINT_LIMBS) ? in[src + 1] : 0ULL;
        if (bit_shift == 0) {
            out[i] = lo;
        } else {
            out[i] = (lo >> bit_shift) | (hi << (64 - bit_shift));
        }
    }
}

__global__ void round_big_centered_by_delta_kernel(
    const uint64_t* in_mag,
    const uint8_t* in_neg,
    int64_t* out,
    size_t total,
    double delta
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    uint64_t scale = (uint64_t)delta;
    if (scale == 0) {
        out[idx] = 0;
        return;
    }

    const uint64_t* mag = &in_mag[idx * HE_CRT_BIGINT_LIMBS];
    uint64_t q[HE_CRT_BIGINT_LIMBS];
    // Integer-only nearest rounding on |v|/Delta, then apply original sign.
    if ((scale & (scale - 1)) == 0) {
        int shift = 0;
        while ((scale >> shift) != 1ULL) ++shift;
        uint64_t half[HE_CRT_BIGINT_LIMBS];
        for (int i = 0; i < HE_CRT_BIGINT_LIMBS; ++i) half[i] = mag[i];
        if (shift > 0) {
            uint64_t addv[HE_CRT_BIGINT_LIMBS] = {0};
            int limb = (shift - 1) / 64;
            int bit = (shift - 1) % 64;
            addv[limb] = (1ULL << bit);
            he_big_add_inplace(half, addv);
        }
        he_big_shr_bits(half, shift, q);
    } else {
        // Fallback: exact integer nearest rounding for non power-of-two Delta.
        uint64_t rem = 0;
        he_big_divmod_u64(mag, scale, q, &rem);
        if ((unsigned __int128)rem * 2 >= (unsigned __int128)scale) {
            he_big_inc_inplace(q);
        }
    }

    out[idx] = he_big_to_i64_checked(q, in_neg[idx] != 0);
}

__global__ void compose_big_pair_to_complex_by_delta_kernel(
    const uint64_t* re_mag,
    const uint8_t* re_neg,
    const uint64_t* im_mag,
    const uint8_t* im_neg,
    cuDoubleComplex* out,
    size_t total,
    double delta
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const uint64_t* re = &re_mag[idx * HE_CRT_BIGINT_LIMBS];
    const uint64_t* im = &im_mag[idx * HE_CRT_BIGINT_LIMBS];
    double vr = he_big_to_f64(re);
    double vi = he_big_to_f64(im);
    if (re_neg[idx]) vr = -vr;
    if (im_neg[idx]) vi = -vi;
    vr /= delta;
    vi /= delta;
    out[idx] = make_cuDoubleComplex(vr, vi);
}

__global__ void wntt_forward_centered_kernel(
    const int64_t* in_coeff_centered,
    int64_t* out_eval_centered,
    const uint64_t* powers,
    int n,
    int phi
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t n2 = (size_t)n * (size_t)n;
    size_t total = (size_t)phi * n2;
    if (idx >= total) return;

    int x = (int)(idx % (size_t)n);
    size_t t = idx / (size_t)n;
    int y = (int)(t % (size_t)n);
    int w_out = (int)(t / (size_t)n);
    size_t pos = (size_t)y * (size_t)n + (size_t)x;

    uint64_t crt_acc[HE_CRT_BIGINT_LIMBS];
    for (int i = 0; i < HE_CRT_BIGINT_LIMBS; ++i) crt_acc[i] = 0;

    for (int limb = 0; limb < RNS_NUM_LIMBS; ++limb) {
        uint64_t q = d_he_moduli[limb];
        uint64_t acc_l = 0;
        size_t pow_base = ((size_t)limb * (size_t)phi + (size_t)w_out) * (size_t)phi;
        for (int r = 0; r < phi; ++r) {
            int64_t v = in_coeff_centered[(size_t)r * n2 + pos];
            int64_t modv = v % (int64_t)q;
            if (modv < 0) modv += (int64_t)q;
            uint64_t a = (uint64_t)modv;
            uint64_t w = powers[pow_base + (size_t)r];
            acc_l = add_mod(acc_l, mul_mod(a, w, q), q);
        }

        uint64_t tcoeff = mul_mod(acc_l, d_crt_inv[limb], q);
        uint64_t term[HE_CRT_BIGINT_LIMBS];
        he_big_mul_u64(&d_crt_M[(size_t)limb * HE_CRT_BIGINT_LIMBS], tcoeff, term);
        he_big_add_inplace(crt_acc, term);
        if (he_big_cmp(crt_acc, d_crt_Q) >= 0) {
            he_big_sub_inplace(crt_acc, d_crt_Q);
        }
    }

    bool neg = false;
    uint64_t mag[HE_CRT_BIGINT_LIMBS];
    if (he_big_cmp(crt_acc, d_crt_Q_half) > 0) {
        he_big_sub_rev(mag, d_crt_Q, crt_acc);
        neg = true;
    } else {
        for (int i = 0; i < HE_CRT_BIGINT_LIMBS; ++i) mag[i] = crt_acc[i];
    }
    out_eval_centered[idx] = he_big_to_i64_checked(mag, neg);
}

__global__ void wntt_inverse_centered_kernel(
    const int64_t* in_eval_centered,
    int64_t* out_coeff_centered,
    const uint64_t* inv_powers,
    int n,
    int phi
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t n2 = (size_t)n * (size_t)n;
    size_t total = (size_t)phi * n2;
    if (idx >= total) return;

    int x = (int)(idx % (size_t)n);
    size_t t = idx / (size_t)n;
    int y = (int)(t % (size_t)n);
    int r = (int)(t / (size_t)n);
    size_t pos = (size_t)y * (size_t)n + (size_t)x;

    uint64_t q = d_he_moduli[0];
    uint64_t acc = 0;
    size_t pow_base = 0; // limb0 slice
    for (int w = 0; w < phi; ++w) {
        int64_t v = in_eval_centered[(size_t)w * n2 + pos];
        int64_t modv = v % (int64_t)q;
        if (modv < 0) modv += (int64_t)q;
        uint64_t a = (uint64_t)modv;
        uint64_t wp = inv_powers[pow_base + (size_t)w * (size_t)phi + (size_t)r];
        acc = add_mod(acc, mul_mod(a, wp, q), q);
    }
    int64_t centered = (acc > (q >> 1)) ? (int64_t)acc - (int64_t)q : (int64_t)acc;
    out_coeff_centered[idx] = centered;
}

__global__ void wdft_forward_pair_kernel(
    const int64_t* in_re_coeff,
    const int64_t* in_im_coeff,
    double* out_re_eval,
    double* out_im_eval,
    const cuDoubleComplex* powers,
    int n,
    int phi
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t n2 = (size_t)n * (size_t)n;
    size_t total = (size_t)phi * n2;
    if (idx >= total) return;

    size_t pos = idx % n2;
    int w_out = (int)(idx / n2);
    size_t base = (size_t)w_out * (size_t)phi;

    double acc_re = 0.0;
    double acc_im = 0.0;
    for (int r = 0; r < phi; ++r) {
        double a_re = (double)in_re_coeff[(size_t)r * n2 + pos];
        double a_im = (double)in_im_coeff[(size_t)r * n2 + pos];
        cuDoubleComplex w = powers[base + (size_t)r];
        acc_re += a_re * cuCreal(w) - a_im * cuCimag(w);
        acc_im += a_re * cuCimag(w) + a_im * cuCreal(w);
    }
    out_re_eval[idx] = acc_re;
    out_im_eval[idx] = acc_im;
}

__global__ void wdft_forward_complex_kernel(
    const cuDoubleComplex* in_coeff,
    cuDoubleComplex* out_eval,
    const cuDoubleComplex* powers,
    int n,
    int phi
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t n2 = (size_t)n * (size_t)n;
    size_t total = (size_t)phi * n2;
    if (idx >= total) return;

    size_t pos = idx % n2;
    int w_out = (int)(idx / n2);
    size_t base = (size_t)w_out * (size_t)phi;

    double acc_re = 0.0;
    double acc_im = 0.0;
    for (int r = 0; r < phi; ++r) {
        cuDoubleComplex a = in_coeff[(size_t)r * n2 + pos];
        cuDoubleComplex w = powers[base + (size_t)r];
        acc_re += cuCreal(a) * cuCreal(w) - cuCimag(a) * cuCimag(w);
        acc_im += cuCreal(a) * cuCimag(w) + cuCimag(a) * cuCreal(w);
    }
    out_eval[idx] = make_cuDoubleComplex(acc_re, acc_im);
}

__global__ void wdft_inverse_pair_kernel(
    const double* in_re_eval,
    const double* in_im_eval,
    double* out_re_coeff,
    double* out_im_coeff,
    const cuDoubleComplex* inv_powers,
    int n,
    int phi
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t n2 = (size_t)n * (size_t)n;
    size_t total = (size_t)phi * n2;
    if (idx >= total) return;

    size_t pos = idx % n2;
    int r = (int)(idx / n2);

    double acc_re = 0.0;
    double acc_im = 0.0;
    for (int w = 0; w < phi; ++w) {
        double a_re = in_re_eval[(size_t)w * n2 + pos];
        double a_im = in_im_eval[(size_t)w * n2 + pos];
        cuDoubleComplex iw = inv_powers[(size_t)w * (size_t)phi + (size_t)r];
        acc_re += a_re * cuCreal(iw) - a_im * cuCimag(iw);
        acc_im += a_re * cuCimag(iw) + a_im * cuCreal(iw);
    }
    out_re_coeff[idx] = acc_re;
    out_im_coeff[idx] = acc_im;
}

__global__ void count_nonzero_i64_kernel(const int64_t* in, size_t total, unsigned long long* out_nonzero) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    if (in[idx] != 0) atomicAdd(out_nonzero, 1ULL);
}

__global__ void count_big_over_i64_kernel(const uint64_t* mag, size_t total, unsigned long long* out_count) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const uint64_t* p = mag + idx * HE_CRT_BIGINT_LIMBS;
    bool over = false;
    for (int i = 1; i < HE_CRT_BIGINT_LIMBS; ++i) {
        if (p[i] != 0) {
            over = true;
            break;
        }
    }
    if (over || p[0] > (uint64_t)INT64_MAX) atomicAdd(out_count, 1ULL);
}

static inline void xy_ntt_forward_selected(uint64_t* data, uint64_t* tmp,
                                           int limbs, int batch_count, int n, cudaStream_t stream) {
    if (kDbgUsePhantomXY) {
        (void)tmp;
        xy_ntt_forward_phantom(data, limbs, batch_count, n, stream);
    } else {
        xy_ntt_forward_gl(data, tmp, limbs, batch_count, n, stream);
    }
}

static inline void xy_ntt_backward_selected(uint64_t* data, uint64_t* tmp,
                                            int limbs, int batch_count, int n, cudaStream_t stream) {
    if (kDbgUsePhantomXY) {
        (void)tmp;
        xy_ntt_backward_phantom(data, limbs, batch_count, n, stream);
    } else {
        xy_ntt_backward_gl(data, tmp, limbs, batch_count, n, stream);
    }
}

// W-CRT forward for secret key layout: [W][Limb][n]
__global__ void wntt_forward_vector_kernel(
    const uint64_t* in,
    uint64_t* out,
    const uint64_t* powers,
    int n, int limbs, int phi
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)phi * (size_t)limbs * (size_t)n;
    if (idx >= total) return;

    int x = (int)(idx % (size_t)n);
    size_t t = idx / (size_t)n;
    int limb = (int)(t % (size_t)limbs);
    int w_out = (int)(t / (size_t)limbs);

    uint64_t q = d_he_moduli[limb];
    uint64_t acc = 0;
    size_t pow_base = ((size_t)limb * (size_t)phi + (size_t)w_out) * (size_t)phi;
    for (int r = 0; r < phi; ++r) {
        size_t in_off = ((size_t)r * (size_t)limbs + (size_t)limb) * (size_t)n + (size_t)x;
        uint64_t a = in[in_off];
        uint64_t w = powers[pow_base + (size_t)r];
        acc = add_mod(acc, mul_mod(a, w, q), q);
    }
    out[idx] = acc;
}

void generate_secret_key(SecretKey& sk, int limbs) {
    sk.num_limbs = limbs;

    const int n = MATRIX_N;
    const int phi = BATCH_SIZE;
    size_t total = (size_t)phi * (size_t)limbs * (size_t)n;

    uint64_t* d_s_coeff = nullptr;
    uint64_t* d_s_tmp = nullptr;
    cudaMalloc(&d_s_coeff, total * sizeof(uint64_t));
    cudaMemset(d_s_coeff, 0, total * sizeof(uint64_t));

    cudaMalloc(&sk.data, total * sizeof(uint64_t));
    cudaMemset(sk.data, 0, total * sizeof(uint64_t));
    cudaMalloc(&d_s_tmp, total * sizeof(uint64_t));

    int thr = 256;
    int blk = (int)((total + thr - 1) / thr);

    ternary_secret_kernel<<<blk, thr>>>(d_s_coeff, total, limbs, n);

    // W-CRT: coeff -> evaluation (Phi_p)
    wntt_forward_vector_kernel<<<blk, thr>>>(d_s_coeff, sk.data, d_wntt_powers, n, limbs, phi);

    // X-NTT over each W slot; output stays in sk.data.
    xy_ntt_forward_selected(sk.data, d_s_tmp, limbs, phi, n, 0);

    // ---- 运行时钉死：确认这一版真的跑到了 ----
    // 只打印一次（避免刷屏）
    cudaDeviceSynchronize();
    printf("[DBG] generate_secret_key: using ternary_secret_kernel, N=%d, XY=%s\n",
           MATRIX_N, kDbgUsePhantomXY ? "phantom" : "gl-custom");

    cudaFree(d_s_coeff);
    cudaFree(d_s_tmp);
}


// 点加 in-place (X-NTT or coeff, depending on caller)
__global__ void add_error_inplace_kernel(uint64_t* b, const uint64_t* e, size_t total_len, int limbs, int n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_len) return;

    // 解析当前 idx 对应的模数 q
    // 假设布局为 Poly-Major: [Y][Limb][X]
    int single_poly_size = limbs * n;
    
    // 计算当前 idx 处于哪个 limb
    int offset_in_poly = (int)(idx % (size_t)single_poly_size);
    int limb_idx = offset_in_poly / n;

    uint64_t q = d_he_moduli[limb_idx];

    // 执行模加: b = (b + e) mod q
    b[idx] = add_mod(b[idx], e[idx], q);
}


__global__ void matrix_to_poly_kernel(const uint64_t* in, uint64_t* out,
                                      int n, int limbs, int phi) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)phi * (size_t)limbs * (size_t)n * (size_t)n;
    if (idx >= total) return;

    int x = (int)(idx % n);
    size_t t = idx / n;
    int y = (int)(t % n);
    t /= n;
    int limb = (int)(t % limbs);
    int w = (int)(t / limbs);

    size_t in_off = ((size_t)w * (size_t)limbs + (size_t)limb) * (size_t)(n * n) +
                    (size_t)y * (size_t)n + (size_t)x;
    int poly = w * n + y;
    size_t out_off = ((size_t)poly * (size_t)limbs + (size_t)limb) * (size_t)n + (size_t)x;
    out[out_off] = in[in_off];
}

__global__ void poly_to_matrix_kernel(const uint64_t* in, uint64_t* out,
                                      int n, int limbs, int phi) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)phi * (size_t)limbs * (size_t)n * (size_t)n;
    if (idx >= total) return;

    int x = (int)(idx % n);
    size_t t = idx / n;
    int y = (int)(t % n);
    t /= n;
    int limb = (int)(t % limbs);
    int w = (int)(t / limbs);

    int poly = w * n + y;
    size_t in_off = ((size_t)poly * (size_t)limbs + (size_t)limb) * (size_t)n + (size_t)x;
    size_t out_off = ((size_t)w * (size_t)limbs + (size_t)limb) * (size_t)(n * n) +
                     (size_t)y * (size_t)n + (size_t)x;
    out[out_off] = in[in_off];
}

void encrypt(const uint64_t* message_coeffs, const SecretKey& sk, RLWECiphertext& ct) {
    const int n = MATRIX_N;
    const int limbs = ct.num_limbs;
    const int phi = BATCH_SIZE;
    size_t total_coeffs = (size_t)phi * (size_t)n * (size_t)limbs * (size_t)n;
    size_t total_bytes = total_coeffs * sizeof(uint64_t);

    uint64_t* d_b = ct.data;
    uint64_t* d_a = ct.data + total_coeffs;

    uint64_t* d_m_poly = nullptr;
    uint64_t* d_a_poly = nullptr;
    uint64_t* d_b_poly = nullptr;
    uint64_t* d_e_poly = nullptr;
    uint64_t* d_e_eval = nullptr;
    uint64_t* d_a_eval = nullptr;
    uint64_t* d_a_ntt  = nullptr;
    uint64_t* d_t_ntt  = nullptr;
    uint64_t* d_t_poly = nullptr;
    uint64_t* d_xy_tmp = nullptr;
    cudaMalloc(&d_m_poly, total_bytes);
    cudaMalloc(&d_a_poly, total_bytes);
    cudaMalloc(&d_b_poly, total_bytes);
    cudaMalloc(&d_e_poly, total_bytes);
    cudaMalloc(&d_e_eval, total_bytes);
    cudaMalloc(&d_a_eval, total_bytes);
    cudaMalloc(&d_a_ntt,  total_bytes);
    cudaMalloc(&d_t_ntt,  total_bytes);
    cudaMalloc(&d_t_poly, total_bytes);
    cudaMalloc(&d_xy_tmp, total_bytes);

    // ==========================================
    // 1. 处理消息 m (W-CRT eval domain, X coeff)
    // ==========================================
    int thr = 256;
    int blk = (int)((total_coeffs + thr - 1) / thr);
    matrix_to_poly_kernel<<<blk, thr>>>(message_coeffs, d_m_poly, n, limbs, phi);

    // ==========================================
    // 2. 处理随机多项式 a (W coeff -> W-CRT eval -> X-NTT)
    // ==========================================
    // Generate Uniform Random (Coeff domain)
    uniform_random_kernel<<<blk, thr>>>(d_a_poly, total_coeffs, ct.num_limbs, n);
    wntt_forward_matrix_kernel<<<blk, thr>>>(d_a_poly, d_a_eval, d_wntt_powers, n, limbs, phi);
    cudaMemcpy(d_a_ntt, d_a_eval, total_bytes, cudaMemcpyDeviceToDevice);
    xy_ntt_forward_selected(d_a_ntt, d_xy_tmp, limbs, phi * n, n, 0);

    // ==========================================
    // 3. 处理噪声 e (W coeff -> W-CRT eval)
    // ==========================================
    if (kDbgZeroNoise) {
        cudaMemset(d_e_eval, 0, total_bytes);
    } else {
        gaussian_noise_kernel<<<blk, thr>>>(d_e_poly, total_coeffs, ct.num_limbs, n);
        wntt_forward_matrix_kernel<<<blk, thr>>>(d_e_poly, d_e_eval, d_wntt_powers, n, limbs, phi);
    }

    // ==========================================
    // 4. 融合计算 (X coeff domain); ciphertext stays in X coeff
    // ==========================================
    
    pointwise_mul_s_kernel<<<blk, thr>>>(d_a_ntt, sk.data, d_t_ntt,
                                         total_coeffs, ct.num_limbs, n);
    cudaMemcpy(d_t_poly, d_t_ntt, total_bytes, cudaMemcpyDeviceToDevice);
    xy_ntt_backward_selected(d_t_poly, d_xy_tmp, limbs, phi * n, n, 0);

    combine_b_kernel<<<blk, thr>>>(d_m_poly, d_t_poly, d_e_eval,
                                   d_b_poly, total_coeffs, ct.num_limbs, n);

    // Store ciphertext in matrix layout (W-CRT eval, X coeff domain)
    poly_to_matrix_kernel<<<blk, thr>>>(d_a_eval, d_a, n, limbs, phi);
    poly_to_matrix_kernel<<<blk, thr>>>(d_b_poly, d_b, n, limbs, phi);

    cudaFree(d_m_poly);
    cudaFree(d_a_poly);
    cudaFree(d_b_poly);
    cudaFree(d_e_poly);
    cudaFree(d_e_eval);
    cudaFree(d_a_eval);
    cudaFree(d_a_ntt);
    cudaFree(d_t_ntt);
    cudaFree(d_t_poly);
    cudaFree(d_xy_tmp);
}

void encrypt_pair(const uint64_t* msg_re, const uint64_t* msg_im,
                  const SecretKey& sk, RLWECiphertext& ct_re, RLWECiphertext& ct_im) {
    const int n = MATRIX_N;
    const int limbs = ct_re.num_limbs;
    const int phi = BATCH_SIZE;
    size_t total_coeffs = (size_t)phi * (size_t)n * (size_t)limbs * (size_t)n;
    size_t total_bytes = total_coeffs * sizeof(uint64_t);

    uint64_t* d_b_re = ct_re.data;
    uint64_t* d_a_re = ct_re.data + total_coeffs;
    uint64_t* d_b_im = ct_im.data;
    uint64_t* d_a_im = ct_im.data + total_coeffs;

    uint64_t* d_m_re = nullptr;
    uint64_t* d_m_im = nullptr;
    uint64_t* d_a_poly = nullptr;
    uint64_t* d_a_eval = nullptr;
    uint64_t* d_a_ntt = nullptr;
    uint64_t* d_t_ntt = nullptr;
    uint64_t* d_t_poly = nullptr;
    uint64_t* d_e_re_poly = nullptr;
    uint64_t* d_e_im_poly = nullptr;
    uint64_t* d_e_re_eval = nullptr;
    uint64_t* d_e_im_eval = nullptr;
    uint64_t* d_b_re_poly = nullptr;
    uint64_t* d_b_im_poly = nullptr;
    uint64_t* d_xy_tmp = nullptr;

    cudaMalloc(&d_m_re, total_bytes);
    cudaMalloc(&d_m_im, total_bytes);
    cudaMalloc(&d_a_poly, total_bytes);
    cudaMalloc(&d_a_eval, total_bytes);
    cudaMalloc(&d_a_ntt, total_bytes);
    cudaMalloc(&d_t_ntt, total_bytes);
    cudaMalloc(&d_t_poly, total_bytes);
    cudaMalloc(&d_e_re_poly, total_bytes);
    cudaMalloc(&d_e_im_poly, total_bytes);
    cudaMalloc(&d_e_re_eval, total_bytes);
    cudaMalloc(&d_e_im_eval, total_bytes);
    cudaMalloc(&d_b_re_poly, total_bytes);
    cudaMalloc(&d_b_im_poly, total_bytes);
    cudaMalloc(&d_xy_tmp, total_bytes);

    int thr = 256;
    int blk = (int)((total_coeffs + thr - 1) / thr);

    // m in W-CRT eval (X coeff)
    matrix_to_poly_kernel<<<blk, thr>>>(msg_re, d_m_re, n, limbs, phi);
    matrix_to_poly_kernel<<<blk, thr>>>(msg_im, d_m_im, n, limbs, phi);

    // shared a: W coeff -> W-CRT eval -> X-NTT
    uniform_random_kernel<<<blk, thr>>>(d_a_poly, total_coeffs, limbs, n);
    wntt_forward_matrix_kernel<<<blk, thr>>>(d_a_poly, d_a_eval, d_wntt_powers, n, limbs, phi);
    cudaMemcpy(d_a_ntt, d_a_eval, total_bytes, cudaMemcpyDeviceToDevice);
    xy_ntt_forward_selected(d_a_ntt, d_xy_tmp, limbs, phi * n, n, 0);

    // independent errors
    if (kDbgZeroNoise) {
        cudaMemset(d_e_re_eval, 0, total_bytes);
        cudaMemset(d_e_im_eval, 0, total_bytes);
    } else {
        gaussian_noise_kernel<<<blk, thr>>>(d_e_re_poly, total_coeffs, limbs, n);
        gaussian_noise_kernel<<<blk, thr>>>(d_e_im_poly, total_coeffs, limbs, n);
        wntt_forward_matrix_kernel<<<blk, thr>>>(d_e_re_poly, d_e_re_eval, d_wntt_powers, n, limbs, phi);
        wntt_forward_matrix_kernel<<<blk, thr>>>(d_e_im_poly, d_e_im_eval, d_wntt_powers, n, limbs, phi);
    }

    // t = a*s (X-NTT), then back to coeff
    // Forward-selected NTT is in-place on d_a_ntt (both phantom/gl-custom paths).
    pointwise_mul_s_kernel<<<blk, thr>>>(d_a_ntt, sk.data, d_t_ntt, total_coeffs, limbs, n);
    cudaMemcpy(d_t_poly, d_t_ntt, total_bytes, cudaMemcpyDeviceToDevice);
    xy_ntt_backward_selected(d_t_poly, d_xy_tmp, limbs, phi * n, n, 0);

    // b_re = m_re - t + e_re, b_im = m_im - t + e_im
    combine_b_kernel<<<blk, thr>>>(d_m_re, d_t_poly, d_e_re_eval, d_b_re_poly, total_coeffs, limbs, n);
    combine_b_kernel<<<blk, thr>>>(d_m_im, d_t_poly, d_e_im_eval, d_b_im_poly, total_coeffs, limbs, n);

    // store ciphertext (W-CRT eval, X coeff)
    poly_to_matrix_kernel<<<blk, thr>>>(d_a_eval, d_a_re, n, limbs, phi);
    poly_to_matrix_kernel<<<blk, thr>>>(d_a_eval, d_a_im, n, limbs, phi);
    poly_to_matrix_kernel<<<blk, thr>>>(d_b_re_poly, d_b_re, n, limbs, phi);
    poly_to_matrix_kernel<<<blk, thr>>>(d_b_im_poly, d_b_im, n, limbs, phi);

    cudaFree(d_m_re);
    cudaFree(d_m_im);
    cudaFree(d_a_poly);
    cudaFree(d_a_eval);
    cudaFree(d_a_ntt);
    cudaFree(d_t_ntt);
    cudaFree(d_t_poly);
    cudaFree(d_e_re_poly);
    cudaFree(d_e_im_poly);
    cudaFree(d_e_re_eval);
    cudaFree(d_e_im_eval);
    cudaFree(d_b_re_poly);
    cudaFree(d_b_im_poly);
    cudaFree(d_xy_tmp);
}
static void decrypt_to_eval(const RLWECiphertext& ct, const SecretKey& sk, uint64_t* out_eval_poly) {
    const int n = MATRIX_N;
    const int limbs = ct.num_limbs;
    const int phi = BATCH_SIZE;
    size_t total_coeffs = (size_t)phi * (size_t)n * (size_t)limbs * (size_t)n;

    uint64_t* d_b = ct.data;
    uint64_t* d_a = ct.data + total_coeffs;

    uint64_t* d_b_poly = nullptr;
    uint64_t* d_a_poly = nullptr;
    uint64_t* d_m_poly = nullptr;
    uint64_t* d_a_ntt  = nullptr;
    uint64_t* d_t_ntt  = nullptr;
    uint64_t* d_t_poly = nullptr;
    uint64_t* d_xy_tmp = nullptr;
    cudaMalloc(&d_b_poly, total_coeffs * sizeof(uint64_t));
    cudaMalloc(&d_a_poly, total_coeffs * sizeof(uint64_t));
    cudaMalloc(&d_m_poly, total_coeffs * sizeof(uint64_t));
    cudaMalloc(&d_a_ntt,  total_coeffs * sizeof(uint64_t));
    cudaMalloc(&d_t_ntt,  total_coeffs * sizeof(uint64_t));
    cudaMalloc(&d_t_poly, total_coeffs * sizeof(uint64_t));
    cudaMalloc(&d_xy_tmp, total_coeffs * sizeof(uint64_t));

    int thr = 256;
    int blk = (int)((total_coeffs + thr - 1) / thr);
    matrix_to_poly_kernel<<<blk, thr>>>(d_b, d_b_poly, n, limbs, phi);
    matrix_to_poly_kernel<<<blk, thr>>>(d_a, d_a_poly, n, limbs, phi);

    cudaMemcpy(d_a_ntt, d_a_poly, total_coeffs * sizeof(uint64_t), cudaMemcpyDeviceToDevice);
    xy_ntt_forward_selected(d_a_ntt, d_xy_tmp, limbs, phi * n, n, 0);
    pointwise_mul_s_kernel<<<blk, thr>>>(d_a_ntt, sk.data, d_t_ntt,
                                         total_coeffs, ct.num_limbs, n);
    cudaMemcpy(d_t_poly, d_t_ntt, total_coeffs * sizeof(uint64_t), cudaMemcpyDeviceToDevice);
    xy_ntt_backward_selected(d_t_poly, d_xy_tmp, limbs, phi * n, n, 0);

    add_poly_kernel<<<blk, thr>>>(d_b_poly, d_t_poly, d_m_poly,
                                  total_coeffs, ct.num_limbs, n);

    cudaMemcpy(out_eval_poly, d_m_poly, total_coeffs * sizeof(uint64_t), cudaMemcpyDeviceToDevice);

    cudaFree(d_b_poly);
    cudaFree(d_a_poly);
    cudaFree(d_m_poly);
    cudaFree(d_a_ntt);
    cudaFree(d_t_ntt);
    cudaFree(d_t_poly);
    cudaFree(d_xy_tmp);
}

void decrypt_to_eval_matrix(const RLWECiphertext& ct, const SecretKey& sk, uint64_t* out_eval_matrix) {
    const int n = MATRIX_N;
    const int limbs = ct.num_limbs;
    const int phi = BATCH_SIZE;
    const size_t total_coeffs = (size_t)phi * (size_t)n * (size_t)limbs * (size_t)n;

    uint64_t* d_eval_poly = nullptr;
    cudaMalloc(&d_eval_poly, total_coeffs * sizeof(uint64_t));
    decrypt_to_eval(ct, sk, d_eval_poly);

    int thr = 256;
    int blk = (int)((total_coeffs + thr - 1) / thr);
    poly_to_matrix_kernel<<<blk, thr>>>(d_eval_poly, out_eval_matrix, n, limbs, phi);
    cudaFree(d_eval_poly);
}

void decrypt_and_decode(const RLWECiphertext& ct_re, const RLWECiphertext& ct_im,
                        const SecretKey& sk, cuDoubleComplex* output_msg) {
    const int n = MATRIX_N;
    const int limbs = ct_re.num_limbs;
    const int phi = BATCH_SIZE;
    size_t total_coeffs = (size_t)phi * (size_t)n * (size_t)limbs * (size_t)n;

    uint64_t* d_eval_re = nullptr;
    uint64_t* d_eval_im = nullptr;
    uint64_t* d_coeff_re = nullptr;
    uint64_t* d_coeff_im = nullptr;
    uint64_t* d_comp_re_mag = nullptr;
    uint64_t* d_comp_im_mag = nullptr;
    uint8_t* d_comp_re_neg = nullptr;
    uint8_t* d_comp_im_neg = nullptr;
    int64_t* d_comp_re = nullptr;
    int64_t* d_comp_im = nullptr;
    cuDoubleComplex* d_coeff_cx = nullptr;
    cuDoubleComplex* d_eval_cx = nullptr;
    cudaMalloc(&d_eval_re, total_coeffs * sizeof(uint64_t));
    cudaMalloc(&d_eval_im, total_coeffs * sizeof(uint64_t));
    cudaMalloc(&d_coeff_re, total_coeffs * sizeof(uint64_t));
    cudaMalloc(&d_coeff_im, total_coeffs * sizeof(uint64_t));
    cudaMalloc(&d_comp_re_mag, (size_t)phi * (size_t)n * (size_t)n * (size_t)HE_CRT_BIGINT_LIMBS * sizeof(uint64_t));
    cudaMalloc(&d_comp_im_mag, (size_t)phi * (size_t)n * (size_t)n * (size_t)HE_CRT_BIGINT_LIMBS * sizeof(uint64_t));
    cudaMalloc(&d_comp_re_neg, (size_t)phi * (size_t)n * (size_t)n * sizeof(uint8_t));
    cudaMalloc(&d_comp_im_neg, (size_t)phi * (size_t)n * (size_t)n * sizeof(uint8_t));
    cudaMalloc(&d_comp_re, (size_t)phi * (size_t)n * (size_t)n * sizeof(int64_t));
    cudaMalloc(&d_comp_im, (size_t)phi * (size_t)n * (size_t)n * sizeof(int64_t));
    cudaMalloc(&d_coeff_cx, (size_t)phi * (size_t)n * (size_t)n * sizeof(cuDoubleComplex));
    cudaMalloc(&d_eval_cx, (size_t)phi * (size_t)n * (size_t)n * sizeof(cuDoubleComplex));

    // decrypt_to_eval outputs poly-major [poly=w*n+y][limb][x], matching wntt_inverse_matrix input.
    decrypt_to_eval(ct_re, sk, d_eval_re);
    decrypt_to_eval(ct_im, sk, d_eval_im);

    const size_t nn = (size_t)n * (size_t)n;
    const int threads = 256;
    const int blocks = (int)((nn + threads - 1) / threads);
    const size_t lane_stride = (size_t)limbs * nn;
    const size_t out_stride = nn;
    Encoder decoder(n);

    wntt_inverse_matrix(d_eval_re, d_coeff_re, n, limbs, phi, 0);
    wntt_inverse_matrix(d_eval_im, d_coeff_im, n, limbs, phi, 0);

    for (int ell = 0; ell < phi; ++ell) {
        const uint64_t* lane_coeff_re = d_coeff_re + (size_t)ell * lane_stride;
        const uint64_t* lane_coeff_im = d_coeff_im + (size_t)ell * lane_stride;
        crt_compose_centerlift_big(
            lane_coeff_re,
            d_comp_re_mag + (size_t)ell * nn * HE_CRT_BIGINT_LIMBS,
            d_comp_re_neg + (size_t)ell * nn,
            (int)nn, limbs, 0
        );
        crt_compose_centerlift_big(
            lane_coeff_im,
            d_comp_im_mag + (size_t)ell * nn * HE_CRT_BIGINT_LIMBS,
            d_comp_im_neg + (size_t)ell * nn,
            (int)nn, limbs, 0
        );
    }

    size_t total_centered = (size_t)phi * nn;
    compose_big_pair_to_complex_by_delta_kernel<<<(int)((total_centered + threads - 1) / threads), threads>>>(
        d_comp_re_mag, d_comp_re_neg, d_comp_im_mag, d_comp_im_neg, d_coeff_cx, total_centered, SCALING_FACTOR);
    wdft_forward_complex(d_coeff_cx, d_eval_cx, n, phi, 0);

    for (int ell = 0; ell < phi; ++ell) {
        const cuDoubleComplex* lane_eval = d_eval_cx + (size_t)ell * nn;
        decoder.decode_from_eval_complex(lane_eval, output_msg + (size_t)ell * out_stride);
    }

    cudaFree(d_eval_re);
    cudaFree(d_eval_im);
    cudaFree(d_coeff_re);
    cudaFree(d_coeff_im);
    cudaFree(d_comp_re_mag);
    cudaFree(d_comp_im_mag);
    cudaFree(d_comp_re_neg);
    cudaFree(d_comp_im_neg);
    cudaFree(d_comp_re);
    cudaFree(d_comp_im);
    cudaFree(d_coeff_cx);
    cudaFree(d_eval_cx);
}

void add_ciphertexts(const RLWECiphertext& ct1, const RLWECiphertext& ct2, RLWECiphertext& res) {
    size_t total_coeffs = (size_t)BATCH_SIZE * (size_t)MATRIX_N * (size_t)ct1.num_limbs * (size_t)MATRIX_N;
    
    uint64_t* b1 = ct1.data;
    uint64_t* a1 = ct1.data + total_coeffs;
    uint64_t* b2 = ct2.data;
    uint64_t* a2 = ct2.data + total_coeffs;
    
    uint64_t* b_res = res.data;
    uint64_t* a_res = res.data + total_coeffs;

    int thr = 256;
    int blk = (total_coeffs + thr - 1) / thr;
    
    add_ct_kernel<<<blk, thr>>>(b1, a1, b2, a2, b_res, a_res, total_coeffs, res.num_limbs, MATRIX_N);
}

void multiply_ciphertexts_raw(const RLWECiphertext& ct1, const RLWECiphertext& ct2, 
                              uint64_t* d0, uint64_t* d1, uint64_t* d2) {
    size_t total_coeffs = (size_t)BATCH_SIZE * (size_t)MATRIX_N * (size_t)ct1.num_limbs * (size_t)MATRIX_N;

    uint64_t* b1 = ct1.data;
    uint64_t* a1 = ct1.data + total_coeffs;
    uint64_t* b2 = ct2.data;
    uint64_t* a2 = ct2.data + total_coeffs;

    int thr = 256;
    int blk = (total_coeffs + thr - 1) / thr;
    
    mul_tensor_kernel<<<blk, thr>>>(b1, a1, b2, a2, d0, d1, d2, total_coeffs, ct1.num_limbs, MATRIX_N);
}

} // namespace matrix_fhe
