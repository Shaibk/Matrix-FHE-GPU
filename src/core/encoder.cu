#include "../../include/core/encoder.cuh"
#include "../../include/core/config.h"
#include "../../include/backend/phantom_math.cuh"
#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>

namespace matrix_fhe {
using namespace phantom_math;

// --- CPU Helpers ---
uint64_t mul_mod_host(uint64_t a, uint64_t b, uint64_t mod) { return (unsigned __int128)a * b % mod; }
uint64_t modInverse(uint64_t n, uint64_t mod) {
    uint64_t res = 1, base = n % mod, exp = mod - 2;
    while(exp > 0) {
        if(exp % 2 == 1) res = mul_mod_host(res, base, mod);
        base = mul_mod_host(base, base, mod);
        exp /= 2;
    }
    return res;
}

// --- Device Kernels ---
__device__ int d_dbg_once = 0;

__constant__ uint64_t d_rns_moduli[RNS_NUM_LIMBS];

// Fixed-size big integer limbs for CRT reconstruction (little-endian limbs)
static constexpr int BIGINT_LIMBS = 7;
__constant__ uint64_t d_crt_M[RNS_NUM_LIMBS * BIGINT_LIMBS];   // M_i = Q / q_i
__constant__ uint64_t d_crt_Q[BIGINT_LIMBS];
__constant__ uint64_t d_crt_Q_half[BIGINT_LIMBS];
__constant__ uint64_t d_crt_inv[RNS_NUM_LIMBS];               // inv_i = (M_i^-1 mod q_i)

__global__ void quantize_soa_kernel(const cuDoubleComplex* input, 
                                  uint64_t* out_real, uint64_t* out_imag, 
                                  int n, double delta, int limbs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int64_t ix = llround(cuCreal(input[idx]) * delta);
        int64_t iy = llround(cuCimag(input[idx]) * delta);
        for(int k=0; k<limbs; k++) {
            uint64_t q = d_rns_moduli[k];
            int out_idx = k * n + idx;
            int64_t mx = ix % (int64_t)q; if(mx < 0) mx += q; out_real[out_idx] = mx;
            int64_t my = iy % (int64_t)q; if(my < 0) my += q; out_imag[out_idx] = my;
        }
    }
}

// ----------------- Exact CRT Reconstruction (device) -----------------
__device__ __forceinline__ int big_cmp(const uint64_t* a, const uint64_t* b) {
    for (int i = BIGINT_LIMBS - 1; i >= 0; --i) {
        if (a[i] > b[i]) return 1;
        if (a[i] < b[i]) return -1;
    }
    return 0;
}

__device__ __forceinline__ void big_add_inplace(uint64_t* a, const uint64_t* b) {
    unsigned __int128 carry = 0;
    for (int i = 0; i < BIGINT_LIMBS; ++i) {
        unsigned __int128 s = (unsigned __int128)a[i] + b[i] + carry;
        a[i] = (uint64_t)s;
        carry = s >> 64;
    }
}

__device__ __forceinline__ void big_sub_inplace(uint64_t* a, const uint64_t* b) {
    uint64_t borrow = 0;
    for (int i = 0; i < BIGINT_LIMBS; ++i) {
        uint64_t bi = b[i] + borrow;
        borrow = (a[i] < bi);
        a[i] -= bi;
    }
}

__device__ __forceinline__ void big_sub_rev(uint64_t* out, const uint64_t* a, const uint64_t* b) {
    // out = b - a  (assumes b >= a)
    uint64_t borrow = 0;
    for (int i = 0; i < BIGINT_LIMBS; ++i) {
        uint64_t ai = a[i] + borrow;
        borrow = (b[i] < ai);
        out[i] = b[i] - ai;
    }
}

__device__ __forceinline__ void big_mul_u64(const uint64_t* a, uint64_t m, uint64_t* out) {
    unsigned __int128 carry = 0;
    for (int i = 0; i < BIGINT_LIMBS; ++i) {
        unsigned __int128 p = (unsigned __int128)a[i] * m + carry;
        out[i] = (uint64_t)p;
        carry = p >> 64;
    }
}

__device__ __forceinline__ double big_to_double(const uint64_t* a) {
    const double two64 = 18446744073709551616.0; // 2^64
    double v = 0.0;
    for (int i = BIGINT_LIMBS - 1; i >= 0; --i) {
        v = v * two64 + (double)a[i];
    }
    return v;
}

__device__ __forceinline__ uint64_t mul_mod_u128(uint64_t a, uint64_t b, uint64_t mod) {
    unsigned __int128 p = (unsigned __int128)a * (unsigned __int128)b;
    return (uint64_t)(p % mod);
}

__global__ void dequantize_exact_kernel(
    const uint64_t* in_real, const uint64_t* in_imag,
    cuDoubleComplex* output, int n, double delta, int limbs
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    auto reconstruct = [&](const uint64_t* data) -> double {
        uint64_t acc[BIGINT_LIMBS];
        for (int i = 0; i < BIGINT_LIMBS; ++i) acc[i] = 0;

        for (int k = 0; k < limbs; ++k) {
            uint64_t qi = d_rns_moduli[k];
            uint64_t xk = data[k * n + idx];
            uint64_t t = mul_mod_u128(xk, d_crt_inv[k], qi);

            uint64_t term[BIGINT_LIMBS];
            const uint64_t* Mk = &d_crt_M[k * BIGINT_LIMBS];
            big_mul_u64(Mk, t, term);           // term = M_k * t (term < Q)
            big_add_inplace(acc, term);         // acc += term
            if (big_cmp(acc, d_crt_Q) >= 0) {
                big_sub_inplace(acc, d_crt_Q);  // acc -= Q
            }
        }

        bool neg = false;
        if (big_cmp(acc, d_crt_Q_half) > 0) {
            uint64_t mag[BIGINT_LIMBS];
            big_sub_rev(mag, acc, d_crt_Q);     // mag = Q - acc
            for (int i = 0; i < BIGINT_LIMBS; ++i) acc[i] = mag[i];
            neg = true;
        }

        double val = big_to_double(acc) / delta;
        return neg ? -val : val;
    };

    output[idx] = make_cuDoubleComplex(reconstruct(in_real), reconstruct(in_imag));
}

__global__ void crt_compose_centerlift_kernel(
    const uint64_t* in_rns,
    int64_t* out_centered,
    int n2,
    int limbs
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n2) return;

    uint64_t acc[BIGINT_LIMBS];
    for (int i = 0; i < BIGINT_LIMBS; ++i) acc[i] = 0;

    for (int k = 0; k < limbs; ++k) {
        uint64_t qi = d_rns_moduli[k];
        uint64_t xk = in_rns[k * n2 + idx];
        uint64_t t = mul_mod_u128(xk, d_crt_inv[k], qi);

        uint64_t term[BIGINT_LIMBS];
        const uint64_t* Mk = &d_crt_M[k * BIGINT_LIMBS];
        big_mul_u64(Mk, t, term);
        big_add_inplace(acc, term);
        if (big_cmp(acc, d_crt_Q) >= 0) {
            big_sub_inplace(acc, d_crt_Q);
        }
    }

    bool neg = false;
    if (big_cmp(acc, d_crt_Q_half) > 0) {
        uint64_t mag[BIGINT_LIMBS];
        big_sub_rev(mag, acc, d_crt_Q);
        for (int i = 0; i < BIGINT_LIMBS; ++i) acc[i] = mag[i];
        neg = true;
    }

    // Expected message coefficients are in small range; clamp if host uses larger values.
    int64_t v = (int64_t)acc[0];
    out_centered[idx] = neg ? -v : v;
}

__global__ void crt_compose_centerlift_big_kernel(
    const uint64_t* in_rns,
    uint64_t* out_mag,
    uint8_t* out_neg,
    int n2,
    int limbs
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n2) return;

    uint64_t acc[BIGINT_LIMBS];
    for (int i = 0; i < BIGINT_LIMBS; ++i) acc[i] = 0;

    for (int k = 0; k < limbs; ++k) {
        uint64_t qi = d_rns_moduli[k];
        uint64_t xk = in_rns[k * n2 + idx];
        uint64_t t = mul_mod_u128(xk, d_crt_inv[k], qi);

        uint64_t term[BIGINT_LIMBS];
        const uint64_t* Mk = &d_crt_M[k * BIGINT_LIMBS];
        big_mul_u64(Mk, t, term);
        big_add_inplace(acc, term);
        if (big_cmp(acc, d_crt_Q) >= 0) {
            big_sub_inplace(acc, d_crt_Q);
        }
    }

    bool neg = false;
    uint64_t mag[BIGINT_LIMBS];
    if (big_cmp(acc, d_crt_Q_half) > 0) {
        big_sub_rev(mag, acc, d_crt_Q); // mag = Q - acc
        neg = true;
    } else {
        for (int i = 0; i < BIGINT_LIMBS; ++i) mag[i] = acc[i];
    }

    size_t base = (size_t)idx * BIGINT_LIMBS;
    for (int i = 0; i < BIGINT_LIMBS; ++i) out_mag[base + (size_t)i] = mag[i];
    out_neg[idx] = (uint8_t)(neg ? 1 : 0);
}

void crt_compose_centerlift_big(
    const uint64_t* d_in_rns,
    uint64_t* d_out_mag,
    uint8_t* d_out_neg,
    int n2,
    int limbs,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (n2 + threads - 1) / threads;
    crt_compose_centerlift_big_kernel<<<blocks, threads, 0, stream>>>(
        d_in_rns, d_out_mag, d_out_neg, n2, limbs
    );
}

// 192-bit Helper: Accumulate
__device__ void add_to_192(uint64_t* acc, uint64_t val_lo, uint64_t val_hi) {
    // 将三个 limbs 的加法放在同一个 asm 块中
    // 强制编译器分配寄存器，并保证指令连续执行
    asm volatile (
        "add.cc.u64      %0, %0, %3; \n\t"  // acc[0] += val_lo, set CC
        "addc.cc.u64     %1, %1, %4; \n\t"  // acc[1] += val_hi + Carry, set CC
        "addc.u64        %2, %2, 0;  \n\t"  // acc[2] += 0 + Carry
        : "+l"(acc[0]), "+l"(acc[1]), "+l"(acc[2]) // 输出操作数 (也是输入) %0, %1, %2
        : "l"(val_lo), "l"(val_hi)                 // 输入操作数 %3, %4
    );
}

// 192-bit Helper: Compare (A >= B)
__device__ bool ge_192(const uint64_t* A, uint64_t B0, uint64_t B1, uint64_t B2) {
    if (A[2] > B2) return true;
    if (A[2] < B2) return false;
    if (A[1] > B1) return true;
    if (A[1] < B1) return false;
    return A[0] >= B0;
}

// 192-bit Helper: Subtract (A -= B)
__device__ void sub_192(uint64_t* A, uint64_t B0, uint64_t B1, uint64_t B2) {
    // 同样合并为一个块
    asm volatile (
        "sub.cc.u64      %0, %0, %3; \n\t" // A[0] -= B0, set CC
        "subc.cc.u64     %1, %1, %4; \n\t" // A[1] -= B1 - Borrow, set CC
        "subc.u64        %2, %2, %5; \n\t" // A[2] -= B2 - Borrow
        : "+l"(A[0]), "+l"(A[1]), "+l"(A[2]) // %0, %1, %2
        : "l"(B0), "l"(B1), "l"(B2)          // %3, %4, %5
    );
}

// 192-bit Helper: Reverse Subtract to Buffer (Res = B - A)
// 用于计算 Q - acc
__device__ void sub_rev_192(const uint64_t* A, uint64_t B0, uint64_t B1, uint64_t B2, uint64_t* Res) {
    asm volatile (
        "sub.cc.u64      %0, %3, %4; \n\t" // Res[0] = B0 - A[0]
        "subc.cc.u64     %1, %5, %6; \n\t" // Res[1] = B1 - A[1]
        "subc.u64        %2, %7, %8; \n\t" // Res[2] = B2 - A[2]
        : "=l"(Res[0]), "=l"(Res[1]), "=l"(Res[2])       // Outputs %0-%2
        : "l"(B0), "l"(A[0]), "l"(B1), "l"(A[1]), "l"(B2), "l"(A[2]) // Inputs %3-%8
    );
}

// 2. Approx Dequantize: floating CRT (kept for reference)
__global__ void dequantize_approx_kernel(
    const uint64_t* in_real, const uint64_t* in_imag,
    cuDoubleComplex* output, int n, double delta,
    const double* crt_coeff, double q_total, double q_half, int limbs
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    auto reconstruct = [&](const uint64_t* data) -> double {
        double sum = 0.0;
        for (int k = 0; k < limbs; ++k) {
            double x = (double)data[k * n + idx];
            sum += x * crt_coeff[k];
        }
        sum = fmod(sum, q_total);
        if (sum > q_half) sum -= q_total;
        return sum / delta;
    };

    output[idx] = make_cuDoubleComplex(reconstruct(in_real), reconstruct(in_imag));
}



__global__ void mat_mul_kernel_complex(const cuDoubleComplex* A, const cuDoubleComplex* B, cuDoubleComplex* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        cuDoubleComplex sum = make_cuDoubleComplex(0, 0);
        for (int k = 0; k < n; ++k) sum = cuCadd(sum, cuCmul(A[row*n+k], B[k*n+col]));
        C[row * n + col] = sum;
    }
}

// --- Host Implementation ---
Encoder::Encoder(int n_dim) : n(n_dim) {
    size_t sz = n * n * sizeof(cuDoubleComplex);
    cudaMalloc(&d_V_cx, sz); cudaMalloc(&d_V_cx_T, sz);
    cudaMalloc(&d_V_inv_cx, sz); cudaMalloc(&d_V_inv_cx_T, sz);
    init_complex_matrices();

    static bool moduli_inited = false;
    if (!moduli_inited) {
        cudaMemcpyToSymbol(d_rns_moduli, RNS_MODULI, sizeof(RNS_MODULI));
        moduli_inited = true;
    }

    // Init exact CRT tables once
    static bool crt_inited = false;
    if (!crt_inited) {
        // Host big-int helpers (little-endian limbs)
        auto big_mul_u64_host = [](const std::vector<uint64_t>& a, uint64_t m) {
            std::vector<uint64_t> out(BIGINT_LIMBS, 0);
            unsigned __int128 carry = 0;
            for (int i = 0; i < BIGINT_LIMBS; ++i) {
                unsigned __int128 p = (unsigned __int128)a[i] * m + carry;
                out[i] = (uint64_t)p;
                carry = p >> 64;
            }
            if (carry != 0) {
                throw std::runtime_error("CRT Q overflow: increase BIGINT_LIMBS");
            }
            return out;
        };
        auto big_div_u64_host = [](const std::vector<uint64_t>& a, uint64_t d, uint64_t* rem_out) {
            std::vector<uint64_t> q(BIGINT_LIMBS, 0);
            unsigned __int128 rem = 0;
            for (int i = BIGINT_LIMBS - 1; i >= 0; --i) {
                unsigned __int128 cur = (rem << 64) | a[i];
                q[i] = (uint64_t)(cur / d);
                rem = cur % d;
            }
            if (rem_out) *rem_out = (uint64_t)rem;
            return q;
        };
        auto big_mod_u64_host = [](const std::vector<uint64_t>& a, uint64_t m) {
            unsigned __int128 rem = 0;
            for (int i = BIGINT_LIMBS - 1; i >= 0; --i) {
                unsigned __int128 cur = (rem << 64) | a[i];
                rem = cur % m;
            }
            return (uint64_t)rem;
        };
        auto big_shift_right1 = [](const std::vector<uint64_t>& a) {
            std::vector<uint64_t> out(BIGINT_LIMBS, 0);
            uint64_t carry = 0;
            for (int i = BIGINT_LIMBS - 1; i >= 0; --i) {
                uint64_t cur = a[i];
                out[i] = (cur >> 1) | (carry << 63);
                carry = cur & 1;
            }
            return out;
        };

        // Build Q
        std::vector<uint64_t> Q(BIGINT_LIMBS, 0);
        Q[0] = 1;
        for (int i = 0; i < RNS_NUM_LIMBS; ++i) {
            Q = big_mul_u64_host(Q, RNS_MODULI[i]);
        }
        std::vector<uint64_t> Q_half = big_shift_right1(Q);

        // Build M_i and inv_i
        std::vector<uint64_t> h_crt_M(RNS_NUM_LIMBS * BIGINT_LIMBS, 0);
        std::vector<uint64_t> h_crt_inv(RNS_NUM_LIMBS, 0);
        for (int i = 0; i < RNS_NUM_LIMBS; ++i) {
            uint64_t qi = RNS_MODULI[i];
            uint64_t rem = 0;
            std::vector<uint64_t> Mi = big_div_u64_host(Q, qi, &rem);
            if (rem != 0) {
                throw std::runtime_error("CRT Q not divisible by qi");
            }
            uint64_t Mi_mod_qi = big_mod_u64_host(Mi, qi);
            uint64_t inv = modInverse(Mi_mod_qi, qi);
            h_crt_inv[i] = inv;
            for (int j = 0; j < BIGINT_LIMBS; ++j) {
                h_crt_M[i * BIGINT_LIMBS + j] = Mi[j];
            }
        }

        cudaMemcpyToSymbol(d_crt_M, h_crt_M.data(),
                           h_crt_M.size() * sizeof(uint64_t));
        cudaMemcpyToSymbol(d_crt_inv, h_crt_inv.data(),
                           h_crt_inv.size() * sizeof(uint64_t));
        cudaMemcpyToSymbol(d_crt_Q, Q.data(), Q.size() * sizeof(uint64_t));
        cudaMemcpyToSymbol(d_crt_Q_half, Q_half.data(), Q_half.size() * sizeof(uint64_t));
        crt_inited = true;
    }
}
Encoder::~Encoder() { cudaFree(d_V_cx); cudaFree(d_V_cx_T); cudaFree(d_V_inv_cx); cudaFree(d_V_inv_cx_T); }

void Encoder::init_complex_matrices() {
    const double PI = 3.141592653589793;
    std::vector<cuDoubleComplex> h_V(n*n), h_V_inv(n*n);
    for (int j = 0; j < n; ++j) {
        uint64_t exp = 1, b5 = 5; int p = j;
        while(p > 0) { if(p&1) exp = (exp*b5)%(4*n); b5=(b5*b5)%(4*n); p>>=1; }
        double ang = 2.0 * PI * exp / (4.0 * n);
        cuDoubleComplex z = {cos(ang), sin(ang)}, zi = cuConj(z), c = {1,0}, ci = {1,0};
        for (int k = 0; k < n; ++k) {
            h_V[j*n+k] = c; h_V_inv[k*n+j] = cuCmul(ci, {1.0/n, 0});
            c = cuCmul(c, z); ci = cuCmul(ci, zi);
        }
    }
    std::vector<cuDoubleComplex> h_VT(n*n), h_ViT(n*n);
    for(int r=0; r<n; r++) for(int c=0; c<n; c++) { h_VT[c*n+r]=h_V[r*n+c]; h_ViT[c*n+r]=h_V_inv[r*n+c]; }
    cudaMemcpy(d_V_cx, h_V.data(), n*n*16, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V_cx_T, h_VT.data(), n*n*16, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V_inv_cx, h_V_inv.data(), n*n*16, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V_inv_cx_T, h_ViT.data(), n*n*16, cudaMemcpyHostToDevice);
}

void Encoder::encode(const cuDoubleComplex* d_msg, uint64_t* d_real_rns, uint64_t* d_imag_rns) {
    cuDoubleComplex *dT, *dP, *dM;
    cudaMalloc(&dT, n*n*16);
    cudaMalloc(&dP, n*n*16);
    cudaMalloc(&dM, n*n*16);
    dim3 blk((n+15)/16, (n+15)/16), thr(16,16);
    mat_mul_kernel_complex<<<blk, thr>>>(d_V_inv_cx, d_msg, dT, n);
    mat_mul_kernel_complex<<<blk, thr>>>(dT, d_V_inv_cx_T, dP, n);
    int threads = 256; int blocks = (n*n + 255)/256;
    quantize_soa_kernel<<<blocks, threads>>>(dP, d_real_rns, d_imag_rns, n*n, SCALING_FACTOR, RNS_NUM_LIMBS);
    cudaFree(dT); cudaFree(dP);
    cudaDeviceSynchronize();
}

void Encoder::idft2(const cuDoubleComplex* d_eval_xy, cuDoubleComplex* d_coeff_xy) {
    cuDoubleComplex* dT = nullptr;
    cudaMalloc(&dT, n * n * sizeof(cuDoubleComplex));
    dim3 blk((n + 15) / 16, (n + 15) / 16), thr(16, 16);
    mat_mul_kernel_complex<<<blk, thr>>>(d_V_inv_cx, d_eval_xy, dT, n);
    mat_mul_kernel_complex<<<blk, thr>>>(dT, d_V_inv_cx_T, d_coeff_xy, n);
    cudaFree(dT);
}

// Lane-level decode helper for encode/decode verification (not the encrypted HE decode path).
void Encoder::decode_lane_from_rns_eval(const uint64_t* d_real_rns, const uint64_t* d_imag_rns, cuDoubleComplex* d_msg) {
    // n*n*16 complex entries
    const size_t nn = (size_t)n * (size_t)n;
    const size_t elems = nn * 16;

    cuDoubleComplex* dP = nullptr;
    cudaMalloc(&dP, elems * sizeof(cuDoubleComplex));

    int threads = 256;
    int blocks  = (int)((nn + threads - 1) / threads);

    dequantize_exact_kernel<<<blocks, threads>>>(
        d_real_rns, d_imag_rns, dP, (int)nn, SCALING_FACTOR, RNS_NUM_LIMBS
    );
    cudaDeviceSynchronize();

    decode_from_eval_complex(dP, d_msg);

    cudaFree(dP);
    cudaDeviceSynchronize();
}

void Encoder::decode_from_eval_complex(const cuDoubleComplex* d_eval, cuDoubleComplex* d_msg) {
    const size_t elems = (size_t)n * (size_t)n;
    cuDoubleComplex* dT = nullptr;
    cudaMalloc(&dT, elems * sizeof(cuDoubleComplex));
    dim3 blk((n + 15) / 16, (n + 15) / 16);
    dim3 thr(16, 16);
    mat_mul_kernel_complex<<<blk, thr>>>(d_V_cx, d_eval, dT, n);
    mat_mul_kernel_complex<<<blk, thr>>>(dT, d_V_cx_T, d_msg, n);
    cudaFree(dT);
}

}
