#include "../../include/core/encoder.cuh"
#include "../../include/core/config.h"
#include "../../include/backend/phantom_math.cuh"
#include <iostream>
#include <vector>
#include <cmath>

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

__global__ void quantize_soa_kernel(const cuDoubleComplex* input, 
                                  uint64_t* out_real, uint64_t* out_imag, 
                                  int n, double delta, 
                                  uint64_t q0, uint64_t q1, uint64_t q2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int64_t ix = llround(cuCreal(input[idx]) * delta);
        int64_t iy = llround(cuCimag(input[idx]) * delta);
        uint64_t m[3] = {q0, q1, q2};
        #pragma unroll
        for(int k=0; k<3; k++) {
            int out_idx = k * n + idx;
            int64_t mx = ix % (int64_t)m[k]; if(mx < 0) mx += m[k]; out_real[out_idx] = mx;
            int64_t my = iy % (int64_t)m[k]; if(my < 0) my += m[k]; out_imag[out_idx] = my;
        }
    }
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

// 2. Exact Dequantize: 192-bit Integer CRT
__global__ void dequantize_exact_kernel(
    const uint64_t* in_real, const uint64_t* in_imag,
    cuDoubleComplex* output, int n, double delta,
    // CRT Parameters
    uint64_t c0, uint64_t c1, uint64_t c2,
    uint64_t c0_sh, uint64_t c1_sh, uint64_t c2_sh,
    uint64_t q0, uint64_t q1, uint64_t q2,
    // Q/qi
    uint64_t Q0_L, uint64_t Q0_H,
    uint64_t Q1_L, uint64_t Q1_H,
    uint64_t Q2_L, uint64_t Q2_H,
    // Total Q (192-bit)
    uint64_t Q_L, uint64_t Q_M, uint64_t Q_H
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // 常量定义
    const double TWO_64  = 18446744073709551616.0; // 2^64
    const double TWO_128 = TWO_64 * TWO_64;        // 2^128

    // Helper: 192-bit -> double conversion (debug/approx)
    auto to_double = [&](const uint64_t v[3]) -> double {
        return (double)v[2] * TWO_128 + (double)v[1] * TWO_64 + (double)v[0];
    };

    auto reconstruct = [&](const uint64_t* data) -> double {
        uint64_t x0 = data[0 * n + idx];
        uint64_t x1 = data[1 * n + idx];
        uint64_t x2 = data[2 * n + idx];

        uint64_t acc[3] = {0, 0, 0};

        // Term 0
        uint64_t v0 = multiply_and_reduce_shoup(x0, c0, c0_sh, q0);
        unsigned __int128 p0 = (unsigned __int128)v0 * Q0_L;
        add_to_192(acc, (uint64_t)p0, (uint64_t)(p0 >> 64));
        p0 = (unsigned __int128)v0 * Q0_H;
        asm("add.cc.u64 %0, %0, %1;" : "+l"(acc[1]) : "l"((uint64_t)p0));
        asm("addc.u64 %0, %0, %1;"   : "+l"(acc[2]) : "l"((uint64_t)(p0 >> 64)));

        // Term 1
        uint64_t v1 = multiply_and_reduce_shoup(x1, c1, c1_sh, q1);
        unsigned __int128 p1 = (unsigned __int128)v1 * Q1_L;
        add_to_192(acc, (uint64_t)p1, (uint64_t)(p1 >> 64));
        p1 = (unsigned __int128)v1 * Q1_H;
        asm("add.cc.u64 %0, %0, %1;" : "+l"(acc[1]) : "l"((uint64_t)p1));
        asm("addc.u64 %0, %0, %1;"   : "+l"(acc[2]) : "l"((uint64_t)(p1 >> 64)));

        // Term 2
        uint64_t v2 = multiply_and_reduce_shoup(x2, c2, c2_sh, q2);
        unsigned __int128 p2 = (unsigned __int128)v2 * Q2_L;
        add_to_192(acc, (uint64_t)p2, (uint64_t)(p2 >> 64));
        p2 = (unsigned __int128)v2 * Q2_H;
        asm("add.cc.u64 %0, %0, %1;" : "+l"(acc[1]) : "l"((uint64_t)p2));
        asm("addc.u64 %0, %0, %1;"   : "+l"(acc[2]) : "l"((uint64_t)(p2 >> 64)));

        // Modulo Reduction: while (acc >= Q) acc -= Q;
        while (ge_192(acc, Q_L, Q_M, Q_H)) {
            sub_192(acc, Q_L, Q_M, Q_H);
        }

        // Center Lift: if (acc >= Q/2) treat as negative
        uint64_t Q_half_L = (Q_L >> 1) | ((Q_M & 1ULL) << 63);
        uint64_t Q_half_M = (Q_M >> 1) | ((Q_H & 1ULL) << 63);
        uint64_t Q_half_H = (Q_H >> 1);

        bool neg = ge_192(acc, Q_half_L, Q_half_M, Q_half_H);

        // =========================
        // Strong debug print: only once
        // =========================
        if (idx == 0) {
            if (atomicCAS(&d_dbg_once, 0, 1) == 0) {
                printf("\n[DBG] ---- dequantize_exact_kernel idx=0 ----\n");
                printf("[DBG] x0,x1,x2 = %llu, %llu, %llu\n",
                       (unsigned long long)x0,
                       (unsigned long long)x1,
                       (unsigned long long)x2);

                printf("[DBG] acc (mod Q) limbs = %llu, %llu, %llu\n",
                       (unsigned long long)acc[0],
                       (unsigned long long)acc[1],
                       (unsigned long long)acc[2]);

                printf("[DBG] Q limbs   = %llu, %llu, %llu\n",
                       (unsigned long long)Q_L,
                       (unsigned long long)Q_M,
                       (unsigned long long)Q_H);

                printf("[DBG] Q/2 limbs = %llu, %llu, %llu\n",
                       (unsigned long long)Q_half_L,
                       (unsigned long long)Q_half_M,
                       (unsigned long long)Q_half_H);

                printf("[DBG] acc >= Q/2 ? %d\n", (int)neg);

                if (neg) {
                    // diff = Q - acc (magnitude of negative)
                    uint64_t diff[3];
                    sub_rev_192(acc, Q_L, Q_M, Q_H, diff); // diff = Q - acc
                    printf("[DBG] diff=Q-acc = %llu, %llu, %llu\n",
                           (unsigned long long)diff[0],
                           (unsigned long long)diff[1],
                           (unsigned long long)diff[2]);

                    double mag = to_double(diff);
                    printf("[DBG] signed_int approx = -%.6e\n", mag);
                    printf("[DBG] signed_int/delta approx = -%.6e\n", mag / delta);
                } else {
                    double val = to_double(acc);
                    printf("[DBG] signed_int approx = +%.6e\n", val);
                    printf("[DBG] signed_int/delta approx = +%.6e\n", val / delta);
                }

                printf("[DBG] delta = %.6e\n", delta);
                printf("[DBG] -------------------------------------\n\n");
            }
        }

        double val_double = 0.0;
        if (neg) {
            uint64_t diff[3];
            sub_rev_192(acc, Q_L, Q_M, Q_H, diff);  // diff = Q - acc
            val_double = -to_double(diff);
        } else {
            val_double = to_double(acc);
        }

        return val_double / delta;
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
    cuDoubleComplex *dT, *dP; cudaMalloc(&dT, n*n*16); cudaMalloc(&dP, n*n*16);
    dim3 blk((n+15)/16, (n+15)/16), thr(16,16);
    mat_mul_kernel_complex<<<blk, thr>>>(d_V_inv_cx, d_msg, dT, n);
    mat_mul_kernel_complex<<<blk, thr>>>(dT, d_V_inv_cx_T, dP, n);
    int threads = 256; int blocks = (n*n + 255)/256;
    quantize_soa_kernel<<<blocks, threads>>>(dP, d_real_rns, d_imag_rns, n*n, SCALING_FACTOR, 
                                           RNS_MODULI[0], RNS_MODULI[1], RNS_MODULI[2]);
    cudaFree(dT); cudaFree(dP);
    cudaDeviceSynchronize();
}

void Encoder::decode(const uint64_t* d_real_rns, const uint64_t* d_imag_rns, cuDoubleComplex* d_msg) {
    // n*n*16 complex entries
    const size_t nn = (size_t)n * (size_t)n;
    const size_t elems = nn * 16;

    cuDoubleComplex* dP = nullptr;
    cudaMalloc(&dP, elems * sizeof(cuDoubleComplex));

    uint64_t q0 = RNS_MODULI[0], q1 = RNS_MODULI[1], q2 = RNS_MODULI[2];

    // 128-bit Q_star
    unsigned __int128 Q0 = (unsigned __int128)q1 * q2;
    unsigned __int128 Q1 = (unsigned __int128)q0 * q2;
    unsigned __int128 Q2 = (unsigned __int128)q0 * q1;

    uint64_t Q0_L = (uint64_t)Q0; uint64_t Q0_H = (uint64_t)(Q0 >> 64);
    uint64_t Q1_L = (uint64_t)Q1; uint64_t Q1_H = (uint64_t)(Q1 >> 64);
    uint64_t Q2_L = (uint64_t)Q2; uint64_t Q2_H = (uint64_t)(Q2 >> 64);

    uint64_t c0 = modInverse((uint64_t)(Q0 % q0), q0);
    uint64_t c1 = modInverse((uint64_t)(Q1 % q1), q1);
    uint64_t c2 = modInverse((uint64_t)(Q2 % q2), q2);

    uint64_t c0_sh = compute_shoup(c0, q0);
    uint64_t c1_sh = compute_shoup(c1, q1);
    uint64_t c2_sh = compute_shoup(c2, q2);

    // Q = q0*q1*q2 as 192-bit split (Q_L, Q_M, Q_H)
    unsigned __int128 term_lo = (unsigned __int128)Q0_L * q0;
    unsigned __int128 term_hi = (unsigned __int128)Q0_H * q0;

    uint64_t Q_L = (uint64_t)term_lo;
    uint64_t term_lo_high = (uint64_t)(term_lo >> 64);
    uint64_t term_hi_low  = (uint64_t)term_hi;

    uint64_t Q_M = term_lo_high + term_hi_low;
    uint64_t carry = (Q_M < term_lo_high) ? 1 : 0;
    uint64_t Q_H = (uint64_t)(term_hi >> 64) + carry;

    int threads = 256;
    int blocks  = (int)((nn + threads - 1) / threads);

    dequantize_exact_kernel<<<blocks, threads>>>(
        d_real_rns, d_imag_rns,
        dP,
        (int)nn,
        SCALING_FACTOR,
        c0, c1, c2,
        c0_sh, c1_sh, c2_sh,
        q0, q1, q2,
        Q0_L, Q0_H,
        Q1_L, Q1_H,
        Q2_L, Q2_H,
        Q_L, Q_M, Q_H
    );
    cudaDeviceSynchronize();

    cuDoubleComplex* dT = nullptr;
    cudaMalloc(&dT, elems * sizeof(cuDoubleComplex));

    dim3 blk((n + 15) / 16, (n + 15) / 16);
    dim3 thr(16, 16);
    mat_mul_kernel_complex<<<blk, thr>>>(d_V_cx,   dP, dT,    n);
    mat_mul_kernel_complex<<<blk, thr>>>(dT, d_V_cx_T, d_msg, n);

    cudaFree(dP);
    cudaFree(dT);
    cudaDeviceSynchronize();
}

}