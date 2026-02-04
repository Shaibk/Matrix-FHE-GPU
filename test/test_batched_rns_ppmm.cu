#include "../include/core/config.h"
#include "../include/core/encoder.cuh"
#include "../include/core/batched_encoder.cuh"
#include "../include/core/batched_trace.cuh"

#include <cuda_runtime.h>
#include <cuComplex.h>

#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <cstdlib>

using namespace matrix_fhe;

// ===================== 辅助工具函数 =====================
static inline void cuda_check(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        std::cerr << "[CUDA ERROR] " << msg << ": " << cudaGetErrorString(e) << "\n";
        std::exit(1);
    }
}

static inline void cuda_sync(const char* msg) {
    cuda_check(cudaGetLastError(), msg);
    cuda_check(cudaDeviceSynchronize(), msg);
}

// 简单的模逆运算 (用于计算 Delta^-1)
static inline uint64_t h_mul_u128(uint64_t a, uint64_t b, uint64_t mod) {
    __uint128_t p = ( __uint128_t)a * b;
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

// 诊断工具
bool check_is_zero(const uint64_t* d_ptr, size_t words, const char* name) {
    std::vector<uint64_t> h_buf(words);
    cudaMemcpy(h_buf.data(), d_ptr, words * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    bool all_zero = true;
    size_t check_lim = std::min(words, (size_t)10000);
    for (size_t i = 0; i < check_lim; ++i) {
        if (h_buf[i] != 0) { all_zero = false; break; }
    }
    if (all_zero) std::cout << "[DEBUG] !!! " << name << " IS ALL ZEROS !!! (FAIL)\n";
    else std::cout << "[DEBUG] " << name << " looks OK. Sample: " << h_buf[0] << "...\n";
    return all_zero;
}

int main() {
    constexpr int PHI = BATCH_SIZE;
    const int n  = MATRIX_N;
    const int n2 = n * n;
    const int RNS_LIMBS = RNS_NUM_LIMBS;
    const double DELTA = SCALING_FACTOR / n;

    std::cout << "=== FHE Batched Trace GEMM Test (Correct Domain) ===\n";

    // ---------------------------------------------------------
    // 1. Init Host Data [关键修改部分]
    // ---------------------------------------------------------
    std::vector<cuDoubleComplex> h_A(PHI * n2);
    std::vector<cuDoubleComplex> h_Bstar(PHI * n2);

    for (int l_logical = 0; l_logical < PHI; ++l_logical) {
        // --- 矩阵 A ---
        // A 不需要经过 Map，所以不做预重排
        // 逻辑上的第 l 个矩阵直接写到物理位置 l
        auto* A_ptr = h_A.data() + l_logical * n2;

        // --- 矩阵 B* ---
        // W-CRT 域不再进行 W 维度的置换
        int l_physical_B = l_logical;
        auto* Bs_ptr = h_Bstar.data() + l_physical_B * n2;

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                // A = Identity
                A_ptr[i*n + j] = (i == j) ? make_cuDoubleComplex(1.0, 0.0) : make_cuDoubleComplex(0.0, 0.0);
                
                // Target B pattern (基于逻辑索引 l_logical 生成)
                double val_r = double(i + j + l_logical);
                double val_i = double(i - j - l_logical);
                
                // 构造 B* (共轭转置)
                // 写入到预重排后的指针 Bs_ptr 所指向的内存
                Bs_ptr[j*n + i] = make_cuDoubleComplex(val_r, -val_i); 
            }
        }
    }

    // 2. Alloc & Copy
    cuDoubleComplex *d_Amsg, *d_Bsmsg;
    cuda_check(cudaMalloc(&d_Amsg,  PHI * n2 * sizeof(cuDoubleComplex)), "malloc Amsg");
    cuda_check(cudaMalloc(&d_Bsmsg, PHI * n2 * sizeof(cuDoubleComplex)), "malloc Bsmsg");
    cuda_check(cudaMemcpy(d_Amsg,  h_A.data(), PHI * n2 * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice), "H2D A");
    cuda_check(cudaMemcpy(d_Bsmsg, h_Bstar.data(), PHI * n2 * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice), "H2D B*");

    // 3. Encode (Packed Domain)
    size_t packed_words = (size_t)PHI * RNS_LIMBS * n2;
    uint64_t *d_A_packed_re, *d_A_packed_im;
    uint64_t *d_Bs_packed_re, *d_Bs_packed_im;

    cuda_check(cudaMalloc(&d_A_packed_re,  packed_words*sizeof(uint64_t)), "malloc A packed");
    cuda_check(cudaMalloc(&d_A_packed_im,  packed_words*sizeof(uint64_t)), "malloc A packed");
    cuda_check(cudaMalloc(&d_Bs_packed_re, packed_words*sizeof(uint64_t)), "malloc Bs packed");
    cuda_check(cudaMalloc(&d_Bs_packed_im, packed_words*sizeof(uint64_t)), "malloc Bs packed");

    BatchedEncoder batched_enc(n);
    batched_enc.encode_to_wntt_eval(d_Amsg,  d_A_packed_re,  d_A_packed_im);
    batched_enc.encode_to_wntt_eval(d_Bsmsg, d_Bs_packed_re, d_Bs_packed_im);
    cuda_sync("Encode Packed");

    // 4. Convert Packed -> Eval (W-NTT)
    uint64_t *d_A_eval_re, *d_A_eval_im;
    uint64_t *d_Bs_eval_re, *d_Bs_eval_im;
    uint64_t *d_Bp_eval_re, *d_Bp_eval_im;
    uint64_t *d_C_eval_re, *d_C_eval_im;

    cuda_check(cudaMalloc(&d_A_eval_re,   packed_words*sizeof(uint64_t)), "malloc A eval");
    cuda_check(cudaMalloc(&d_A_eval_im,   packed_words*sizeof(uint64_t)), "malloc A eval");
    cuda_check(cudaMalloc(&d_Bs_eval_re,  packed_words*sizeof(uint64_t)), "malloc Bs eval");
    cuda_check(cudaMalloc(&d_Bs_eval_im,  packed_words*sizeof(uint64_t)), "malloc Bs eval");
    cuda_check(cudaMalloc(&d_Bp_eval_re,  packed_words*sizeof(uint64_t)), "malloc Bp eval");
    cuda_check(cudaMalloc(&d_Bp_eval_im,  packed_words*sizeof(uint64_t)), "malloc Bp eval");
    cuda_check(cudaMalloc(&d_C_eval_re,   packed_words*sizeof(uint64_t)), "malloc C eval");
    cuda_check(cudaMalloc(&d_C_eval_im,   packed_words*sizeof(uint64_t)), "malloc C eval");

    // Unpack: Packed -> Eval
    batched_enc.unpack_eval_p17(d_A_packed_re, d_A_packed_im, d_A_eval_re, d_A_eval_im);
    batched_enc.unpack_eval_p17(d_Bs_packed_re, d_Bs_packed_im, d_Bs_eval_re, d_Bs_eval_im);
    cuda_sync("Unpack/Eval Conversion");

    if (check_is_zero(d_A_eval_re, packed_words, "A (Eval Domain)")) return 1;

    // 5. Map B* -> B' (Eval Domain)
    map_B_to_Bprime_batched(
        d_Bs_eval_re, d_Bs_eval_im, 
        d_Bp_eval_re, d_Bp_eval_im, 
        n, RNS_LIMBS, PHI
    );
    cuda_sync("Map Finished");

    if (check_is_zero(d_Bp_eval_re, packed_words, "Mapped B (Eval Domain)")) return 1;

    // 6. Trace GEMM (Eval Domain)
    trace_gemm_batched(
        d_A_eval_re, d_A_eval_im, 
        d_Bp_eval_re, d_Bp_eval_im, 
        d_C_eval_re, d_C_eval_im, 
        n, RNS_LIMBS, PHI
    );
    cuda_sync("GEMM Finished");

    if (check_is_zero(d_C_eval_re, packed_words, "GEMM Result (Eval Domain)")) return 1;



    // 8. Decode (Eval -> Host)
    std::vector<cuDoubleComplex> h_C_out(PHI * n2);
    cuDoubleComplex* d_C_out_buf;
    cuda_check(cudaMalloc(&d_C_out_buf, PHI * n2 * sizeof(cuDoubleComplex)), "malloc Cout");

    Encoder single_enc(n);
    size_t blk_words = (size_t)RNS_LIMBS * n2;
    
    // d_C_eval_re 已经在 Eval 域了，直接 Decode
    for (int ell = 0; ell < PHI; ++ell) {
        single_enc.decode_lane_from_rns_eval(
            d_C_eval_re + ell * blk_words,
            d_C_eval_im + ell * blk_words,
            d_C_out_buf + ell * n2
        );
    }
    cuda_sync("Decode Finished");
    
    cudaMemcpy(h_C_out.data(), d_C_out_buf, PHI * n2 * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

    // 9. Verify (Batch 0)
    std::cout << "\n>>> Verification (Checking Batch 0)...\n";
    double max_err = 0.0;
    int ell = 0;
    
    int pts[3][2] = {{0,0}, {0,1}, {10,5}};
    for(auto& p : pts) {
        int r = p[0], c = p[1];
        size_t idx = ell * n2 + r * n + c;
        cuDoubleComplex got = h_C_out[idx];
        got.x /= DELTA; got.y /= DELTA;
        double exp_r = double(r + c + ell);
        double exp_i = double(r - c - ell);
        
        std::cout << "Pos (" << r << "," << c << ") Exp: " << exp_r << "+" << exp_i << "i | Got: " << got.x << "+" << got.y << "i\n";
    }

    for(int i=0; i<n2; ++i) {
        cuDoubleComplex got = h_C_out[i];
        int r = i / n, c = i % n;
        double exp_r = double(r + c + ell);
        double exp_i = double(r - c - ell);
        double err = std::hypot((got.x/DELTA) - exp_r, (got.y/DELTA) - exp_i);
        if(err > max_err) max_err = err;
    }
    
    std::cout << "\nGlobal Max Error (Batch 0): " << std::scientific << max_err << "\n";
    // =========================================================
    // 10. Final Success/Failure Check
    // =========================================================
    // 设置一个合理的误差阈值。
    // 由于是浮点数运算且经过了多次变换，通常设定在 1e-5 或 1e-4 左右。
    const double TOLERANCE = 1e-5; 

    if (max_err < TOLERANCE) {
        std::cout << ">>> [SUCCESS] Test Passed! Max error is within tolerance (" << TOLERANCE << ").\n";
    } else {
        std::cout << ">>> [FAILURE] Test Failed! Max error (" << max_err << ") exceeds tolerance.\n";
    }
    // Cleanup
    cudaFree(d_Amsg); cudaFree(d_Bsmsg); cudaFree(d_C_out_buf);
    cudaFree(d_A_packed_re); cudaFree(d_A_packed_im);
    cudaFree(d_Bs_packed_re); cudaFree(d_Bs_packed_im);
    cudaFree(d_A_eval_re); cudaFree(d_A_eval_im);
    cudaFree(d_Bs_eval_re); cudaFree(d_Bs_eval_im);
    cudaFree(d_Bp_eval_re); cudaFree(d_Bp_eval_im);
    cudaFree(d_C_eval_re); cudaFree(d_C_eval_im);
    
    return (max_err < 1e-2) ? 0 : 1;
}
