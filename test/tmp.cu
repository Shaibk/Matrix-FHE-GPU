#include "../include/core/config.h"
#include "../include/core/common.cuh"
#include "../include/core/encoder.cuh"
#include "../include/core/batched_encoder.cuh"
#include "../include/core/HE.cuh" 

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>

using namespace matrix_fhe;

// ===================== 辅助工具 =====================
static void cuda_check(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        std::cerr << "[CUDA ERROR] " << msg << ": " << cudaGetErrorString(e) << "\n";
        exit(1);
    }
}

static void cuda_sync(const char* msg) {
    cuda_check(cudaGetLastError(), msg);
    cuda_check(cudaDeviceSynchronize(), msg);
}

// ===================== Main Pipeline Test =====================
int main() {
    // 1. Configuration from config.h
    const int n = MATRIX_N;           // 256
    const int PHI = BATCH_PRIME_P - 1;               // Batch size (p-1)
    const int N_poly = n * PHI;        // 8192 (Base Ring Degree)
    const int LIMBS = RNS_NUM_LIMBS;  // 3
    const int n2 = n * n;             // 65536

    std::cout << "=== FHE Pipeline Test (Poly-Major Layout) ===\n";
    std::cout << "Matrix Size: " << n << "x" << n << "\n";
    std::cout << "Batch Size (PHI): " << PHI << "\n";
    std::cout << "Polynomial Degree (N): " << N_poly << "\n";
    
    // Calculate total sizes
    size_t total_complex_elements = (size_t)PHI * n2;
    size_t total_poly_coeffs = (size_t)n * LIMBS * N_poly; // [Y][Limb][X]
    
    std::cout << "Total Input Complex Elements: " << total_complex_elements << "\n";
    std::cout << "Total Encoded Coefficients: " << total_poly_coeffs << "\n";

    // 2. Initialize HE Backend & Keys
    std::cout << ">>> Initializing HE Backend & Keys...\n";
    init_he_backend(); 

    SecretKey sk;
    generate_secret_key(sk, LIMBS);
    cuda_sync("Key Gen");

    // 3. Prepare Input Data (Host)
    // 生成 16 个随机复数矩阵
    // Host Layout: [Batch][Row][Col] (Flat as [PHI][n*n])
    std::cout << ">>> Generating Input Data...\n";
    std::vector<cuDoubleComplex> h_in(total_complex_elements);
    for (int ell = 0; ell < PHI; ++ell) {
        for (int i = 0; i < n2; ++i) {
            // Pattern: Batch.Index
            double val_real = (double)ell + (double)i * 0.00001;
            double val_imag = (double)ell - (double)i * 0.00001;
            h_in[ell * n2 + i] = make_cuDoubleComplex(val_real, val_imag);
        }
    }

    // Copy to Device
    cuDoubleComplex* d_in;
    cuda_check(cudaMalloc(&d_in, total_complex_elements * sizeof(cuDoubleComplex)), "Malloc Input");
    cuda_check(cudaMemcpy(d_in, h_in.data(), total_complex_elements * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice), "H2D Input");

    // ---------------------------------------------------------
    // Step A: Batched Encode (Input -> Poly-Major Coeffs)
    // ---------------------------------------------------------
    // Output Layout: [Y=n][Limbs][X=RLWE_N]
    std::cout << ">>> Step A: Batched Encode...\n";
    
    uint64_t *d_packed_re, *d_packed_im;
    cuda_check(cudaMalloc(&d_packed_re, total_poly_coeffs * sizeof(uint64_t)), "Malloc Packed Re");
    cuda_check(cudaMalloc(&d_packed_im, total_poly_coeffs * sizeof(uint64_t)), "Malloc Packed Im");

    BatchedEncoder batched_enc(n);
    // 此函数内部已包含 Single Encode + Transpose Kernel
    batched_enc.encode_packed_p17(d_in, d_packed_re, d_packed_im);
    cuda_sync("Batched Encode");

    // ---------------------------------------------------------
    // Step B: Encrypt (Poly-Major)
    // ---------------------------------------------------------
    std::cout << ">>> Step B: Encrypting...\n";

    RLWECiphertext ct_re, ct_im;
    allocate_ciphertext(ct_re, LIMBS); 
    allocate_ciphertext(ct_im, LIMBS);

    // encrypt 函数现在能够处理 [Y][Limbs][X] 布局
    // 并且会正确地对 n 个多项式分别调用 Phantom 的 NTT
    encrypt(d_packed_re, sk, ct_re);
    encrypt(d_packed_im, sk, ct_im);
    cuda_sync("Encrypt");

    // ---------------------------------------------------------
    // Step C: Decrypt (Poly-Major)
    // ---------------------------------------------------------
    std::cout << ">>> Step C: Decrypting...\n";

    uint64_t *d_decrypted_re, *d_decrypted_im;
    cuda_check(cudaMalloc(&d_decrypted_re, total_poly_coeffs * sizeof(uint64_t)), "Malloc Decrypt Re");
    cuda_check(cudaMalloc(&d_decrypted_im, total_poly_coeffs * sizeof(uint64_t)), "Malloc Decrypt Im");

    decrypt(ct_re, sk, d_decrypted_re);
    decrypt(ct_im, sk, d_decrypted_im);
    cuda_sync("Decrypt");

    // ---------------------------------------------------------
    // Step D: Decode (Poly-Major Coeffs -> Complex Matrices)
    // ---------------------------------------------------------
    std::cout << ">>> Step D: Decode...\n";
    
    // 1. Unpack (Poly-Major -> Batch-Major Eval RNS)
    // d_eval buffer size: PHI * LIMBS * n^2
    // Note: n^2 (65536) might be larger than RLWE_N (8192), 
    // but SingleEncoder works on n^2 slots in RNS domain.
    // unpack_eval_p17 handles the mapping from packed coefficients to RNS slots.
    size_t rns_eval_size = (size_t)PHI * LIMBS * n2;
    uint64_t *d_eval_re, *d_eval_im;
    cuda_check(cudaMalloc(&d_eval_re, rns_eval_size * sizeof(uint64_t)), "Malloc Eval Re");
    cuda_check(cudaMalloc(&d_eval_im, rns_eval_size * sizeof(uint64_t)), "Malloc Eval Im");

    batched_enc.unpack_eval_p17(d_decrypted_re, d_decrypted_im, d_eval_re, d_eval_im);
    cuda_sync("Unpack");
    
    // 2. Single Decode (RNS Slots -> Complex)
    cuDoubleComplex* d_out;
    cuda_check(cudaMalloc(&d_out, total_complex_elements * sizeof(cuDoubleComplex)), "Malloc Out");

    Encoder single_enc(n);
    size_t batch_rns_stride = (size_t)LIMBS * n2;
    size_t batch_complex_stride = (size_t)n2;
    
    for (int ell = 0; ell < PHI; ++ell) {
        single_enc.decode(
            d_eval_re + ell * batch_rns_stride,
            d_eval_im + ell * batch_rns_stride,
            d_out + ell * batch_complex_stride
        );
    }
    cuda_sync("Single Decode");

    // ---------------------------------------------------------
    // Verification
    // ---------------------------------------------------------
    std::cout << ">>> Verifying results...\n";
    std::vector<cuDoubleComplex> h_out(total_complex_elements);
    cuda_check(cudaMemcpy(h_out.data(), d_out, total_complex_elements * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost), "D2H Out");

    double max_err = 0.0;
    int err_batch = -1;
    int err_idx = -1;

    // 检查所有数据
    for (int i = 0; i < total_complex_elements; ++i) {
        double got_r = h_out[i].x;
        double got_i = h_out[i].y;
        
        double ref_r = h_in[i].x;
        double ref_i = h_in[i].y;

        // 注意：Encoder::decode 内部的 dequantize_exact_kernel 通常已经除以了 Delta
        // 如果误差巨大 (1e12级别)，说明没有除。
        // 根据之前的代码审查，dequantize_exact_kernel 包含 `return final_val / delta;`
        // 所以这里直接比较。

        double err = std::hypot(got_r - ref_r, got_i - ref_i);
        if (err > max_err) {
            max_err = err;
            err_idx = i;
            err_batch = i / n2;
        }
    }

    std::cout << "Global Max Error: " << std::scientific << max_err << "\n";
    
    if (err_batch != -1) {
        int local_idx = err_idx % n2;
        std::cout << "Worst case at Batch " << err_batch << ", Index " << local_idx << "\n";
        std::cout << "  Exp: " << h_in[err_idx].x << " + " << h_in[err_idx].y << "i\n";
        std::cout << "  Got: " << h_out[err_idx].x << " + " << h_out[err_idx].y << "i\n";
    }

    // Cleanup
    cudaFree(d_in); cudaFree(d_out);
    cudaFree(d_packed_re); cudaFree(d_packed_im);
    cudaFree(d_decrypted_re); cudaFree(d_decrypted_im);
    cudaFree(d_eval_re); cudaFree(d_eval_im);
    free_ciphertext(ct_re);
    free_ciphertext(ct_im);

    // 阈值判定
    // 经历 Encode -> Encrypt -> Decrypt -> Decode，包含多次 RNS 变换和 FFT
    // 1e-4 到 1e-5 是合理的误差范围
    if (max_err < 1e-4) {
        std::cout << ">>> [SUCCESS] Pipeline Verified! Data integrity preserved.\n";
        return 0;
    } else {
        std::cout << ">>> [FAILURE] Error too high. Check scaling factors or kernel logic.\n";
        return 1;
    }
}