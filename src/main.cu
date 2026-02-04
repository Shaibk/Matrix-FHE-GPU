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
    // 1) sizes
    const int n = MATRIX_N;                 // 64
    const int PHI = BATCH_SIZE;             // 512
    const int N_pack = n * PHI;             // 32768
    const int N_he   = HE_N;                // 65536 for Route A
    const int LIMBS = RNS_NUM_LIMBS;        // 11
    const int n2 = n * n;                   // 65536

    std::cout << "=== FHE Pipeline Test (Route-A Embed: [re|im]) ===\n";
    std::cout << "Matrix Size: " << n << "x" << n << "\n";
    std::cout << "Batch Size (PHI): " << PHI << "\n";
    std::cout << "Packed Degree (N_pack): " << N_pack << "\n";
    std::cout << "HE Degree (N_he): " << N_he << "\n";

    size_t total_complex_elements = (size_t)PHI * (size_t)n2;
    size_t total_pack_coeffs = (size_t)PHI * (size_t)LIMBS * (size_t)n2; // [W][Limb][n2]

    std::cout << "Total Input Complex Elements: " << total_complex_elements << "\n";
    std::cout << "Total Packed Coefficients (re or im): " << total_pack_coeffs << "\n";

    // 2) init backend & key
    std::cout << ">>> Initializing HE Backend & Keys...\n";
    init_he_backend();

    SecretKey sk;
    generate_secret_key(sk, LIMBS);
    cuda_sync("Key Gen");

    // 3) input
    std::cout << ">>> Generating Input Data...\n";
    std::vector<cuDoubleComplex> h_in(total_complex_elements);
    for (int ell = 0; ell < PHI; ++ell) {
        for (int i = 0; i < n2; ++i) {
            double val_real = (double)ell + (double)i * 0.00001;
            double val_imag = (double)ell - (double)i * 0.00001;
            h_in[ell * n2 + i] = make_cuDoubleComplex(val_real, val_imag);
        }
    }

    cuDoubleComplex* d_in;
    cuda_check(cudaMalloc(&d_in, total_complex_elements * sizeof(cuDoubleComplex)), "Malloc Input");
    cuda_check(cudaMemcpy(d_in, h_in.data(),
                          total_complex_elements * sizeof(cuDoubleComplex),
                          cudaMemcpyHostToDevice),
               "H2D Input");

    // Step A: Batched Encode -> packed re/im (N_pack each)
    std::cout << ">>> Step A: Batched Encode...\n";

    uint64_t *d_packed_re, *d_packed_im;
    cuda_check(cudaMalloc(&d_packed_re, total_pack_coeffs * sizeof(uint64_t)), "Malloc Packed Re");
    cuda_check(cudaMalloc(&d_packed_im, total_pack_coeffs * sizeof(uint64_t)), "Malloc Packed Im");

    BatchedEncoder batched_enc(n);
    batched_enc.encode_to_wntt_eval(d_in, d_packed_re, d_packed_im);
    cuda_sync("Batched Encode");

    // Step B: Encrypt (re/im separately)
    std::cout << ">>> Step B: Encrypting...\n";
    RLWECiphertext ct_re, ct_im;
    allocate_ciphertext(ct_re, LIMBS);
    allocate_ciphertext(ct_im, LIMBS);

    encrypt_pair(d_packed_re, d_packed_im, sk, ct_re, ct_im);
    cuda_sync("Encrypt");

    // Step C: Decrypt
    std::cout << ">>> Step C: Decrypting...\n";

    cuDoubleComplex *d_decrypted;
    cuda_check(cudaMalloc(&d_decrypted, total_complex_elements * sizeof(cuDoubleComplex)), "Malloc Decrypt");

    decrypt_and_decode(ct_re, ct_im, sk, d_decrypted);
    cuda_sync("Decrypt");

    // Step D: Combine re/im on host
    std::cout << ">>> Step D: Combine...\n";

    // Verification (same)
    std::cout << ">>> Verifying results...\n";
    std::vector<cuDoubleComplex> h_out(total_complex_elements);
    cuda_check(cudaMemcpy(h_out.data(), d_decrypted,
                          total_complex_elements * sizeof(cuDoubleComplex),
                          cudaMemcpyDeviceToHost),
               "D2H Decrypted");

    double max_err = 0.0;
    int err_batch = -1;
    int err_idx = -1;

    for (int i = 0; i < (int)total_complex_elements; ++i) {
        double got_r = h_out[i].x;
        double got_i = h_out[i].y;
        double ref_r = h_in[i].x;
        double ref_i = h_in[i].y;

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
    cudaFree(d_in);
    cudaFree(d_packed_re); cudaFree(d_packed_im);
    free_ciphertext(ct_re);
    free_ciphertext(ct_im);
    cudaFree(d_decrypted);
    if (max_err < 1e-4) {
        std::cout << ">>> [SUCCESS] Pipeline Verified! Data integrity preserved.\n";
        return 0;
    } else {
        std::cout << ">>> [FAILURE] Error too high. Check embedding/scale/noise logic.\n";
        return 1;
    }
}
