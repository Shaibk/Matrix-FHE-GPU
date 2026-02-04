// Encode -> Encrypt -> Decrypt -> Decode loopback test (W-CRT domain)

#include "../include/core/config.h"
#include "../include/core/encoder.cuh"
#include "../include/core/batched_encoder.cuh"
#include "../include/core/HE.cuh"

#include <cuda_runtime.h>
#include <cuComplex.h>

#include <vector>
#include <iostream>
#include <cmath>

using namespace matrix_fhe;

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

int main() {
    const int n = MATRIX_N;
    const int PHI = BATCH_SIZE;
    const int n2 = n * n;
    const int LIMBS = RNS_NUM_LIMBS;

    std::cout << "=== Test: Encode -> Encrypt -> Decrypt -> Decode (W-CRT) ===\n";
    std::cout << "n=" << n << ", PHI=" << PHI << ", limbs=" << LIMBS << "\n";

    init_he_backend();

    SecretKey sk;
    generate_secret_key(sk, LIMBS);
    cuda_sync("keygen");

    // Host input
    std::vector<cuDoubleComplex> h_in((size_t)PHI * (size_t)n2);
    for (int ell = 0; ell < PHI; ++ell) {
        for (int i = 0; i < n2; ++i) {
            double val = (double)ell + (double)i * 0.001;
            h_in[(size_t)ell * (size_t)n2 + (size_t)i] = make_cuDoubleComplex(val, -val);
        }
    }

    // Device input
    cuDoubleComplex* d_in = nullptr;
    cuda_check(cudaMalloc(&d_in, h_in.size() * sizeof(cuDoubleComplex)), "malloc input");
    cuda_check(cudaMemcpy(d_in, h_in.data(), h_in.size() * sizeof(cuDoubleComplex),
                          cudaMemcpyHostToDevice), "H2D input");

    // Encode (W-CRT layout)
    size_t packed_words = (size_t)PHI * (size_t)LIMBS * (size_t)n2;
    uint64_t* d_packed_re = nullptr;
    uint64_t* d_packed_im = nullptr;
    cuda_check(cudaMalloc(&d_packed_re, packed_words * sizeof(uint64_t)), "malloc packed re");
    cuda_check(cudaMalloc(&d_packed_im, packed_words * sizeof(uint64_t)), "malloc packed im");

    BatchedEncoder batched_enc(n);
    batched_enc.encode_packed_p17(d_in, d_packed_re, d_packed_im);
    cuda_sync("encode");

    // Encrypt/Decrypt (re/im separately)
    RLWECiphertext ct_re, ct_im;
    allocate_ciphertext(ct_re, LIMBS);
    allocate_ciphertext(ct_im, LIMBS);

    encrypt_pair(d_packed_re, d_packed_im, sk, ct_re, ct_im);
    cuda_sync("encrypt");

    cuDoubleComplex* d_dec_re = nullptr;
    cuda_check(cudaMalloc(&d_dec_re, h_in.size() * sizeof(cuDoubleComplex)), "malloc dec");

    decrypt_and_decode(ct_re, ct_im, sk, d_dec_re);
    cuda_sync("decrypt");

    // Verify in complex domain against original input
    std::vector<cuDoubleComplex> h_dec(h_in.size());
    cuda_check(cudaMemcpy(h_dec.data(), d_dec_re, h_in.size() * sizeof(cuDoubleComplex),
                          cudaMemcpyDeviceToHost), "D2H dec");

    double max_err = 0.0;
    size_t worst = 0;
    for (size_t i = 0; i < h_in.size(); ++i) {
        double err = std::hypot(h_dec[i].x - h_in[i].x, h_dec[i].y - h_in[i].y);
        if (err > max_err) {
            max_err = err;
            worst = i;
        }
    }

    std::cout << "Max error: " << std::scientific << max_err << "\n";
    std::cout << "Worst idx: " << worst << "\n";

    cudaFree(d_in);
    cudaFree(d_packed_re);
    cudaFree(d_packed_im);
    cudaFree(d_dec_re);
    free_ciphertext(ct_re);
    free_ciphertext(ct_im);

    return (max_err < 1e-3) ? 0 : 1;
}
