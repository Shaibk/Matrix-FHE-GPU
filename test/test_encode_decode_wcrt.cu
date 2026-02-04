// Encode -> Decode loopback test (W-CRT domain)

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

    std::cout << "=== Test: Encode -> Decode (W-CRT) ===\n";
    std::cout << "n=" << n << ", PHI=" << PHI << ", limbs=" << LIMBS << "\n";

    // Host input
    std::vector<cuDoubleComplex> h_in((size_t)PHI * (size_t)n2);
    for (int ell = 0; ell < PHI; ++ell) {
        for (int i = 0; i < n2; ++i) {
            double val = (double)(ell * 10000 + i);
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
    batched_enc.encode_to_wntt_eval(d_in, d_packed_re, d_packed_im);
    cuda_sync("encode");

    // Decode through the same packed W-NTT -> (W-INTT + compose/scale + W-DFT + XY-DFT) path.
    cuDoubleComplex* d_out = nullptr;
    cuda_check(cudaMalloc(&d_out, h_in.size() * sizeof(cuDoubleComplex)), "malloc out");

    RLWECiphertext ct_re, ct_im;
    allocate_ciphertext(ct_re, LIMBS);
    allocate_ciphertext(ct_im, LIMBS);
    size_t total_coeffs = packed_words;
    uint64_t* ct_re_b = ct_re.data;
    uint64_t* ct_re_a = ct_re.data + total_coeffs;
    uint64_t* ct_im_b = ct_im.data;
    uint64_t* ct_im_a = ct_im.data + total_coeffs;
    cuda_check(cudaMemcpy(ct_re_b, d_packed_re, total_coeffs * sizeof(uint64_t), cudaMemcpyDeviceToDevice), "copy packed re->ct b");
    cuda_check(cudaMemcpy(ct_im_b, d_packed_im, total_coeffs * sizeof(uint64_t), cudaMemcpyDeviceToDevice), "copy packed im->ct b");
    cuda_check(cudaMemset(ct_re_a, 0, total_coeffs * sizeof(uint64_t)), "zero ct re a");
    cuda_check(cudaMemset(ct_im_a, 0, total_coeffs * sizeof(uint64_t)), "zero ct im a");

    SecretKey sk_zero{};
    sk_zero.num_limbs = LIMBS;
    cuda_check(cudaMalloc(&sk_zero.data, (size_t)PHI * (size_t)LIMBS * (size_t)n * sizeof(uint64_t)), "malloc zero sk");
    cuda_check(cudaMemset(sk_zero.data, 0, (size_t)PHI * (size_t)LIMBS * (size_t)n * sizeof(uint64_t)), "zero sk");

    decrypt_and_decode(ct_re, ct_im, sk_zero, d_out);
    cuda_sync("decode");

    // Verify
    std::vector<cuDoubleComplex> h_out(h_in.size());
    cuda_check(cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(cuDoubleComplex),
                          cudaMemcpyDeviceToHost), "D2H out");

    double max_err = 0.0;
    size_t worst = 0;
    for (size_t i = 0; i < h_in.size(); ++i) {
        double err = std::hypot(h_out[i].x - h_in[i].x, h_out[i].y - h_in[i].y);
        if (err > max_err) {
            max_err = err;
            worst = i;
        }
    }

    std::cout << "Max error: " << std::scientific << max_err << "\n";
    std::cout << "Worst idx: " << worst << "\n";

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_packed_re);
    cudaFree(d_packed_im);
    free_ciphertext(ct_re);
    free_ciphertext(ct_im);
    cudaFree(sk_zero.data);

    return (max_err < 1e-3) ? 0 : 1;
}
