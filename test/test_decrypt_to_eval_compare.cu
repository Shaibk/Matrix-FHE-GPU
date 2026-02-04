#include "../include/core/config.h"
#include "../include/core/batched_encoder.cuh"
#include "../include/core/encoder.cuh"
#include "../include/core/HE.cuh"

#include <cuda_runtime.h>
#include <cuComplex.h>

#include <vector>
#include <iostream>

using namespace matrix_fhe;

static inline void cuda_check(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        std::cerr << "[CUDA ERROR] " << msg << ": " << cudaGetErrorString(e) << "\n";
        std::exit(1);
    }
}

int main() {
    std::cout << "=== Test: decrypt_to_eval vs encoded packed ===\n";
    init_he_backend();

    const int n = MATRIX_N;
    const int phi = BATCH_SIZE;
    const int limbs = RNS_NUM_LIMBS;
    const int n2 = n * n;
    const size_t msg_count = (size_t)phi * (size_t)n2;
    const size_t packed_words = (size_t)phi * (size_t)limbs * (size_t)n2;

    SecretKey sk;
    generate_secret_key(sk, limbs);

    std::vector<cuDoubleComplex> h_in(msg_count);
    for (int ell = 0; ell < phi; ++ell) {
        for (int i = 0; i < n2; ++i) {
            double v = (double)ell + (double)i * 0.001;
            h_in[(size_t)ell * (size_t)n2 + (size_t)i] = make_cuDoubleComplex(v, -v);
        }
    }

    cuDoubleComplex* d_in = nullptr;
    uint64_t* d_packed_re = nullptr;
    uint64_t* d_packed_im = nullptr;
    cudaMalloc(&d_in, msg_count * sizeof(cuDoubleComplex));
    cudaMalloc(&d_packed_re, packed_words * sizeof(uint64_t));
    cudaMalloc(&d_packed_im, packed_words * sizeof(uint64_t));
    cudaMemcpy(d_in, h_in.data(), msg_count * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

    BatchedEncoder be(n);
    be.encode_to_wntt_eval(d_in, d_packed_re, d_packed_im);

    RLWECiphertext ct_re, ct_im;
    allocate_ciphertext(ct_re, limbs);
    allocate_ciphertext(ct_im, limbs);
    encrypt_pair(d_packed_re, d_packed_im, sk, ct_re, ct_im);

    uint64_t* d_eval_re = nullptr;
    uint64_t* d_eval_im = nullptr;
    cudaMalloc(&d_eval_re, packed_words * sizeof(uint64_t));
    cudaMalloc(&d_eval_im, packed_words * sizeof(uint64_t));
    decrypt_to_eval_matrix(ct_re, sk, d_eval_re);
    decrypt_to_eval_matrix(ct_im, sk, d_eval_im);
    cudaDeviceSynchronize();

    std::vector<uint64_t> h_ref_re(packed_words), h_ref_im(packed_words);
    std::vector<uint64_t> h_got_re(packed_words), h_got_im(packed_words);
    cudaMemcpy(h_ref_re.data(), d_packed_re, packed_words * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ref_im.data(), d_packed_im, packed_words * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_got_re.data(), d_eval_re, packed_words * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_got_im.data(), d_eval_im, packed_words * sizeof(uint64_t), cudaMemcpyDeviceToHost);

    uint64_t max_err = 0;
    size_t worst = 0;
    for (size_t i = 0; i < packed_words; ++i) {
        int limb = (int)((i / (size_t)n2) % (size_t)limbs);
        uint64_t q = RNS_MODULI[limb];
        auto dist = [q](uint64_t a, uint64_t b) {
            uint64_t x = a % q, y = b % q;
            uint64_t d = (x >= y) ? (x - y) : (y - x);
            return (d > q - d) ? (q - d) : d;
        };
        uint64_t dr = dist(h_ref_re[i], h_got_re[i]);
        uint64_t di = dist(h_ref_im[i], h_got_im[i]);
        uint64_t d = (dr > di) ? dr : di;
        if (d > max_err) {
            max_err = d;
            worst = i;
        }
    }

    std::cout << "max_err(mod q)=" << max_err << ", worst_idx=" << worst << "\n";

    // Extra check on same test: decode directly from decrypt_to_eval output (no extra post-processing)
    cuDoubleComplex* d_dec_direct = nullptr;
    cudaMalloc(&d_dec_direct, msg_count * sizeof(cuDoubleComplex));
    Encoder single_enc(n);
    const size_t lane_stride = (size_t)limbs * (size_t)n2;
    for (int ell = 0; ell < phi; ++ell) {
        const uint64_t* lane_re = d_eval_re + (size_t)ell * lane_stride;
        const uint64_t* lane_im = d_eval_im + (size_t)ell * lane_stride;
        single_enc.decode_lane_from_rns_eval(lane_re, lane_im, d_dec_direct + (size_t)ell * (size_t)n2);
    }
    cudaDeviceSynchronize();

    std::vector<cuDoubleComplex> h_dec_direct(msg_count);
    cudaMemcpy(h_dec_direct.data(), d_dec_direct, msg_count * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    double max_err_direct = 0.0;
    size_t worst_direct = 0;
    for (size_t i = 0; i < msg_count; ++i) {
        double err = std::hypot(h_dec_direct[i].x - h_in[i].x, h_dec_direct[i].y - h_in[i].y);
        if (err > max_err_direct) {
            max_err_direct = err;
            worst_direct = i;
        }
    }
    std::cout << "direct_decode_from_eval max_err=" << std::scientific
              << max_err_direct << ", worst_idx=" << worst_direct << "\n";

    // Full pipeline check
    cuDoubleComplex* d_dec_full = nullptr;
    cudaMalloc(&d_dec_full, msg_count * sizeof(cuDoubleComplex));
    decrypt_and_decode(ct_re, ct_im, sk, d_dec_full);
    cudaDeviceSynchronize();

    std::vector<cuDoubleComplex> h_dec_full(msg_count);
    cudaMemcpy(h_dec_full.data(), d_dec_full, msg_count * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    double max_err_full = 0.0;
    size_t worst_full = 0;
    for (size_t i = 0; i < msg_count; ++i) {
        double err = std::hypot(h_dec_full[i].x - h_in[i].x, h_dec_full[i].y - h_in[i].y);
        if (err > max_err_full) {
            max_err_full = err;
            worst_full = i;
        }
    }
    std::cout << "decrypt_and_decode max_err=" << std::scientific
              << max_err_full << ", worst_idx=" << worst_full << "\n";

    cudaFree(d_in);
    cudaFree(d_packed_re);
    cudaFree(d_packed_im);
    cudaFree(d_eval_re);
    cudaFree(d_eval_im);
    cudaFree(d_dec_direct);
    cudaFree(d_dec_full);
    free_ciphertext(ct_re);
    free_ciphertext(ct_im);
    return (max_err == 0) ? 0 : 1;
}
