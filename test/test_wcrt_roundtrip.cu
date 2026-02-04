#include "../include/core/config.h"
#include "../include/core/HE.cuh"

#include <cuda_runtime.h>
#include <cstdint>
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
    std::cout << "=== W-CRT Roundtrip Test ===\n";
    init_he_backend();

    const int n = MATRIX_N;
    const int phi = BATCH_SIZE;
    const size_t n2 = (size_t)n * (size_t)n;
    const size_t total = (size_t)phi * n2;

    int64_t* d_coeff = nullptr;
    int64_t* d_eval = nullptr;
    int64_t* d_coeff_rt = nullptr;
    cuda_check(cudaMalloc(&d_coeff, total * sizeof(uint64_t)), "malloc d_coeff");
    cuda_check(cudaMalloc(&d_eval, total * sizeof(uint64_t)), "malloc d_eval");
    cuda_check(cudaMalloc(&d_coeff_rt, total * sizeof(uint64_t)), "malloc d_coeff_rt");

    // Fill coeff with a simple centered integer pattern in W-coeff domain
    std::vector<int64_t> h_coeff(total, 0);
    for (int w = 0; w < phi; ++w) {
        for (int y = 0; y < n; ++y) {
            for (int x = 0; x < n; ++x) {
                size_t off = (size_t)w * n2 + (size_t)y * (size_t)n + (size_t)x;
                int64_t v = (int64_t)((w + x + y) % 17) - 8;
                h_coeff[off] = v;
            }
        }
    }

    cuda_check(cudaMemcpy(d_coeff, h_coeff.data(), total * sizeof(uint64_t), cudaMemcpyHostToDevice), "H2D coeff");

    // Forward W-CRT (coeff -> eval), then inverse (eval -> coeff)
    wntt_forward_centered(d_coeff, d_eval, n, phi, 0);
    wntt_inverse_centered(d_eval, d_coeff_rt, n, phi, 0);
    cuda_check(cudaDeviceSynchronize(), "wcrt roundtrip");

    std::vector<int64_t> h_rt(total, 0);
    cuda_check(cudaMemcpy(h_rt.data(), d_coeff_rt, total * sizeof(uint64_t), cudaMemcpyDeviceToHost), "D2H coeff_rt");

    int64_t max_err = 0;
    size_t worst = 0;
    for (size_t i = 0; i < total; ++i) {
        int64_t diff = h_coeff[i] - h_rt[i];
        if (diff < 0) diff = -diff;
        if (diff > max_err) {
            max_err = diff;
            worst = i;
        }
    }

    std::cout << "max_err=" << max_err << ", worst_idx=" << worst << "\n";

    cudaFree(d_coeff);
    cudaFree(d_eval);
    cudaFree(d_coeff_rt);
    return (max_err == 0) ? 0 : 1;
}
