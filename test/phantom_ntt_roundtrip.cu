#include "../include/core/config.h"
#include "../include/core/HE.cuh"
#include "../include/core/common.cuh"

#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <iostream>

using namespace matrix_fhe;

// -----------------------------------------------------------------------------
// IMPORTANT: this must refer to the SAME constant symbol defined in your HE backend.
// If your backend uses a different name, change it here accordingly.
// Also ensure the definition in HE.cu is NOT "static __constant__" (must be global).
// -----------------------------------------------------------------------------
__constant__ uint64_t d_he_moduli[matrix_fhe::RNS_NUM_LIMBS];

// ---------------- CUDA helpers ----------------
static void cuda_check(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        std::cerr << "[CUDA ERROR] " << msg << ": " << cudaGetErrorString(e) << "\n";
        std::exit(1);
    }
}
static void cuda_sync(const char* msg) {
    cuda_check(cudaGetLastError(), msg);
    cuda_check(cudaDeviceSynchronize(), msg);
}

// Layout: [tower][coeff] contiguous, total_len = limbs * N
__global__ void fill_uniform_u64(uint64_t* out, size_t total_len, int N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_len) return;

    int tower = (int)(idx / (size_t)N);
    uint64_t q = d_he_moduli[tower];   // <-- MUST be nonzero

    // If q is 0, modulo is undefined; guard to make bug obvious.
    if (q == 0) {
        out[idx] = 0xFFFFFFFFFFFFFFFFULL;
        return;
    }

    // deterministic xorshift-ish RNG
    uint64_t x = 0x9E3779B97F4A7C15ULL ^ (uint64_t)idx;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    x *= 2685821657736338717ULL;

    out[idx] = x % q;
}

int main() {
    std::cout << "=== Phantom NTT Roundtrip Test (reuse HE backend) ===\n";
    std::cout << "POLY_N = " << POLY_N << ", LIMBS = " << RNS_NUM_LIMBS << "\n";

    // 1) init backend (should also cudaMemcpyToSymbol the moduli constant)
    init_he_backend();
    cuda_sync("init_he_backend");
    cudaMemcpyToSymbol(d_he_moduli, matrix_fhe::RNS_MODULI, sizeof(matrix_fhe::RNS_MODULI));

    // 2) verify device moduli constant is correctly initialized
    uint64_t h_mod[RNS_NUM_LIMBS] = {0, 0, 0};
    cudaError_t e = cudaMemcpyFromSymbol(h_mod, d_he_moduli, sizeof(h_mod));
    if (e != cudaSuccess) {
        std::cerr << "[FATAL] cudaMemcpyFromSymbol(d_he_moduli) failed: "
                  << cudaGetErrorString(e) << "\n";
        std::cerr << "This usually means the symbol name doesn't match, OR the original "
                     "definition is 'static __constant__' in another TU.\n";
        return 1;
    }
    std::cout << "[DBG] d_he_moduli readback:\n";
    for (int i = 0; i < RNS_NUM_LIMBS; ++i) {
        std::cout << "  limb" << i << ": " << h_mod[i] << "\n";
    }

    // sanity: must match host config
    bool mod_ok = true;
    for (int i = 0; i < RNS_NUM_LIMBS; ++i) {
        if (h_mod[i] != RNS_MODULI[i]) mod_ok = false;
    }
    if (!mod_ok) {
        std::cerr << "[FATAL] d_he_moduli does NOT match RNS_MODULI from config.h.\n";
        std::cerr << "Host RNS_MODULI:\n";
        for (int i = 0; i < RNS_NUM_LIMBS; ++i) {
            std::cerr << "  limb" << i << ": " << RNS_MODULI[i] << "\n";
        }
        std::cerr << "Fix: ensure init_he_backend() calls cudaMemcpyToSymbol(d_he_moduli, ...)\n"
                     "and that d_he_moduli is a single global __constant__ symbol (not static).\n";
        return 1;
    }

    // 3) allocate one poly buffer (plus padding to avoid any potential tail overread)
    const int N = POLY_N;
    const int L = RNS_NUM_LIMBS;

    size_t core_u64  = (size_t)L * (size_t)N;
    size_t pad_u64   = 2048;                 // 16KB padding
    size_t total_u64 = core_u64 + pad_u64;

    uint64_t* d_x   = nullptr;
    uint64_t* d_ref = nullptr;
    cuda_check(cudaMalloc(&d_x,   total_u64 * sizeof(uint64_t)), "malloc d_x");
    cuda_check(cudaMalloc(&d_ref, total_u64 * sizeof(uint64_t)), "malloc d_ref");
    cuda_check(cudaMemset(d_x + core_u64,   0, pad_u64 * sizeof(uint64_t)), "pad x");
    cuda_check(cudaMemset(d_ref + core_u64, 0, pad_u64 * sizeof(uint64_t)), "pad ref");

    // 4) fill random, backup
    int thr = 256;
    int blk = (int)((core_u64 + thr - 1) / thr);
    fill_uniform_u64<<<blk, thr>>>(d_x, core_u64, N);
    cuda_sync("fill_uniform_u64");

    // quick sanity print of first 10 values (limb0)
    std::vector<uint64_t> h_first(16, 0);
    cuda_check(cudaMemcpy(h_first.data(), d_x, 16 * sizeof(uint64_t), cudaMemcpyDeviceToHost),
               "D2H first16");
    std::cout << "[DBG] first 10 x (limb0):\n";
    for (int i = 0; i < 10; ++i) std::cout << "  " << h_first[i] << "\n";
    // they must be < q0
    if (h_first[0] >= RNS_MODULI[0]) {
        std::cerr << "[FATAL] fill kernel produced value >= q0. Moduli read is broken.\n";
        return 1;
    }

    cuda_check(cudaMemcpy(d_ref, d_x, core_u64 * sizeof(uint64_t), cudaMemcpyDeviceToDevice),
               "copy ref");

    // 5) Phantom NTT forward/backward via your existing wrappers
    cudaStream_t stream = 0;
    nwt_2d_radix8_forward_inplace(d_x, get_ntt_table(), (size_t)L, 0, stream);
    cuda_sync("phantom forward");

    nwt_2d_radix8_backward_inplace(d_x, get_ntt_table(), (size_t)L, 0, stream);
    cuda_sync("phantom backward");

    // 6) compare (exact compare first; if it fails but looks like a fixed scale, we can extend)
    std::vector<uint64_t> h_x(core_u64), h_ref(core_u64);
    cuda_check(cudaMemcpy(h_x.data(),   d_x,   core_u64 * sizeof(uint64_t), cudaMemcpyDeviceToHost), "D2H x");
    cuda_check(cudaMemcpy(h_ref.data(), d_ref, core_u64 * sizeof(uint64_t), cudaMemcpyDeviceToHost), "D2H ref");

    size_t mismatch = 0;
    for (size_t i = 0; i < core_u64; ++i) {
        if (h_x[i] != h_ref[i]) {
            if (mismatch < 10) {
                size_t limb  = i / (size_t)N;
                size_t coeff = i % (size_t)N;
                std::cout << "mismatch @" << i
                          << " (limb=" << limb << ", coeff=" << coeff << ")"
                          << " ref=" << h_ref[i]
                          << " got=" << h_x[i] << "\n";
            }
            mismatch++;
        }
    }

    cudaFree(d_x);
    cudaFree(d_ref);

    if (mismatch == 0) {
        std::cout << ">>> [PASS] NTT roundtrip exact match\n";
        return 0;
    } else {
        std::cout << ">>> [FAIL] mismatches = " << mismatch << " (total=" << core_u64 << ")\n";
        std::cout << "NOTE: If Phantom iNTT returns a scaled result, we can extend this test to\n"
                     "      detect a per-limb constant scale and validate got == ref * scale (mod q).\n";
        return 1;
    }
}
