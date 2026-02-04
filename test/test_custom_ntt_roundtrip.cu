#include "../include/core/config.h"
#include "../include/core/ntt_core.cuh"
#include "../include/core/HE.cuh"

#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <iostream>

using namespace matrix_fhe;

static uint64_t h_pow_mod(uint64_t base, uint64_t exp, uint64_t mod) {
    uint64_t res = 1;
    base %= mod;
    while (exp > 0) {
        if (exp & 1) res = (unsigned __int128)res * base % mod;
        base = (unsigned __int128)base * base % mod;
        exp >>= 1;
    }
    return res;
}

static uint64_t h_get_psi4n(uint64_t mod, int n) {
    uint64_t order = 4ULL * (uint64_t)n;
    if ((mod - 1) % order != 0) {
        std::cerr << "modulus does not support 4n root\n";
        std::exit(1);
    }
    for (uint64_t root = 2; root < 100000; ++root) {
        uint64_t g = h_pow_mod(root, (mod - 1) / order, mod);
        if (h_pow_mod(g, 2ULL * (uint64_t)n, mod) == mod - 1) return g;
    }
    std::cerr << "failed to find psi4n\n";
    std::exit(1);
}

static inline void cuda_check(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        std::cerr << "[CUDA ERROR] " << msg << ": " << cudaGetErrorString(e) << "\n";
        std::exit(1);
    }
}

static inline uint64_t sub_mod_host(uint64_t a, uint64_t b, uint64_t q) {
    return (a >= b) ? (a - b) : (q - (b - a));
}

int main() {
    init_he_backend();

    // Check moduli一致性
    std::vector<uint64_t> h_dev(RNS_NUM_LIMBS, 0);
    copy_device_moduli(h_dev.data(), RNS_NUM_LIMBS);
    std::cout << "=== Moduli Check ===\n";
    for (int i = 0; i < RNS_NUM_LIMBS; ++i) {
        std::cout << "RNS_MODULI[" << i << "]=" << RNS_MODULI[i]
                  << "  d_he_moduli[" << i << "]=" << h_dev[i] << "\n";
    }

    const int n = MATRIX_N;
    const int limbs = RNS_NUM_LIMBS;

    auto run_roundtrip_std = [&](int batch_count, const char* tag) {
        size_t total = (size_t)batch_count * (size_t)limbs * (size_t)n;
        std::vector<uint64_t> h_in(total, 0);
        for (int b = 0; b < batch_count; ++b) {
            for (int l = 0; l < limbs; ++l) {
                uint64_t q = RNS_MODULI[l];
                for (int x = 0; x < n; ++x) {
                    size_t idx = ((size_t)b * (size_t)limbs + (size_t)l) * (size_t)n + (size_t)x;
                    uint64_t v = (uint64_t)(b + l + x + 1) % q;
                    h_in[idx] = v;
                }
            }
        }

        uint64_t* d_buf = nullptr;
        cuda_check(cudaMalloc(&d_buf, total * sizeof(uint64_t)), "malloc d_buf");
        cuda_check(cudaMemcpy(d_buf, h_in.data(), total * sizeof(uint64_t), cudaMemcpyHostToDevice),
                   "H2D d_buf");

        xy_ntt_forward_phantom(d_buf, limbs, batch_count, n, 0);
        xy_ntt_backward_phantom(d_buf, limbs, batch_count, n, 0);
        cuda_check(cudaDeviceSynchronize(), "ntt sync");

        std::vector<uint64_t> h_out(total, 0);
        cuda_check(cudaMemcpy(h_out.data(), d_buf, total * sizeof(uint64_t), cudaMemcpyDeviceToHost),
                   "D2H d_buf");

        uint64_t max_err = 0;
        size_t worst = 0;
        for (size_t i = 0; i < total; ++i) {
            int l = (int)((i / n) % (size_t)limbs);
            uint64_t q = RNS_MODULI[l];
            uint64_t ao = h_out[i] % q;
            uint64_t ai = h_in[i];
            uint64_t diff = (ao >= ai) ? (ao - ai) : (ai - ao);
            if (diff > q - diff) diff = q - diff;
            if (diff > max_err) {
                max_err = diff;
                worst = i;
            }
        }

        std::cout << "=== XY NTT Roundtrip (standard, " << tag << ") ===\n";
        std::cout << "batch=" << batch_count << ", max_err=" << max_err << ", worst_idx=" << worst << "\n";

        cudaFree(d_buf);
    };

    run_roundtrip_std(1, "batch=1");
    run_roundtrip_std(BATCH_SIZE * MATRIX_N, "batch=phi*n");

    // GL permutation roundtrip: perm -> NTT -> INTT -> inv_perm
    auto run_roundtrip_gl = [&](int batch_count, const char* tag) {
        size_t total = (size_t)batch_count * (size_t)limbs * (size_t)n;
        std::vector<uint64_t> h_in(total, 0);
        for (int b = 0; b < batch_count; ++b) {
            for (int l = 0; l < limbs; ++l) {
                uint64_t q = RNS_MODULI[l];
                for (int x = 0; x < n; ++x) {
                    size_t idx = ((size_t)b * (size_t)limbs + (size_t)l) * (size_t)n + (size_t)x;
                    uint64_t v = (uint64_t)(b + l + x + 7) % q;
                    h_in[idx] = v;
                }
            }
        }

        uint64_t *d_in = nullptr, *d_tmp = nullptr;
        cuda_check(cudaMalloc(&d_in, total * sizeof(uint64_t)), "malloc d_in");
        cuda_check(cudaMalloc(&d_tmp, total * sizeof(uint64_t)), "malloc d_tmp");
        cuda_check(cudaMemcpy(d_in, h_in.data(), total * sizeof(uint64_t), cudaMemcpyHostToDevice),
                   "H2D d_in");

        xy_ntt_forward_gl(d_in, d_tmp, limbs, batch_count, n, 0);
        xy_ntt_backward_gl(d_in, d_tmp, limbs, batch_count, n, 0);
        cuda_check(cudaDeviceSynchronize(), "ntt gl sync");

        std::vector<uint64_t> h_out(total, 0);
        cuda_check(cudaMemcpy(h_out.data(), d_in, total * sizeof(uint64_t), cudaMemcpyDeviceToHost),
                   "D2H d_in");

        uint64_t max_err = 0;
        size_t worst = 0;
        for (size_t i = 0; i < total; ++i) {
            int l = (int)((i / n) % (size_t)limbs);
            uint64_t q = RNS_MODULI[l];
            uint64_t ao = h_out[i] % q;
            uint64_t ai = h_in[i];
            uint64_t diff = (ao >= ai) ? (ao - ai) : (ai - ao);
            if (diff > q - diff) diff = q - diff;
            if (diff > max_err) {
                max_err = diff;
                worst = i;
            }
        }

        std::cout << "=== XY NTT Roundtrip (GL perm, " << tag << ") ===\n";
        std::cout << "batch=" << batch_count << ", max_err=" << max_err << ", worst_idx=" << worst << "\n";

        cudaFree(d_in);
        cudaFree(d_tmp);
    };

    run_roundtrip_gl(1, "batch=1");
    run_roundtrip_gl(BATCH_SIZE * MATRIX_N, "batch=phi*n");

    // W-CRT sanity: single-limb, single position basis check
    {
        std::cout << "=== W-CRT Sanity (basis vector) ===\n";
        const int n = MATRIX_N;
        const int phi = BATCH_SIZE;
        const int limbs = RNS_NUM_LIMBS;
        const size_t n2 = (size_t)n * (size_t)n;
        const size_t total = (size_t)phi * (size_t)limbs * n2;

        uint64_t* d_in = nullptr;
        uint64_t* d_out = nullptr;
        cuda_check(cudaMalloc(&d_in, total * sizeof(uint64_t)), "malloc d_in wcrt");
        cuda_check(cudaMalloc(&d_out, total * sizeof(uint64_t)), "malloc d_out wcrt");
        cuda_check(cudaMemset(d_in, 0, total * sizeof(uint64_t)), "memset d_in wcrt");
        cuda_check(cudaMemset(d_out, 0, total * sizeof(uint64_t)), "memset d_out wcrt");

        const int limb = 0;
        const int y = 0;
        const int x = 0;
        const int r0 = 7; // W-coeff index
        size_t in_off = ((size_t)r0 * (size_t)limbs + (size_t)limb) * n2 + (size_t)y * (size_t)n + (size_t)x;
        uint64_t one = 1;
        cuda_check(cudaMemcpy(d_in + in_off, &one, sizeof(uint64_t), cudaMemcpyHostToDevice), "set basis");

        wntt_forward_matrix(d_in, d_out, n, limbs, phi, 0);
        cuda_check(cudaDeviceSynchronize(), "wcrt forward");

        std::vector<uint64_t> h_out(total, 0);
        cuda_check(cudaMemcpy(h_out.data(), d_out, total * sizeof(uint64_t), cudaMemcpyDeviceToHost), "D2H wcrt");

        // Host-side expected for limb0 using same eta construction
        auto pow_mod = [](uint64_t base, uint64_t exp, uint64_t mod) {
            uint64_t res = 1;
            base %= mod;
            while (exp > 0) {
                if (exp & 1) res = (unsigned __int128)res * base % mod;
                base = (unsigned __int128)base * base % mod;
                exp >>= 1;
            }
            return res;
        };
        auto find_eta = [&](uint64_t q) {
            const uint64_t p = BATCH_PRIME_P;
            const uint64_t f1 = 3;
            const uint64_t f2 = 257;
            const uint64_t exp = (q - 1) / p;
            for (uint64_t g = 2; g < q; ++g) {
                uint64_t eta = pow_mod(g, exp, q);
                if (eta == 1) continue;
                if (pow_mod(eta, p, q) != 1) continue;
                if (pow_mod(eta, p / f1, q) == 1) continue;
                if (pow_mod(eta, p / f2, q) == 1) continue;
                return eta;
            }
            return (uint64_t)0;
        };

        uint64_t q0 = RNS_MODULI[0];
        uint64_t eta0 = find_eta(q0);
        if (eta0 == 0) {
            std::cout << "[W-CRT] failed to find eta for limb0\n";
        } else {
            // exp for W: a outer, b inner (same as HE.cu)
            std::vector<uint32_t> exp(phi, 0);
            int idx = 0;
            for (int a = 1; a <= 2; ++a) {
                for (int b = 1; b <= 256; ++b) {
                    exp[idx++] = (uint32_t)((a * 257 + b * 3) % BATCH_PRIME_P);
                }
            }

            uint64_t max_err = 0;
            for (int w = 0; w < 8; ++w) {
                uint64_t root = pow_mod(eta0, exp[w], q0);
                uint64_t expected = pow_mod(root, (uint64_t)r0, q0);
                size_t out_off = ((size_t)(w * n + y) * (size_t)limbs + (size_t)limb) * (size_t)n + (size_t)x;
                uint64_t got = h_out[out_off] % q0;
                uint64_t diff = (got >= expected) ? (got - expected) : (expected - got);
                if (diff > q0 - diff) diff = q0 - diff;
                if (diff > max_err) max_err = diff;
            }
            std::cout << "[W-CRT] max_err (first 8 lanes, limb0) = " << max_err << "\n";
        }

        cudaFree(d_in);
        cudaFree(d_out);
    }

    std::cout << "=== XY NTT Mul Check (X^n - i, limb0) ===\n";
    {
        const int limbs1 = 1;
        const uint64_t q = RNS_MODULI[0];
        const uint64_t psi4n = h_get_psi4n(q, n);
        const uint64_t iroot = h_pow_mod(psi4n, n, q);

        std::vector<uint64_t> h_a(n), h_b(n), h_c_ref(n);
        for (int j = 0; j < n; ++j) {
            h_a[j] = (uint64_t)(j + 1) % q;
            h_b[j] = (uint64_t)(j + 3) % q;
        }
        std::fill(h_c_ref.begin(), h_c_ref.end(), 0);
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                uint64_t prod = (unsigned __int128)h_a[j] * h_b[k] % q;
                int idx = j + k;
                if (idx < n) {
                    h_c_ref[idx] = (h_c_ref[idx] + prod) % q;
                } else {
                    uint64_t term = (unsigned __int128)prod * iroot % q;
                    h_c_ref[idx - n] = (h_c_ref[idx - n] + term) % q;
                }
            }
        }

        uint64_t* d_a = nullptr;
        uint64_t* d_b = nullptr;
        uint64_t* d_tmp = nullptr;
        cuda_check(cudaMalloc(&d_a, n * sizeof(uint64_t)), "malloc a");
        cuda_check(cudaMalloc(&d_b, n * sizeof(uint64_t)), "malloc b");
        cuda_check(cudaMalloc(&d_tmp, n * sizeof(uint64_t)), "malloc tmp");
        cuda_check(cudaMemcpy(d_a, h_a.data(), n * sizeof(uint64_t), cudaMemcpyHostToDevice), "H2D a");
        cuda_check(cudaMemcpy(d_b, h_b.data(), n * sizeof(uint64_t), cudaMemcpyHostToDevice), "H2D b");

        xy_ntt_forward_gl(d_a, d_tmp, limbs1, 1, n, 0);
        xy_ntt_forward_gl(d_b, d_tmp, limbs1, 1, n, 0);

        std::vector<uint64_t> h_a_ntt(n), h_b_ntt(n);
        cuda_check(cudaMemcpy(h_a_ntt.data(), d_a, n * sizeof(uint64_t), cudaMemcpyDeviceToHost), "D2H a_ntt");
        cuda_check(cudaMemcpy(h_b_ntt.data(), d_b, n * sizeof(uint64_t), cudaMemcpyDeviceToHost), "D2H b_ntt");
        for (int j = 0; j < n; ++j) {
            h_a_ntt[j] = (unsigned __int128)h_a_ntt[j] * h_b_ntt[j] % q;
        }
        cuda_check(cudaMemcpy(d_a, h_a_ntt.data(), n * sizeof(uint64_t), cudaMemcpyHostToDevice), "H2D prod");

        xy_ntt_backward_gl(d_a, d_tmp, limbs1, 1, n, 0);
        std::vector<uint64_t> h_c(n);
        cuda_check(cudaMemcpy(h_c.data(), d_a, n * sizeof(uint64_t), cudaMemcpyDeviceToHost), "D2H c");

        uint64_t max_err = 0;
        for (int j = 0; j < n; ++j) {
            uint64_t a = h_c[j];
            uint64_t b = h_c_ref[j];
            uint64_t d = (a >= b) ? (a - b) : (b - a);
            uint64_t diff = (d > q - d) ? (q - d) : d;
            if (diff > max_err) max_err = diff;
        }
        std::cout << "mul max_err=" << max_err << "\n";

        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_tmp);
    }

    return 0;
}
