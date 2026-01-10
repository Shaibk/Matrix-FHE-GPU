#include "core/ntt_core.cuh"
#include "core/config.h"
#include "core/common.cuh" 
#include <vector>
#include <cmath>
#include <iostream>

namespace matrix_fhe {

// 全局 NTT 表
static NTTTable g_ntt_table = {nullptr, nullptr, nullptr, 0, 0};

// ==========================================
// Host Math Helpers (Private to this file)
// 重命名以避免与 common.cuh 中的 __device__ 函数冲突
// ==========================================
static uint64_t h_pow_mod(uint64_t base, uint64_t exp, uint64_t mod) {
    uint64_t res = 1;
    base %= mod;
    while (exp > 0) {
        if (exp % 2 == 1) res = (unsigned __int128)res * base % mod;
        base = (unsigned __int128)base * base % mod;
        exp /= 2;
    }
    return res;
}

static uint64_t h_inv_mod(uint64_t n, uint64_t mod) {
    return h_pow_mod(n, mod - 2, mod);
}

// 查找 2N 次本原单位根 (psi)
static uint64_t get_psi(uint64_t mod, int n) {
    uint64_t root = 2; 
    uint64_t order = 2 * n;
    
    if ((mod - 1) % order != 0) {
        std::cerr << "Error: Modulus " << mod << " does not support NTT size " << n << "\n";
        exit(1);
    }
    
    while (true) {
        // 使用重命名后的 h_pow_mod
        uint64_t g = h_pow_mod(root, (mod - 1) / order, mod);
        if (h_pow_mod(g, n, mod) != 1) { 
             if (h_pow_mod(g, n, mod) == (mod - 1)) return g;
        }
        root++;
        if (root > 1000) break; 
    }
    return 0; 
}

// ==========================================
// Initialization
// ==========================================
void init_ntt_tables_manual(int n, int limbs) {
    if (g_ntt_table.d_psi_powers != nullptr) return;

    // 假设 RNS_MODULI 在 config.h 或 globals.cpp 中可见
    // 如果这里报错 RNS_MODULI 未定义，请确保包含了正确的头文件
    // 或者临时使用 extern 声明
    // extern const uint64_t RNS_MODULI[RNS_NUM_LIMBS]; 

    std::vector<uint64_t> all_psi(limbs * n);
    std::vector<uint64_t> all_psi_inv(limbs * n);
    std::vector<uint64_t> all_n_inv(limbs);

    for (int l = 0; l < limbs; ++l) {
        uint64_t q = matrix_fhe::RNS_MODULI[l]; // 显式指明命名空间
        uint64_t psi = get_psi(q, n);
        uint64_t psi_inv = h_inv_mod(psi, q);
        uint64_t n_inv = h_inv_mod(n, q);

        all_n_inv[l] = n_inv;

        uint64_t temp_psi[n];
        uint64_t temp_psi_inv[n];
        
        temp_psi[0] = 1;
        temp_psi_inv[0] = 1;
        for(int i=1; i<n; ++i) {
            temp_psi[i] = (unsigned __int128)temp_psi[i-1] * psi % q;
            temp_psi_inv[i] = (unsigned __int128)temp_psi_inv[i-1] * psi_inv % q;
        }

        // Bit Reverse Copy
        for (int i = 0; i < n; ++i) {
            int rev = 0;
            int bit = n >> 1;
            int k = i;
            while (bit > 0) {
                if (k & 1) rev |= bit;
                k >>= 1;
                bit >>= 1;
            }
            all_psi[l * n + i] = temp_psi[rev];
            all_psi_inv[l * n + i] = temp_psi_inv[rev];
        }
    }

    size_t table_size = limbs * n * sizeof(uint64_t);
    cudaMalloc(&g_ntt_table.d_psi_powers, table_size);
    cudaMalloc(&g_ntt_table.d_psi_inv_powers, table_size);
    cudaMalloc(&g_ntt_table.d_n_inv, limbs * sizeof(uint64_t));

    cudaMemcpy(g_ntt_table.d_psi_powers, all_psi.data(), table_size, cudaMemcpyHostToDevice);
    cudaMemcpy(g_ntt_table.d_psi_inv_powers, all_psi_inv.data(), table_size, cudaMemcpyHostToDevice);
    cudaMemcpy(g_ntt_table.d_n_inv, all_n_inv.data(), limbs * sizeof(uint64_t), cudaMemcpyHostToDevice);

    g_ntt_table.n = n;
    g_ntt_table.modulus_count = limbs;
}

// 引用外部常量模数 (由 HE.cu 定义)
extern __constant__ uint64_t d_he_moduli[3]; 

// ==========================================
// CUDA Kernels
// ==========================================

// Bit Reversal Kernel
__global__ void bit_reverse_kernel(uint64_t* data, int n, int limbs, int batch_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_len = n * limbs * batch_count;
    if (idx >= total_len) return;

    // 逻辑索引: [Poly][Limb][Coeff]
    // 只需要在 Coeff 维度做 Bit Reverse
    int coeff_idx = idx % n;
    int prefix = idx / n; // Poly * Limbs + Limb

    int rev = 0;
    int k = coeff_idx;
    int bit = n >> 1;
    while (bit > 0) {
        if (k & 1) rev |= bit;
        k >>= 1;
        bit >>= 1;
    }

    if (rev > coeff_idx) {
        uint64_t* base = data + prefix * n;
        uint64_t temp = base[coeff_idx];
        base[coeff_idx] = base[rev];
        base[rev] = temp;
    }
}

// Cooley-Tukey Butterfly (Forward)
__global__ void ntt_ct_butterfly_kernel(uint64_t* data, const uint64_t* psi_powers, int n, int stride, int m, int limbs, int batch_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int butterflies_per_poly = n / 2;
    int total_butterflies = butterflies_per_poly * limbs * batch_count;
    
    if (idx >= total_butterflies) return;

    int poly_limb_idx = idx / butterflies_per_poly; 
    int bf_idx = idx % butterflies_per_poly;        

    int limb_idx = poly_limb_idx % limbs; 

    int k = bf_idx / (m / 2); 
    int j = bf_idx % (m / 2); 
    
    int i = k * m + j;        
    int i_plus_t = i + m / 2; 

    uint64_t* poly_base = data + poly_limb_idx * n;
    uint64_t q = d_he_moduli[limb_idx];
    
    // Twiddle
    uint64_t w = psi_powers[limb_idx * n + k]; 

    uint64_t u = poly_base[i];
    uint64_t v = poly_base[i_plus_t];

    uint64_t vw = (unsigned __int128)v * w % q;
    
    poly_base[i] = (u + vw >= q) ? (u + vw - q) : (u + vw);
    poly_base[i_plus_t] = (u >= vw) ? (u - vw) : (u + q - vw);
}

// Gentleman-Sande Butterfly (Inverse)
__global__ void ntt_gs_butterfly_kernel(uint64_t* data, const uint64_t* psi_inv_powers, int n, int stride, int m, int limbs, int batch_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int butterflies_per_poly = n / 2;
    int total_butterflies = butterflies_per_poly * limbs * batch_count;
    
    if (idx >= total_butterflies) return;

    int poly_limb_idx = idx / butterflies_per_poly;
    int bf_idx = idx % butterflies_per_poly;
    int limb_idx = poly_limb_idx % limbs;

    int k = bf_idx / (m / 2);
    int j = bf_idx % (m / 2);
    
    int i = k * m + j;
    int i_plus_t = i + m / 2;

    uint64_t* poly_base = data + poly_limb_idx * n;
    uint64_t q = d_he_moduli[limb_idx];
    
    uint64_t w = psi_inv_powers[limb_idx * n + k]; 

    uint64_t u = poly_base[i];
    uint64_t v = poly_base[i_plus_t];

    uint64_t u_prime = (u + v >= q) ? (u + v - q) : (u + v);
    uint64_t sub = (u >= v) ? (u - v) : (u + q - v);
    uint64_t v_prime = (unsigned __int128)sub * w % q;

    poly_base[i] = u_prime;
    poly_base[i_plus_t] = v_prime;
}

// 标量乘法 (乘 N^-1)
__global__ void scalar_mul_kernel(uint64_t* data, const uint64_t* n_inv, int n, int limbs, int batch_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * limbs * batch_count) return;

    int offset_in_poly = idx % (n * limbs);
    int limb_idx = offset_in_poly / n;
    
    uint64_t q = d_he_moduli[limb_idx];
    uint64_t inv = n_inv[limb_idx];
    
    data[idx] = (unsigned __int128)data[idx] * inv % q;
}

// ==========================================
// Implementation Wrappers
// ==========================================

void custom_ntt_forward(uint64_t* data, int limbs, int batch_count, int n, cudaStream_t stream) {
    int total_elements = n * limbs * batch_count;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    bit_reverse_kernel<<<blocks, threads, 0, stream>>>(data, n, limbs, batch_count);

    int total_butterflies = (n / 2) * limbs * batch_count;
    int bf_blocks = (total_butterflies + threads - 1) / threads;

    for (int m = 2; m <= n; m <<= 1) {
        int stride = m / 2;
        ntt_ct_butterfly_kernel<<<bf_blocks, threads, 0, stream>>>(
            data, g_ntt_table.d_psi_powers, n, stride, m, limbs, batch_count
        );
    }
}

void custom_ntt_backward(uint64_t* data, int limbs, int batch_count, int n, cudaStream_t stream) {
    int total_butterflies = (n / 2) * limbs * batch_count;
    int bf_blocks = (total_butterflies + 256 - 1) / 256;

    for (int m = n; m >= 2; m >>= 1) {
        int stride = m / 2;
        ntt_gs_butterfly_kernel<<<bf_blocks, 256, 0, stream>>>(
            data, g_ntt_table.d_psi_inv_powers, n, stride, m, limbs, batch_count
        );
    }

    int total_elements = n * limbs * batch_count;
    int blocks = (total_elements + 256 - 1) / 256;
    bit_reverse_kernel<<<blocks, 256, 0, stream>>>(data, n, limbs, batch_count);

    scalar_mul_kernel<<<blocks, 256, 0, stream>>>(
        data, g_ntt_table.d_n_inv, n, limbs, batch_count
    );
}

} // namespace matrix_fhe