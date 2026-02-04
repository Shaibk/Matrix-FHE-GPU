#include "core/ntt_core.cuh"
#include "core/config.h"
#include "core/common.cuh" 
#include "core/HE.cuh"
#include "../../extern/phantom-fhe/include/ntt.cuh"
#include "../../extern/phantom-fhe/include/context.cuh"
#include <vector>
#include <cmath>
#include <iostream>

namespace matrix_fhe {

// 全局 NTT 表
static NTTTable g_ntt_table = {nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0};
static uint32_t* d_gl_perm = nullptr;
static uint32_t* d_gl_inv_perm = nullptr;
static uint64_t* d_gl_twist = nullptr;
static uint64_t* d_gl_twist_inv = nullptr;

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

static uint32_t bit_reverse_u32(uint32_t x, int bits) {
    uint32_t r = 0;
    for (int i = 0; i < bits; ++i) {
        r = (r << 1) | (x & 1U);
        x >>= 1U;
    }
    return r;
}

// 查找 4N 次本原单位根 (psi4n)
static uint64_t get_psi(uint64_t mod, int n) {
    uint64_t root = 2; 
    uint64_t order = 4 * n;
    
    if ((mod - 1) % order != 0) {
        std::cerr << "Error: Modulus " << mod << " does not support NTT size " << n << "\n";
        exit(1);
    }
    
    while (true) {
        // 使用重命名后的 h_pow_mod
        uint64_t g = h_pow_mod(root, (mod - 1) / order, mod);
        // g^(2n) == -1 ensures order 4n (g^(4n)=1)
        if (h_pow_mod(g, 2 * n, mod) == (mod - 1)) return g;
        root++;
        if (root > 100000) {
            std::cerr << "Error: Failed to find psi4n for mod " << mod << "\n";
            exit(1);
        }
    }
    return 0;
}

// ==========================================
// Initialization
// ==========================================
void init_ntt_tables_manual(int n, int limbs) {
    if (g_ntt_table.d_psi_powers != nullptr) return;
    init_ntt_moduli_manual(matrix_fhe::RNS_MODULI);

    // 假设 RNS_MODULI 在 config.h 或 globals.cpp 中可见
    // 如果这里报错 RNS_MODULI 未定义，请确保包含了正确的头文件
    // 或者临时使用 extern 声明
    // extern const uint64_t RNS_MODULI[RNS_NUM_LIMBS]; 

    std::vector<uint64_t> all_psi(limbs * n);
    std::vector<uint64_t> all_psi_inv(limbs * n);
    std::vector<uint64_t> all_twist(limbs * n);
    std::vector<uint64_t> all_twist_inv(limbs * n);
    std::vector<uint64_t> all_n_inv(limbs);

    for (int l = 0; l < limbs; ++l) {
        uint64_t q = matrix_fhe::RNS_MODULI[l]; // 显式指明命名空间
        uint64_t psi4n = get_psi(q, n);               // order 4n
        uint64_t omega = h_pow_mod(psi4n, 4, q);      // order n
        uint64_t omega_inv = h_inv_mod(omega, q);
        uint64_t psi4n_inv = h_inv_mod(psi4n, q);
        uint64_t n_inv = h_inv_mod(n, q);

        all_n_inv[l] = n_inv;

        uint64_t temp_psi[n];
        uint64_t temp_psi_inv[n];
        uint64_t temp_twist[n];
        uint64_t temp_twist_inv[n];
        
        temp_psi[0] = 1;
        temp_psi_inv[0] = 1;
        temp_twist[0] = 1;
        temp_twist_inv[0] = 1;
        for(int i=1; i<n; ++i) {
            temp_psi[i] = (unsigned __int128)temp_psi[i-1] * omega % q;
            temp_psi_inv[i] = (unsigned __int128)temp_psi_inv[i-1] * omega_inv % q;
            temp_twist[i] = (unsigned __int128)temp_twist[i-1] * psi4n % q;
            temp_twist_inv[i] = (unsigned __int128)temp_twist_inv[i-1] * psi4n_inv % q;
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
            all_psi[l * n + i] = temp_psi[i];
            all_psi_inv[l * n + i] = temp_psi_inv[i];
            all_twist[l * n + i] = temp_twist[i];
            all_twist_inv[l * n + i] = temp_twist_inv[i];
        }
    }

    size_t table_size = limbs * n * sizeof(uint64_t);
    cudaMalloc(&g_ntt_table.d_psi_powers, table_size);
    cudaMalloc(&g_ntt_table.d_psi_inv_powers, table_size);
    cudaMalloc(&g_ntt_table.d_twist_powers, table_size);
    cudaMalloc(&g_ntt_table.d_twist_inv_powers, table_size);
    cudaMalloc(&g_ntt_table.d_n_inv, limbs * sizeof(uint64_t));

    cudaMemcpy(g_ntt_table.d_psi_powers, all_psi.data(), table_size, cudaMemcpyHostToDevice);
    cudaMemcpy(g_ntt_table.d_psi_inv_powers, all_psi_inv.data(), table_size, cudaMemcpyHostToDevice);
    cudaMemcpy(g_ntt_table.d_twist_powers, all_twist.data(), table_size, cudaMemcpyHostToDevice);
    cudaMemcpy(g_ntt_table.d_twist_inv_powers, all_twist_inv.data(), table_size, cudaMemcpyHostToDevice);
    cudaMemcpy(g_ntt_table.d_n_inv, all_n_inv.data(), limbs * sizeof(uint64_t), cudaMemcpyHostToDevice);

    g_ntt_table.n = n;
    g_ntt_table.modulus_count = limbs;
}

void init_gl_perm_tables(int n) {
    if (d_gl_perm != nullptr) return;
    if ((n & (n - 1)) != 0) {
        std::cerr << "Error: GL perm requires power-of-two n\n";
        exit(1);
    }
    int logn = 0;
    while ((1 << logn) < n) ++logn;
    const int m = 4 * n;
    std::vector<uint32_t> h_perm(n, 0);
    std::vector<uint32_t> h_inv(n, 0);
    uint32_t e = 1 % m;
    for (int j = 0; j < n; ++j) {
        uint32_t idx = (e - 1) / 4; // e is 1 mod 4
        uint32_t idx_target = bit_reverse_u32(idx, logn);
        h_perm[j] = idx_target;
        h_inv[idx_target] = (uint32_t)j;
        e = (uint32_t)((uint64_t)e * 5ULL % (uint64_t)m);
    }
    cudaMalloc(&d_gl_perm, n * sizeof(uint32_t));
    cudaMalloc(&d_gl_inv_perm, n * sizeof(uint32_t));
    cudaMemcpy(d_gl_perm, h_perm.data(), n * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gl_inv_perm, h_inv.data(), n * sizeof(uint32_t), cudaMemcpyHostToDevice);
}

void init_gl_twist_tables(int n, int limbs) {
    if (d_gl_twist != nullptr) return;
    std::vector<uint64_t> h_twist((size_t)limbs * (size_t)n, 0);
    std::vector<uint64_t> h_twist_inv((size_t)limbs * (size_t)n, 0);
    for (int l = 0; l < limbs; ++l) {
        uint64_t q = matrix_fhe::RNS_MODULI[l];
        uint64_t psi4n = get_psi(q, n);         // order 4n
        uint64_t beta = psi4n;                  // beta^n = i
        uint64_t beta_inv = h_inv_mod(beta, q);
        uint64_t cur = 1;
        uint64_t cur_inv = 1;
        for (int i = 0; i < n; ++i) {
            h_twist[l * n + i] = cur;
            h_twist_inv[l * n + i] = cur_inv;
            cur = (unsigned __int128)cur * beta % q;
            cur_inv = (unsigned __int128)cur_inv * beta_inv % q;
        }
    }
    size_t bytes = (size_t)limbs * (size_t)n * sizeof(uint64_t);
    cudaMalloc(&d_gl_twist, bytes);
    cudaMalloc(&d_gl_twist_inv, bytes);
    cudaMemcpy(d_gl_twist, h_twist.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_gl_twist_inv, h_twist_inv.data(), bytes, cudaMemcpyHostToDevice);
}

const uint32_t* get_gl_perm() { return d_gl_perm; }
const uint32_t* get_gl_inv_perm() { return d_gl_inv_perm; }

// Local moduli for kernels in this TU (avoid RDC symbol issues)
__device__ __constant__ uint64_t d_ntt_moduli[RNS_NUM_LIMBS];

void init_ntt_moduli_manual(const uint64_t* h_moduli) {
    cudaMemcpyToSymbol(d_ntt_moduli, h_moduli, RNS_NUM_LIMBS * sizeof(uint64_t));
}

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

// Negacyclic twist kernel: multiply by psi^i (or psi^-i)
__global__ void twist_kernel(uint64_t* data, const uint64_t* twist_powers,
                             int n, int limbs, int batch_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_len = n * limbs * batch_count;
    if (idx >= total_len) return;

    int coeff_idx = idx % n;
    int prefix = idx / n; // Poly * Limbs + Limb
    int limb_idx = prefix % limbs;

    uint64_t q = d_ntt_moduli[limb_idx];
    uint64_t w = twist_powers[limb_idx * n + coeff_idx];
    data[idx] = (unsigned __int128)data[idx] * w % q;
}

__global__ void gl_perm_kernel(const uint64_t* in, uint64_t* out, const uint32_t* perm,
                               int n, int limbs, int batch_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_len = n * limbs * batch_count;
    if (idx >= total_len) return;

    int x = idx % n;
    int prefix = idx / n; // Poly * Limbs + Limb
    int x_out = perm[x];
    int out_idx = prefix * n + x_out;
    out[out_idx] = in[idx];
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
    int step = n / m;

    int i = k * m + j;
    int i_plus_t = i + m / 2;

    uint64_t* poly_base = data + poly_limb_idx * n;
    uint64_t q = d_ntt_moduli[limb_idx];
    
    // Twiddle
    uint64_t w = psi_powers[limb_idx * n + j * step];

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
    int step = n / m;
    
    int i = k * m + j;
    int i_plus_t = i + m / 2;

    uint64_t* poly_base = data + poly_limb_idx * n;
    uint64_t q = d_ntt_moduli[limb_idx];
    
    uint64_t w = psi_inv_powers[limb_idx * n + j * step];

    uint64_t u = poly_base[i];
    uint64_t v = poly_base[i_plus_t];

    uint64_t u_prime = (u + v >= q) ? (u + v - q) : (u + v);
    uint64_t sub = (u >= v) ? (u - v) : (u + q - v);
    uint64_t v_prime = (unsigned __int128)sub * w % q;

    poly_base[i] = u_prime;
    poly_base[i_plus_t] = v_prime;
}

// Cooley-Tukey Butterfly (Inverse, DIT-style)
__global__ void ntt_ct_inv_butterfly_kernel(uint64_t* data, const uint64_t* psi_inv_powers,
                                            int n, int stride, int m, int limbs, int batch_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int butterflies_per_poly = n / 2;
    int total_butterflies = butterflies_per_poly * limbs * batch_count;
    
    if (idx >= total_butterflies) return;

    int poly_limb_idx = idx / butterflies_per_poly; 
    int bf_idx = idx % butterflies_per_poly;        

    int limb_idx = poly_limb_idx % limbs; 

    int k = bf_idx / (m / 2);
    int j = bf_idx % (m / 2);
    int step = n / m;

    int i = k * m + j;
    int i_plus_t = i + m / 2;

    uint64_t* poly_base = data + poly_limb_idx * n;
    uint64_t q = d_ntt_moduli[limb_idx];
    
    uint64_t w = psi_inv_powers[limb_idx * n + j * step];

    uint64_t u = poly_base[i];
    uint64_t v = poly_base[i_plus_t];

    uint64_t sum = (u + v >= q) ? (u + v - q) : (u + v);
    uint64_t diff = (u >= v) ? (u - v) : (u + q - v);
    uint64_t v_prime = (unsigned __int128)diff * w % q;

    poly_base[i] = sum;
    poly_base[i_plus_t] = v_prime;
}
// 标量乘法 (乘 N^-1)
__global__ void scalar_mul_kernel(uint64_t* data, const uint64_t* n_inv, int n, int limbs, int batch_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * limbs * batch_count) return;

    int offset_in_poly = idx % (n * limbs);
    int limb_idx = offset_in_poly / n;
    
    uint64_t q = d_ntt_moduli[limb_idx];
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
    int total_elements = n * limbs * batch_count;
    int blocks = (total_elements + 256 - 1) / 256;
    // Inverse: bit-reverse at input (DIT-style)
    bit_reverse_kernel<<<blocks, 256, 0, stream>>>(data, n, limbs, batch_count);

    int total_butterflies = (n / 2) * limbs * batch_count;
    int bf_blocks = (total_butterflies + 256 - 1) / 256;

    for (int m = 2; m <= n; m <<= 1) {
        int stride = m / 2;
        ntt_ct_butterfly_kernel<<<bf_blocks, 256, 0, stream>>>(
            data, g_ntt_table.d_psi_inv_powers, n, stride, m, limbs, batch_count
        );
    }

    scalar_mul_kernel<<<blocks, 256, 0, stream>>>(
        data, g_ntt_table.d_n_inv, n, limbs, batch_count
    );
}

void apply_gl_perm(const uint64_t* in, uint64_t* out, int limbs, int batch_count, int n, bool inverse,
                   cudaStream_t stream) {
    if (d_gl_perm == nullptr) init_gl_perm_tables(n);
    const uint32_t* perm = inverse ? d_gl_inv_perm : d_gl_perm;
    int total_elements = n * limbs * batch_count;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    gl_perm_kernel<<<blocks, threads, 0, stream>>>(in, out, perm, n, limbs, batch_count);
}

void xy_ntt_forward_phantom(uint64_t* data, int limbs, int batch_count, int n, cudaStream_t stream) {
    const auto& tab = get_xy_ntt_table();
    for (int b = 0; b < batch_count; ++b) {
        uint64_t* poly = data + (size_t)b * (size_t)limbs * (size_t)n;
        fnwt_1d(poly, tab.twiddle(), tab.twiddle_shoup(), tab.modulus(),
                (size_t)n, (size_t)limbs, 0, stream);
    }
}

void xy_ntt_backward_phantom(uint64_t* data, int limbs, int batch_count, int n, cudaStream_t stream) {
    const auto& tab = get_xy_ntt_table();
    for (int b = 0; b < batch_count; ++b) {
        uint64_t* poly = data + (size_t)b * (size_t)limbs * (size_t)n;
        inwt_1d(poly, tab.itwiddle(), tab.itwiddle_shoup(), tab.modulus(),
                tab.n_inv_mod_q(), tab.n_inv_mod_q_shoup(),
                (size_t)n, (size_t)limbs, 0, stream);
    }
}

void xy_ntt_forward_gl(uint64_t* data, uint64_t* tmp, int limbs, int batch_count, int n, cudaStream_t stream) {
    if (d_gl_twist == nullptr) init_gl_twist_tables(n, limbs);
    if (g_ntt_table.d_psi_powers == nullptr) init_ntt_tables_manual(n, limbs);
    size_t bytes = (size_t)batch_count * (size_t)limbs * (size_t)n * sizeof(uint64_t);
    cudaMemcpyAsync(tmp, data, bytes, cudaMemcpyDeviceToDevice, stream);
    int total_elements = n * limbs * batch_count;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    twist_kernel<<<blocks, threads, 0, stream>>>(tmp, d_gl_twist, n, limbs, batch_count);
    custom_ntt_forward(tmp, limbs, batch_count, n, stream);
    cudaMemcpyAsync(data, tmp, bytes, cudaMemcpyDeviceToDevice, stream);
}

void xy_ntt_backward_gl(uint64_t* data, uint64_t* tmp, int limbs, int batch_count, int n, cudaStream_t stream) {
    custom_ntt_backward(data, limbs, batch_count, n, stream);
    int total_elements = n * limbs * batch_count;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    twist_kernel<<<blocks, threads, 0, stream>>>(data, d_gl_twist_inv, n, limbs, batch_count);
}

} // namespace matrix_fhe
