#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace matrix_fhe {

// NTT 配置表结构
struct NTTTable {
    uint64_t* d_psi_powers;      // Forward twiddles (omega^k), omega is n-th root
    uint64_t* d_psi_inv_powers;  // Inverse twiddles (omega^{-k})
    uint64_t* d_twist_powers;    // Negacyclic twist (psi^i), psi is 2n-th root
    uint64_t* d_twist_inv_powers;// Negacyclic inverse twist (psi^{-i})
    uint64_t* d_n_inv;           // N^-1 mod q
    int       modulus_count;
    int       n;
};

// 初始化 NTT 表 (在 Host 端计算并上传)
void init_ntt_tables_manual(int n, int limbs);
// Initialize local moduli for ntt_core.cu kernels
void init_ntt_moduli_manual(const uint64_t* h_moduli);

// 获取表实例
const NTTTable& get_manual_ntt_table();

// GL permutation tables for X/Y axis (n must match MATRIX_N)
void init_gl_perm_tables(int n);
void init_gl_twist_tables(int n, int limbs);
const uint32_t* get_gl_perm();
const uint32_t* get_gl_inv_perm();

// Apply GL permutation on X dimension: [batch][limb][x] -> [batch][limb][x]
void apply_gl_perm(const uint64_t* in, uint64_t* out, int limbs, int batch_count, int n, bool inverse,
                   cudaStream_t stream = 0);

// XY NTT using Phantom tables (radix-2), one NTT per poly
void xy_ntt_forward_phantom(uint64_t* data, int limbs, int batch_count, int n, cudaStream_t stream = 0);
void xy_ntt_backward_phantom(uint64_t* data, int limbs, int batch_count, int n, cudaStream_t stream = 0);

// XY NTT with GL permutation at boundary (GL order <-> standard NTT order)
void xy_ntt_forward_gl(uint64_t* data, uint64_t* tmp, int limbs, int batch_count, int n, cudaStream_t stream = 0);
void xy_ntt_backward_gl(uint64_t* data, uint64_t* tmp, int limbs, int batch_count, int n, cudaStream_t stream = 0);


// ================= API =================
// 替代 Phantom 的 Forward 接口
// data 布局: [Y][Limb][X] (Flat)
// batch_count: 对应 Y 维度 (n)
// n: 对应 X 维度 (RLWE_N)
void custom_ntt_forward(uint64_t* data, int limbs, int batch_count, int n, cudaStream_t stream = 0);

// 替代 Phantom 的 Backward 接口
void custom_ntt_backward(uint64_t* data, int limbs, int batch_count, int n, cudaStream_t stream = 0);

} // namespace matrix_fhe
