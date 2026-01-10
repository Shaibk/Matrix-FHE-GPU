#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace matrix_fhe {

// NTT 配置表结构
struct NTTTable {
    uint64_t* d_psi_powers;      // Forward 旋转因子 (Powers of psi)
    uint64_t* d_psi_inv_powers;  // Inverse 旋转因子
    uint64_t* d_n_inv;           // N^-1 mod q
    int       modulus_count;
    int       n;
};

// 初始化 NTT 表 (在 Host 端计算并上传)
void init_ntt_tables_manual(int n, int limbs);

// 获取表实例
const NTTTable& get_manual_ntt_table();

// ================= API =================
// 替代 Phantom 的 Forward 接口
// data 布局: [Y][Limb][X] (Flat)
// batch_count: 对应 Y 维度 (n)
// n: 对应 X 维度 (RLWE_N)
void custom_ntt_forward(uint64_t* data, int limbs, int batch_count, int n, cudaStream_t stream = 0);

// 替代 Phantom 的 Backward 接口
void custom_ntt_backward(uint64_t* data, int limbs, int batch_count, int n, cudaStream_t stream = 0);

} // namespace matrix_fhe