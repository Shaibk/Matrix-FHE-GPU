#pragma once
#include "config.h" // 必须包含，获取 RLWE_N, MATRIX_N, P 等定义
#include <vector>
#include <cstdint>

class DNTTTable; 
namespace matrix_fhe {

// ==========================================
// Data Structures
// ==========================================

/**
 * @brief RLWE Ciphertext (b, a) for Matrix FHE
 * * **物理内存布局 (Poly-Major / Y-Major):**
 * 数据包含 'b' 部分和 'a' 部分。
 * * b_part: [ Poly(Y=0) ][ Poly(Y=1) ] ... [ Poly(Y=n-1) ]
 * a_part: [ Poly(Y=0) ][ Poly(Y=1) ] ... [ Poly(Y=n-1) ]
 * * 每个 Poly 内部布局: [ Limb 0 ][ Limb 1 ] ... [ Limb L-1 ]
 * 每个 Limb 内部: 连续的 RLWE_N 个系数 (包含 X 和 W 维度信息)
 * * 总大小 (uint64_t): 2 * MATRIX_N * num_limbs * RLWE_N
 */
struct RLWECiphertext {
    uint64_t* data; // Device pointer. 指向 b 的起始位置。 a 位于 data + size/2 处。
    int num_limbs;
    bool is_ntt;    // 标记是否在 NTT 域

    RLWECiphertext() : data(nullptr), num_limbs(0), is_ntt(false) {}
};

/**
 * @brief Secret Key s
 * * 密钥 s(X, W) 不随 Y 变化，因此只有一份。
 * 布局: [ Limb 0 ][ Limb 1 ] ... [ Limb L-1 ]
 * 大小: num_limbs * RLWE_N
 */
struct SecretKey {
    uint64_t* data; // Device pointer. 
    int num_limbs;
};

// ==========================================
// API Functions
// ==========================================

// 初始化 HE 后端 (构建 Phantom Context, 生成 NTT 表等)
void init_he_backend();

// 获取 Phantom 的 NTT 表 (供其他模块如 BatchedEncoder 使用)
// 前向声明，避免在此处引入 Phantom 头文件

const DNTTTable& get_ntt_table();

// 内存管理
// 分配大小为 2 * MATRIX_N * limbs * RLWE_N 的显存
void allocate_ciphertext(RLWECiphertext& ct, int limbs);
void free_ciphertext(RLWECiphertext& ct);

// 密钥生成
// 生成 s 并转换为 NTT 域
void generate_secret_key(SecretKey& sk, int limbs);

// 加密
// message_coeffs: 输入明文多项式系数。
// 布局必须为 Poly-Major: [Y][Limbs][X]
// 函数内部会执行 MATRIX_N 次 NTT，然后计算 b = m - a*s + e
void encrypt(const uint64_t* message_coeffs, const SecretKey& sk, RLWECiphertext& ct);

// 解密
// output_coeffs: 输出明文多项式系数。
// 布局为 Poly-Major: [Y][Limbs][X]
// 函数内部计算 m = b + a*s，然后执行 MATRIX_N 次 INTT
void decrypt(const RLWECiphertext& ct, const SecretKey& sk, uint64_t* output_coeffs);

// 同态加法
// res = ct1 + ct2 (Point-wise addition in NTT domain)
void add_ciphertexts(const RLWECiphertext& ct1, const RLWECiphertext& ct2, RLWECiphertext& res);

// 同态乘法 (Tensor Product / Point-wise Mul)
// 计算 (b1, a1) * (b2, a2) -> (d0, d1, d2)
// d0 = b1*b2, d1 = b1*a2 + a1*b2, d2 = a1*a2
// 注意：d0, d1, d2 需要预先分配空间，大小均为 MATRIX_N * limbs * RLWE_N
void multiply_ciphertexts_raw(const RLWECiphertext& ct1, const RLWECiphertext& ct2, 
                              uint64_t* d0, uint64_t* d1, uint64_t* d2);

} // namespace matrix_fhe