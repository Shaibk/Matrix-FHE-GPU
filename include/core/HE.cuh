#pragma once
#include "config.h" // 必须包含，获取 RLWE_N, MATRIX_N, P 等定义
#include <vector>
#include <cstdint>
#include <cuComplex.h>

class DNTTTable; 
namespace matrix_fhe {

// Copy device-side moduli table to host (initialized in init_he_backend)
void copy_device_moduli(uint64_t* h_out, int count);

// ==========================================
// Data Structures
// ==========================================

/**
 * @brief RLWE Ciphertext (b, a) for Matrix FHE
 * * **物理内存布局 (W-CRT eval / Matrix-Major):**
 * 数据包含 'b' 部分和 'a' 部分。
 * * b_part: [ W=0 ][ W=1 ] ... [ W=phi-1 ]
 * a_part: [ W=0 ][ W=1 ] ... [ W=phi-1 ]
 * * 每个 W 内部布局: [ Limb 0 ][ Limb 1 ] ... [ Limb L-1 ]
 * 每个 Limb 内部: 连续的 n*n 个系数 (Y major, X contiguous)
 * * 总大小 (uint64_t): 2 * phi * n * num_limbs * n
 */
struct RLWECiphertext {
    uint64_t* data; // Device pointer. 指向 b 的起始位置。 a 位于 data + size/2 处。
    int num_limbs;
    bool is_ntt;    // 标记 X 维度是否在 NTT 域（W 维度保持 CRT-eval）

    RLWECiphertext() : data(nullptr), num_limbs(0), is_ntt(false) {}
};

/**
 * @brief Secret Key s
 * * 密钥 s(X) 不随 Y/W 变化，因此只有一份。
 * 布局: [ Limb 0 ][ Limb 1 ] ... [ Limb L-1 ]
 * 大小: num_limbs * n
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
const DNTTTable& get_xy_ntt_table();

// W-CRT forward (Phi_p evaluation) for matrix layout
void wntt_forward_matrix(const uint64_t* in, uint64_t* out, int n, int limbs, int phi, cudaStream_t stream = 0);
void wntt_inverse_matrix(const uint64_t* in_eval, uint64_t* out_coeff, int n, int limbs, int phi, cudaStream_t stream = 0);
void wntt_forward_centered(const int64_t* in_coeff_centered, int64_t* out_eval_centered,
                           int n, int phi, cudaStream_t stream = 0);
void wntt_inverse_centered(const int64_t* in_eval_centered, int64_t* out_coeff_centered,
                           int n, int phi, cudaStream_t stream = 0);
// W-DFT (decode semantics, Q-independent): centered coeff pair -> eval pair (double)
void wdft_forward_centered_pair(const int64_t* in_re_centered, const int64_t* in_im_centered,
                                double* out_re_eval, double* out_im_eval,
                                int n, int phi, cudaStream_t stream = 0);
// Inverse W-DFT: eval pair (double) -> coeff pair (double)
void wdft_inverse_pair(const double* in_re_eval, const double* in_im_eval,
                       double* out_re_coeff, double* out_im_coeff,
                       int n, int phi, cudaStream_t stream = 0);

// 内存管理
// 分配大小为 2 * phi * n * limbs * n 的显存
void allocate_ciphertext(RLWECiphertext& ct, int limbs);
void free_ciphertext(RLWECiphertext& ct);

// 密钥生成
// 生成 s (coeff) 并转换为 X-NTT 域
void generate_secret_key(SecretKey& sk, int limbs);

// 加密
// message_coeffs: 输入明文多项式系数（已在 W-CRT eval 域）。
// 布局必须为 W-CRT eval Matrix-Major: [W][Limbs][n*n]
// 函数内部生成 a/s/e 于 W-coeff 域，再映射到 W-CRT eval 域，
// 然后做 X-NTT：a_ntt = NTT_X(a_eval), b_ntt = NTT_X(m - a*s + e_eval)
void encrypt(const uint64_t* message_coeffs, const SecretKey& sk, RLWECiphertext& ct);
// Encrypt complex pair: share the same 'a' for re/im, independent errors
void encrypt_pair(const uint64_t* msg_re, const uint64_t* msg_im,
                  const SecretKey& sk, RLWECiphertext& ct_re, RLWECiphertext& ct_im);

// 解密
// decrypt_and_decode: decrypt re/im pair, filter noise in W coeff, then decode back to complex matrices
// output_msg: 输出复数矩阵 [W][n*n]
void decrypt_and_decode(const RLWECiphertext& ct_re, const RLWECiphertext& ct_im,
                        const SecretKey& sk, cuDoubleComplex* output_msg);

// Debug helper: decrypt ciphertext to W-CRT eval matrix layout [W][limb][n*n]
void decrypt_to_eval_matrix(const RLWECiphertext& ct, const SecretKey& sk, uint64_t* out_eval_matrix);

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
