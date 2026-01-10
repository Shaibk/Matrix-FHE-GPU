#include "core/HE.cuh"
#include "core/common.cuh" 
#include <vector>
#include <iostream>
#include <random>


// === Phantom Headers ===
#include "../../extern/phantom-fhe/include/context.cuh"
#include "../../extern/phantom-fhe/include/ntt.cuh"
#include "../../extern/phantom-fhe/include/host/encryptionparams.h"
#include "../../include/core/config.h"

namespace matrix_fhe {

// 全局单例 Context
static PhantomContext* g_phantom_ctx = nullptr;

// 常量显存
__constant__ uint64_t d_he_moduli[matrix_fhe::RNS_NUM_LIMBS];

// ==========================================
// 1. Initialization
// ==========================================

void init_he_backend() {
    if (g_phantom_ctx != nullptr) return;

    phantom::EncryptionParameters parms(phantom::scheme_type::ckks);
    parms.set_poly_modulus_degree(HE_N);
    
    std::vector<phantom::arith::Modulus> mods;
    for(int i=0; i<RNS_NUM_LIMBS; ++i) {
        mods.emplace_back(RNS_MODULI[i]);
    }
    parms.set_coeff_modulus(mods);

    cudaStream_t stream = 0;
    g_phantom_ctx = new PhantomContext(parms);
    
    cudaMemcpyToSymbol(d_he_moduli, matrix_fhe::RNS_MODULI, sizeof(matrix_fhe::RNS_MODULI));


    std::cout << ">>> Phantom HE Backend Initialized. Poly Degree: " <<HE_N << "\n";
    std::cout << ">>> Matrix Config: n=" << MATRIX_N << ", BATCH_PRIME_P=" << BATCH_PRIME_P 
              << ", logical_poly_size=" << MATRIX_N * (BATCH_PRIME_P-1) << "\n";
}

const DNTTTable& get_ntt_table() {
    if (!g_phantom_ctx) {
        std::cerr << "Error: HE backend not initialized! Call init_he_backend() first.\n";
        exit(1);
    }
    return g_phantom_ctx->gpu_rns_tables();
}

// ==========================================
// 2. Custom Kernels (Arithmetic)
// ==========================================

// --- Encrypt Kernel (Poly-Major Layout) ---
__global__ void encrypt_kernel_poly_major(
    const uint64_t* m,
    const uint64_t* a,
    const uint64_t* s,
    uint64_t* b,
    size_t total_len,
    int limbs)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_len) return;

    constexpr int N = HE_N;
    int single_poly_len = limbs * N;
    int offset_in_poly = idx % single_poly_len;
    int l = offset_in_poly / N;

    uint64_t q = d_he_moduli[l];
    int s_off = offset_in_poly;

    uint64_t as = mul_mod(a[idx], s[s_off], q);
    b[idx] = sub_mod(m[idx], as, q);
}


// --- Decrypt Kernel (Poly-Major Layout) ---
__global__ void decrypt_kernel_poly_major(const uint64_t* b, const uint64_t* a, const uint64_t* s, 
                                          uint64_t* m, size_t total_len, int poly_degree, int limbs) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_len) return;

    int single_poly_len = limbs * poly_degree;
    int offset_in_poly = idx % single_poly_len;
    int l = offset_in_poly / poly_degree;
    
    uint64_t q = d_he_moduli[l];
    int s_off = offset_in_poly;

    uint64_t val_b = b[idx];
    uint64_t val_a = a[idx];
    uint64_t val_s = s[s_off];

    uint64_t as = mul_mod(val_a, val_s, q);
    m[idx] = add_mod(val_b, as, q);
}

// --- Noise Generation Kernels ---
__global__ void uniform_random_kernel(uint64_t* data, size_t total_len, int limbs) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_len) return;

    const int N = HE_N;                 // 用 HE_N，不要 HE_N
    const int single_poly_size = limbs * N;

    // 只需要 limb_idx 用来取对应的模数
    int offset_in_poly = (int)(idx % (size_t)single_poly_size);
    int limb_idx = offset_in_poly / N;

    uint64_t q = d_he_moduli[limb_idx];

    uint64_t seed = 123456789ULL + idx;
    seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
    data[idx] = seed % q;
}


__device__ __forceinline__ uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

__global__ void gaussian_noise_kernel(uint64_t* data, size_t total_len, int limbs) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_len) return;

    const int N = HE_N;
    const int single_poly_len = limbs * N;

    // Layout: [poly][limb][coeff]
    int offset_in_poly = (int)(idx % (size_t)single_poly_len);
    int limb_idx       = offset_in_poly / N;

    uint64_t q = d_he_moduli[limb_idx];

    // 让每个 (idx, limb) 都独立，避免 limb 强相关
    uint64_t seed = (0xD6E8FEB86659FD93ULL ^ (uint64_t)idx)
                  + (uint64_t)limb_idx * 0xA5A3564E27F2C2B1ULL;
    uint64_t r = splitmix64(seed);

    // centered binomial 近似：noise = popcount(low16) - popcount(high16) ∈ [-16,16]
    int a = __popc((unsigned int)( r        & 0xFFFFu));
    int b = __popc((unsigned int)((r >> 16) & 0xFFFFu));
    int noise = a - b;  // 零均值、小幅度

    // 映射到模 q（把负数写成 q - |noise|）
    uint64_t out = (noise >= 0) ? (uint64_t)noise : (q - (uint64_t)(-noise));
    data[idx] = out;
}



__global__ void add_ct_kernel(const uint64_t* b1, const uint64_t* a1,
                              const uint64_t* b2, const uint64_t* a2,
                              uint64_t* b_out, uint64_t* a_out, size_t total_len, int limbs) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_len) return;

    int N = HE_N;
    int single_poly_len = limbs * N;
    int offset_in_poly = idx % single_poly_len;
    int l = offset_in_poly / N;
    uint64_t q = d_he_moduli[l];

    b_out[idx] = add_mod(b1[idx], b2[idx], q);
    a_out[idx] = add_mod(a1[idx], a2[idx], q);
}

__global__ void mul_tensor_kernel(const uint64_t* b1, const uint64_t* a1,
                                  const uint64_t* b2, const uint64_t* a2,
                                  uint64_t* d0, uint64_t* d1, uint64_t* d2,
                                  size_t total_len, int limbs) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_len) return;

    int N = HE_N;
    int single_poly_len = limbs * N;
    int offset_in_poly = idx % single_poly_len;
    int l = offset_in_poly / N;
    uint64_t q = d_he_moduli[l];

    uint64_t vb1 = b1[idx]; uint64_t va1 = a1[idx];
    uint64_t vb2 = b2[idx]; uint64_t va2 = a2[idx];

    d0[idx] = mul_mod(vb1, vb2, q);
    
    uint64_t term1 = mul_mod(vb1, va2, q);
    uint64_t term2 = mul_mod(va1, vb2, q);
    d1[idx] = add_mod(term1, term2, q);
    
    d2[idx] = mul_mod(va1, va2, q);
}

// ==========================================
// 3. Host API Implementation
// ==========================================

void allocate_ciphertext(RLWECiphertext& ct, int limbs) {
    ct.num_limbs = limbs;
    ct.is_ntt = true;
    size_t total_coeffs = (size_t)MATRIX_N * limbs * HE_N; 
    size_t size = 2 * total_coeffs * sizeof(uint64_t);
    cudaMalloc(&ct.data, size);
    cudaMemset(ct.data, 0, size);
}

void free_ciphertext(RLWECiphertext& ct) {
    if(ct.data) {
        cudaFree(ct.data);
        ct.data = nullptr;
    }
}
__global__ void ternary_secret_kernel(uint64_t* s, size_t total_len, int limbs) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_len) return;

    const int N = HE_N;
    const int single_poly_len = limbs * N;

    // [poly][limb][coeff]
    int offset_in_poly = (int)(idx % (size_t)single_poly_len);
    int limb = offset_in_poly / N;
    int coeff = offset_in_poly - limb * N;

    // 让同一 coeff 在所有 limb 上一致：seed 不包含 limb
    // 同时为了让不同 poly 不同，加入 poly_idx
    int poly = (int)(idx / (size_t)single_poly_len);

    uint64_t t = (uint64_t)poly * 1315423911ULL + (uint64_t)coeff * 2654435761ULL;
    // 取三值：0,1,2 -> 映射到 0,+1,-1
    int r = (int)((t * 11400714819323198485ULL) % 3ULL);

    uint64_t q = d_he_moduli[limb];
    if (r == 0) s[idx] = 0;
    else if (r == 1) s[idx] = 1;
    else s[idx] = q - 1; // -1 mod q
}

void generate_secret_key(SecretKey& sk, int limbs) {
    sk.num_limbs = limbs;

    size_t n = (size_t)HE_N * (size_t)limbs;   // ✅ HE 层必须 HE_N

    cudaMalloc(&sk.data, n * sizeof(uint64_t));
    cudaMemset(sk.data, 0, n * sizeof(uint64_t));

    int thr = 256;
    int blk = (n + thr - 1) / thr;

    ternary_secret_kernel<<<blk, thr>>>(sk.data, n, limbs);

    // ✅ NTT over HE_N
    nwt_2d_radix8_forward_inplace(sk.data, get_ntt_table(), limbs, 0, 0);

    // ---- 运行时钉死：确认这一版真的跑到了 ----
    // 只打印一次（避免刷屏）
    cudaDeviceSynchronize();
    printf("[DBG] generate_secret_key: using ternary_secret_kernel, HE_N=%d\n", HE_N);
}


// 确保 b 和 e 都在 Eval 域，且模数 q 匹配
__global__ void add_error_inplace_kernel(uint64_t* b, const uint64_t* e, size_t total_len, int limbs) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_len) return;

    // 解析当前 idx 对应的模数 q
    // 假设布局为 Poly-Major: [Y][Limb][X]
    int N = HE_N; // 务必使用全局定义的 N
    int single_poly_size = limbs * N;
    
    // 计算当前 idx 处于哪个 limb
    int offset_in_poly = (int)(idx % (size_t)single_poly_size);
    int limb_idx = offset_in_poly / N;

    uint64_t q = d_he_moduli[limb_idx];

    // 执行模加: b = (b + e) mod q
    b[idx] = add_mod(b[idx], e[idx], q);
}


void encrypt(const uint64_t* message_coeffs, const SecretKey& sk, RLWECiphertext& ct) {
    // 1. 维度计算
    // 注意：这里使用全局配置 HE_N
    size_t single_poly_len = (size_t)ct.num_limbs * HE_N; 
    size_t total_coeffs = (size_t)MATRIX_N * single_poly_len;
    size_t total_bytes = total_coeffs * sizeof(uint64_t);

    uint64_t* d_b = ct.data;
    uint64_t* d_a = ct.data + total_coeffs;

    // ==========================================
    // 1. 处理消息 m (b部分)
    // ==========================================
    // Copy Coeffs -> d_b
    cudaMemcpy(d_b, message_coeffs, total_bytes, cudaMemcpyDeviceToDevice);

    // NTT: Coeff -> Eval
    // 针对每个 Poly (Y维度) 独立做 NTT
    for (int i = 0; i < MATRIX_N; ++i) {
        uint64_t* poly_ptr = d_b + i * single_poly_len;
        // 使用 Phantom 或 Custom NTT
        nwt_2d_radix8_forward_inplace(poly_ptr, get_ntt_table(), ct.num_limbs, 0, 0);
    }

    // ==========================================
    // 2. 处理公钥部分 a
    // ==========================================
    // Generate Uniform Random (Coeff domain)
    int thr = 256;
    int blk = (total_coeffs + thr - 1) / thr;
    uniform_random_kernel<<<blk, thr>>>(d_a, total_coeffs, ct.num_limbs);
    
    // NTT: Coeff -> Eval
    for (int i = 0; i < MATRIX_N; ++i) {
        uint64_t* poly_ptr = d_a + i * single_poly_len;
        nwt_2d_radix8_forward_inplace(poly_ptr, get_ntt_table(), ct.num_limbs, 0, 0);
    }

    // ==========================================
    // 3. 处理噪声 e (CRITICAL STEP)
    // ==========================================
    uint64_t* d_e;
    cudaMalloc(&d_e, total_bytes);

    // A. 生成小高斯噪声 (Coeff domain)
    gaussian_noise_kernel<<<blk, thr>>>(d_e, total_coeffs, ct.num_limbs);
    
    // B. [关键修复] 对噪声做 NTT: Coeff -> Eval
    // 必须在加法之前将噪声变换到 Eval 域！
    for (int i = 0; i < MATRIX_N; ++i) {
        uint64_t* poly_ptr = d_e + i * single_poly_len;
        nwt_2d_radix8_forward_inplace(poly_ptr, get_ntt_table(), ct.num_limbs, 0, 0);
    }

    // ==========================================
    // 4. 融合计算
    // ==========================================
    
    // Step 1: b = m - a * s (都在 Eval 域)
    // 注意：需要传入 HE_N 参数供 kernel 内部计算索引
    encrypt_kernel_poly_major<<<blk, thr>>>(
        d_b, d_a, sk.data, d_b, 
        total_coeffs, ct.num_limbs
    );

    // Step 2: b = b + e (都在 Eval 域)
    add_error_inplace_kernel<<<blk, thr>>>(d_b, d_e, total_coeffs, ct.num_limbs);

    // 清理
    cudaFree(d_e);
}
void decrypt(const RLWECiphertext& ct, const SecretKey& sk, uint64_t* output_coeffs) {
    size_t single_poly_len = (size_t)ct.num_limbs * HE_N;
    size_t total_coeffs = (size_t)MATRIX_N * single_poly_len;

    uint64_t* d_b = ct.data;
    uint64_t* d_a = ct.data + total_coeffs;

    int thr = 256;
    int blk = (total_coeffs + thr - 1) / thr;

    decrypt_kernel_poly_major<<<blk, thr>>>(d_b, d_a, sk.data, output_coeffs, 
                                            total_coeffs, HE_N, ct.num_limbs);

    for (int i = 0; i < MATRIX_N; ++i) {
        uint64_t* poly_ptr = output_coeffs + i * single_poly_len;
        // [Fix] 使用 phantom:: 且 stream=0
        nwt_2d_radix8_backward_inplace(poly_ptr, get_ntt_table(), ct.num_limbs, 0, 0);
    }
}

void add_ciphertexts(const RLWECiphertext& ct1, const RLWECiphertext& ct2, RLWECiphertext& res) {
    size_t total_coeffs = (size_t)MATRIX_N * ct1.num_limbs * HE_N;
    
    uint64_t* b1 = ct1.data;
    uint64_t* a1 = ct1.data + total_coeffs;
    uint64_t* b2 = ct2.data;
    uint64_t* a2 = ct2.data + total_coeffs;
    
    uint64_t* b_res = res.data;
    uint64_t* a_res = res.data + total_coeffs;

    int thr = 256;
    int blk = (total_coeffs + thr - 1) / thr;
    
    add_ct_kernel<<<blk, thr>>>(b1, a1, b2, a2, b_res, a_res, total_coeffs, res.num_limbs);
}

void multiply_ciphertexts_raw(const RLWECiphertext& ct1, const RLWECiphertext& ct2, 
                              uint64_t* d0, uint64_t* d1, uint64_t* d2) {
    size_t total_coeffs = (size_t)MATRIX_N * ct1.num_limbs * HE_N;

    uint64_t* b1 = ct1.data;
    uint64_t* a1 = ct1.data + total_coeffs;
    uint64_t* b2 = ct2.data;
    uint64_t* a2 = ct2.data + total_coeffs;

    int thr = 256;
    int blk = (total_coeffs + thr - 1) / thr;
    
    mul_tensor_kernel<<<blk, thr>>>(b1, a1, b2, a2, d0, d1, d2, total_coeffs, ct1.num_limbs);
}

} // namespace matrix_fhe