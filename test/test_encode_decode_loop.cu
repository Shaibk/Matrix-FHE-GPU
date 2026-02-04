#include "../include/core/config.h"
#include "../include/core/encoder.cuh"
#include "../include/core/batched_encoder.cuh"

#include <cuda_runtime.h>
#include <cuComplex.h>

#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <cstdlib>

using namespace matrix_fhe;

// ===================== 辅助工具函数 =====================
static inline void cuda_check(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        std::cerr << "[CUDA ERROR] " << msg << ": " << cudaGetErrorString(e) << "\n";
        std::exit(1);
    }
}

static inline void cuda_sync(const char* msg) {
    cuda_check(cudaGetLastError(), msg);
    cuda_check(cudaDeviceSynchronize(), msg);
}

// ---------------------------------------------------------
// Encode -> Layout Transpose -> Decode 环回测试
// ---------------------------------------------------------
int main() {
    // 1. 参数设置
    constexpr int PHI = BATCH_SIZE;
    const int n       = MATRIX_N;
    const int n2      = n * n;
    const double DELTA = SCALING_FACTOR / n; // 缩放因子

    std::cout << "=== Test: Encode -> PolyMajor Layout -> Decode Loopback ===\n";
    std::cout << "Matrix N: " << n << ", Batch Size: " << PHI << "\n";
    std::cout << "Target Layout: [Y=" << n << "][Limb=" << RNS_NUM_LIMBS << "][X=" << n*PHI << "] (Poly-Major)\n";

    // ---------------------------------------------------------
    // 2. 初始化 Host 数据 (生成随机测试数据)
    // ---------------------------------------------------------
    std::vector<cuDoubleComplex> h_Input(PHI * n2);
    
    // 我们用一个确定的模式填充数据，方便Debug
    // Value = (Batch * 10000) + (Row * 100) + Col
    for (int l = 0; l < PHI; ++l) {
        for (int i = 0; i < n2; ++i) {
            int r = i / n;
            int c = i % n;
            double val = (double)(l * 10000 + r * 100 + c);
            h_Input[l * n2 + i] = make_cuDoubleComplex(val, -val); // Real, -Real
        }
    }

    // ---------------------------------------------------------
    // 3. 设备内存分配
    // ---------------------------------------------------------
    cuDoubleComplex *d_Input;
    cuda_check(cudaMalloc(&d_Input, PHI * n2 * sizeof(cuDoubleComplex)), "malloc input");
    cuda_check(cudaMemcpy(d_Input, h_Input.data(), PHI * n2 * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice), "H2D Input");

    // Poly-Major Layout Buffer (用于存放 Encode 后的结果)
    // Size = MATRIX_N * Limbs * PACK_N (PACK_N = n * PHI)
    size_t poly_major_size = (size_t)MATRIX_N * RNS_NUM_LIMBS * PACK_N; 
    
    uint64_t *d_Packed_Re, *d_Packed_Im;
    cuda_check(cudaMalloc(&d_Packed_Re, poly_major_size * sizeof(uint64_t)), "malloc packed re");
    cuda_check(cudaMalloc(&d_Packed_Im, poly_major_size * sizeof(uint64_t)), "malloc packed im");

    // Eval Domain Buffer (用于 Unpack 还原回来的结果)
    // Size = Batch * Limbs * Space
    size_t eval_size = (size_t)PHI * RNS_NUM_LIMBS * n2;
    uint64_t *d_Eval_Re, *d_Eval_Im;
    cuda_check(cudaMalloc(&d_Eval_Re, eval_size * sizeof(uint64_t)), "malloc eval re");
    cuda_check(cudaMalloc(&d_Eval_Im, eval_size * sizeof(uint64_t)), "malloc eval im");

    // Output Buffer (最终 Decode 出来的复数)
    cuDoubleComplex *d_Output;
    cuda_check(cudaMalloc(&d_Output, PHI * n2 * sizeof(cuDoubleComplex)), "malloc output");

    // ---------------------------------------------------------
    // 4. 执行 Encode Pipeline
    // ---------------------------------------------------------
    BatchedEncoder batched_enc(n);

    std::cout << ">> Step 1: Batched Encode (Input -> PolyMajor)...\n";
    // 这步会调用 SingleEncoder -> pack_w_phi16_kernel (Transpose)
    batched_enc.encode_to_wntt_eval(d_Input, d_Packed_Re, d_Packed_Im);
    cuda_sync("Encode Packed");

    std::cout << ">> Step 2: Batched Unpack (PolyMajor -> Eval)...\n";
    // 这步会调用 eval_w_phi16_kernel (Inverse Transpose)
    // 注意：这里得到的是 RNS 域的数据，尚未转回复数
    batched_enc.unpack_eval_p17(d_Packed_Re, d_Packed_Im, d_Eval_Re, d_Eval_Im);
    cuda_sync("Unpack Eval");

    std::cout << ">> Step 3: Single Decode (Eval -> Output)...\n";
    // 使用 SingleEncoder 对每个 Batch 进行 Decode
    Encoder single_enc(n);
    size_t blk_words = (size_t)RNS_NUM_LIMBS * n2; // 单个 Batch 的 RNS 大小

    for (int l = 0; l < PHI; ++l) {
        single_enc.decode_lane_from_rns_eval(
            d_Eval_Re + l * blk_words,
            d_Eval_Im + l * blk_words,
            d_Output  + l * n2
        );
    }
    cuda_sync("Decode Loop");

    // ---------------------------------------------------------
    // 5. 结果验证
    // ---------------------------------------------------------
    std::vector<cuDoubleComplex> h_Output(PHI * n2);
    cuda_check(cudaMemcpy(h_Output.data(), d_Output, PHI * n2 * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost), "D2H Output");

    std::cout << "\n>>> Verification Results:\n";
    
    double max_err = 0.0;
    int err_idx = -1;
    int err_batch = -1;

    for (int l = 0; l < PHI; ++l) {
        for (int i = 0; i < n2; ++i) {
            size_t idx = l * n2 + i;
            
            // Expected
            double exp_r = h_Input[idx].x;
            double exp_i = h_Input[idx].y;

            // Got (需要除以 Scaling Factor)
            // 注意：SingleEncoder::decode_lane_from_rns_eval 内部如果没除 Delta，这里需要除
            // 查看你的 Encoder::decode_lane_from_rns_eval 实现，通常它最后输出的是带 Scaling 的值
            // 或者它内部已经处理了？
            // 通常 CKKS decode 出来就是原值（如果内部处理了）。
            // 但你的 quantize 用了 SCALING_FACTOR，decode 时如果是 dequantize_exact_kernel 
            // 且该 kernel 内部除以了 Delta，那就不需要再除。
            // 假设 dequantize_exact_kernel 内部做了 / delta。
            
            double got_r = h_Output[idx].x;
            double got_i = h_Output[idx].y;

            // 如果结果相差巨大（比如差 10^12 倍），说明 Decode 内部没除 Delta。
            // 你的 dequantize_exact_kernel 结尾有 return final_val / delta; 所以不需要额外除。

            double err = std::hypot(got_r - exp_r, got_i - exp_i);
            
            if (err > max_err) {
                max_err = err;
                err_idx = i;
                err_batch = l;
            }
        }
    }

    std::cout << "Global Max Error: " << std::scientific << max_err << "\n";
    if (err_batch != -1) {
        int r = err_idx / n;
        int c = err_idx % n;
        size_t idx = err_batch * n2 + err_idx;
        std::cout << "Worst Case at Batch " << err_batch << ", Pos(" << r << "," << c << ")\n";
        std::cout << "Expected: " << h_Input[idx].x << " + " << h_Input[idx].y << "i\n";
        std::cout << "Got:      " << h_Output[idx].x << " + " << h_Output[idx].y << "i\n";
    }

    // 阈值判定
    if (max_err < 1e-3) { // 稍微放宽一点，因为是 float/double 转换且经过了多次变换
        std::cout << "\n>>> [PASSED] Layout Transformation Logic is Correct!\n";
    } else {
        std::cout << "\n>>> [FAILED] Error too high. Check Indexing Logic.\n";
    }

    // Cleanup
    cudaFree(d_Input);
    cudaFree(d_Packed_Re); cudaFree(d_Packed_Im);
    cudaFree(d_Eval_Re);   cudaFree(d_Eval_Im);
    cudaFree(d_Output);

    return 0;
}
