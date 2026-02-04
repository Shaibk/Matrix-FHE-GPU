#pragma once
#include <cstdint>
#include <vector>
#include <cuComplex.h>
#include <cuda_runtime.h>

namespace matrix_fhe {

// Shared decode helpers (used by HE decrypt+decode path)
__global__ void dequantize_exact_kernel(
    const uint64_t* d_real_rns,
    const uint64_t* d_imag_rns,
    cuDoubleComplex* d_out,
    int n2,
    double scaling_factor,
    int limbs
);

__global__ void crt_compose_centerlift_kernel(
    const uint64_t* d_in_rns,
    int64_t* d_out_centered,
    int n2,
    int limbs
);

void crt_compose_centerlift_big(
    const uint64_t* d_in_rns,
    uint64_t* d_out_mag,
    uint8_t* d_out_neg,
    int n2,
    int limbs,
    cudaStream_t stream = 0
);

__global__ void mat_mul_kernel_complex(
    const cuDoubleComplex* A,
    const cuDoubleComplex* B,
    cuDoubleComplex* C,
    int n
);

class Encoder {
public:
    int n;
    cuDoubleComplex* d_V_cx;        
    cuDoubleComplex* d_V_cx_T;      
    cuDoubleComplex* d_V_inv_cx;    
    cuDoubleComplex* d_V_inv_cx_T;  

    Encoder(int n);
    ~Encoder();

    // [New Interface] Phantom Style SoA
    // d_real_rns: 指向大小为 n * num_limbs 的数组
    // 布局: [Limb0 (n) | Limb1 (n) | Limb2 (n)]
    void encode(const cuDoubleComplex* d_msg, uint64_t* d_real_rns, uint64_t* d_imag_rns);
    // Decode one lane from RNS eval (XY only). This helper is used in encode/decode
    // validation paths; encrypted decode pipeline uses HE.cu kernels instead.
    void decode_lane_from_rns_eval(const uint64_t* d_real_rns, const uint64_t* d_imag_rns, cuDoubleComplex* d_msg);
    void decode_from_eval_complex(const cuDoubleComplex* d_eval, cuDoubleComplex* d_msg);
    void idft2(const cuDoubleComplex* d_eval_xy, cuDoubleComplex* d_coeff_xy);

private:
    void init_complex_matrices();
};

} 
