#pragma once
#include <cstdint>
#include <vector>
#include <cuComplex.h>

namespace matrix_fhe {

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
    void decode(const uint64_t* d_real_rns, const uint64_t* d_imag_rns, cuDoubleComplex* d_msg);

private:
    void init_complex_matrices();
};

} 
