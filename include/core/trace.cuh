#pragma once
#include <cstdint>

namespace matrix_fhe {

// B' = b(X^{-1}, Y) rewritten in basis {X^j Y^k} under X^n = i
// Mapping: for each (j,k):
//  j_dst = (-j mod n)
//  if j==0: B'[0,k] = B[0,k]
//  else:    B'[j_dst,k] = (-i) * B[j,k]   where (-i)*(a+bi) = b - a i
void map_B_to_Bprime_Xinv_twist(
    const uint64_t* B_real, const uint64_t* B_imag,
    uint64_t* Bp_real, uint64_t* Bp_imag,
    int n, int rns_limbs
);

// C = A * (B')^T   over each RNS modulus, complex arithmetic in (real,imag) limbs
void trace_gemm_ABpT_rns(
    const uint64_t* A_real, const uint64_t* A_imag,
    const uint64_t* Bp_real, const uint64_t* Bp_imag,
    uint64_t* C_real, uint64_t* C_imag,
    int n, int rns_limbs
);
void rescale_by_delta_rns(uint64_t* C_real, uint64_t* C_imag, int n, int rns_limbs,
                          uint64_t inv0, uint64_t inv1, uint64_t inv2);

} // namespace matrix_fhe
