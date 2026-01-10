#pragma once
#include <cstdint>

namespace matrix_fhe {

/**
 * @brief Batched B -> B' transformation.
 * * Implements the pre-processing for Theorem 3.8/3.9.
 * Transforms a batch of matrices B into B' such that C = A * (B')^T corresponds to the trace operation.
 * * @param B_real Input real parts [batch_size][limbs][n][n]
 * @param B_imag Input imag parts [batch_size][limbs][n][n]
 * @param Bp_real Output real parts 
 * @param Bp_imag Output imag parts
 * @param n Matrix dimension (256)
 * @param rns_limbs Number of RNS limbs (3)
 * @param batch_size Number of batched matrices (phi(p) = 16)
 */
void map_B_to_Bprime_batched(
    const uint64_t* B_real, const uint64_t* B_imag,
    uint64_t* Bp_real, uint64_t* Bp_imag,
    int n, int rns_limbs, int batch_size
);

/**
 * @brief Batched Trace GEMM: C = A * (B')^T
 * * Performs phi(p) independent matrix multiplications in parallel.
 * * @param A_real Input A [batch_size][limbs][n][n]
 * @param Bp_real Input B' (transformed)
 * @param C_real Output C
 * @param n Matrix dimension
 * @param rns_limbs Number of RNS limbs
 * @param batch_size Number of batched matrices
 */
void trace_gemm_batched(
    const uint64_t* A_real, const uint64_t* A_imag,
    const uint64_t* Bp_real, const uint64_t* Bp_imag,
    uint64_t* C_real, uint64_t* C_imag,
    int n, int rns_limbs, int batch_size
);

/**
 * @brief Batched Rescaling
 * * Rescales the result C by dividing by Delta (scaling factor) for all batches.
 */
void rescale_by_delta_batched(
    uint64_t* C_real, uint64_t* C_imag, 
    int n, int rns_limbs, int batch_size,
    uint64_t inv0, uint64_t inv1, uint64_t inv2
);

} // namespace matrix_fhe