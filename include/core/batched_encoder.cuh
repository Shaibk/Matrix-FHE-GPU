#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <cuComplex.h>

namespace matrix_fhe {

class BatchedEncoder {
public:
    explicit BatchedEncoder(int n);

    // Encode 16 matrices (ell=0..15) into one packed polynomial in W (coeff form mod Phi_17).
    // Input layout:
    //   d_msg_batch[ell*n2 + idx]  (cuDoubleComplex)
    // Output packed RNS layout (choice (1)):
    //   d_out_re[(r*3 + limb)*n2 + idx], r=0..15
    //   d_out_im[(r*3 + limb)*n2 + idx]
    void encode_packed_p17(
        const cuDoubleComplex* d_msg_batch,
        uint64_t* d_out_re,
        uint64_t* d_out_im,
        cudaStream_t stream = 0
    );

    // Unpack/evaluate packed W-coeff representation back to 16 evaluations (ell=0..15),
    // i.e., compute v_ell = sum_r V[ell,r] * c_r (mod q) for each limb and idx.
    // Input packed layout:
    //   d_in_re[(r*3 + limb)*n2 + idx]
    // Output eval layout:
    //   d_eval_re[ell*(3*n2) + limb*n2 + idx]
    //   d_eval_im[ell*(3*n2) + limb*n2 + idx]
    void unpack_eval_p17(
        const uint64_t* d_in_re,
        const uint64_t* d_in_im,
        uint64_t* d_eval_re,
        uint64_t* d_eval_im,
        cudaStream_t stream = 0
    );

    int n()  const { return n_; }
    int n2() const { return n2_; }

private:
    int n_;
    int n2_;

    void init_tables_p17(); // builds V and V^{-1} per limb and uploads to device constants
};

} // namespace matrix_fhe
