#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <cuComplex.h>

namespace matrix_fhe {

class BatchedEncoder {
public:
    explicit BatchedEncoder(int n);

    // Encode BATCH_SIZE matrices and output W-NTT evaluation layout (RNS, matrix-major).
    // Input layout:
    //   d_msg_batch[ell*n2 + idx]  (cuDoubleComplex)
    // Output layout (W-NTT eval):
    //   d_out_re[ell*(limbs*n2) + limb*n2 + idx]
    //   d_out_im[ell*(limbs*n2) + limb*n2 + idx]
    void encode_to_wntt_eval(
        const cuDoubleComplex* d_msg_batch,
        uint64_t* d_out_re,
        uint64_t* d_out_im,
        cudaStream_t stream = 0
    );

    // Unpack packed W-batched CRT representation back to BATCH_SIZE evaluations,
    // i.e., compute v_ell = sum_r V[ell,r] * c_r (mod q) for each limb and idx.
    // Input layout:
    //   d_in_re[ell*(limbs*n2) + limb*n2 + idx]
    // Output eval layout:
    //   d_eval_re[ell*(limbs*n2) + limb*n2 + idx]
    //   d_eval_im[ell*(limbs*n2) + limb*n2 + idx]
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

    void init_tables_p17(); // uploads moduli to device constants
};

} // namespace matrix_fhe
