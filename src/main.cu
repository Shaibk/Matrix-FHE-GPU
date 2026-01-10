#include "../include/core/config.h"
#include "../include/core/common.cuh"
#include "../include/core/encoder.cuh"
#include "../include/core/batched_encoder.cuh"
#include "../include/core/HE.cuh"

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>

using namespace matrix_fhe;

// ===================== 辅助工具 =====================
static void cuda_check(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        std::cerr << "[CUDA ERROR] " << msg << ": " << cudaGetErrorString(e) << "\n";
        exit(1);
    }
}

static void cuda_sync(const char* msg) {
    cuda_check(cudaGetLastError(), msg);
    cuda_check(cudaDeviceSynchronize(), msg);
}

// ===================== Route-A embedding helpers =====================
// packed layout: [Y][Limb][X] with X = N_pack (4096)
// embed layout : [Y][Limb][X] with X = N_he   (8192)
// embed = [re | im] concatenation along X
__global__ void pack_re_im_to_embed(uint64_t* out_embed,
                                    const uint64_t* in_re,
                                    const uint64_t* in_im,
                                    int n_rows,
                                    int limbs,
                                    int N_pack,
                                    int N_he) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)n_rows * (size_t)limbs * (size_t)N_pack;
    if (idx >= total) return;

    // idx corresponds to a coefficient in the packed buffers
    // packed linear: ((y * limbs + limb) * N_pack + x)
    int x = (int)(idx % (size_t)N_pack);
    size_t t = idx / (size_t)N_pack;
    int limb = (int)(t % (size_t)limbs);
    int y = (int)(t / (size_t)limbs);

    // base offsets
    size_t base_embed = ((size_t)y * limbs + (size_t)limb) * (size_t)N_he;

    // out[0..N_pack-1] = re
    out_embed[base_embed + (size_t)x] = in_re[idx];
    // out[N_pack..2*N_pack-1] = im
    out_embed[base_embed + (size_t)N_pack + (size_t)x] = in_im[idx];
}

__global__ void unpack_embed_to_re_im(uint64_t* out_re,
                                      uint64_t* out_im,
                                      const uint64_t* in_embed,
                                      int n_rows,
                                      int limbs,
                                      int N_pack,
                                      int N_he) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)n_rows * (size_t)limbs * (size_t)N_pack;
    if (idx >= total) return;

    int x = (int)(idx % (size_t)N_pack);
    size_t t = idx / (size_t)N_pack;
    int limb = (int)(t % (size_t)limbs);
    int y = (int)(t / (size_t)limbs);

    size_t base_embed = ((size_t)y * limbs + (size_t)limb) * (size_t)N_he;

    out_re[idx] = in_embed[base_embed + (size_t)x];
    out_im[idx] = in_embed[base_embed + (size_t)N_pack + (size_t)x];
}

// ===================== Main Pipeline Test =====================
int main() {
    // 1) sizes
    const int n = MATRIX_N;                 // 256
    const int PHI = BATCH_PRIME_P - 1;      // 16
    const int N_pack = n * PHI;             // 4096  (encoder output length)
    const int N_he   = HE_N;              // should be 8192 for Route A
    const int LIMBS = RNS_NUM_LIMBS;        // 3
    const int n2 = n * n;                   // 65536

    std::cout << "=== FHE Pipeline Test (Route-A Embed: [re|im]) ===\n";
    std::cout << "Matrix Size: " << n << "x" << n << "\n";
    std::cout << "Batch Size (PHI): " << PHI << "\n";
    std::cout << "Packed Degree (N_pack): " << N_pack << "\n";
    std::cout << "HE Degree (N_he): " << N_he << "\n";

    if (N_he != 2 * N_pack) {
        std::cerr << "[FATAL] Route-A requires N_he == 2 * N_pack. "
                  << "Got N_he=" << N_he << ", N_pack=" << N_pack << "\n";
        return 1;
    }

    size_t total_complex_elements = (size_t)PHI * (size_t)n2;
    size_t total_pack_coeffs = (size_t)n * (size_t)LIMBS * (size_t)N_pack; // [Y][Limb][X_pack]
    size_t total_he_coeffs   = (size_t)n * (size_t)LIMBS * (size_t)N_he;   // [Y][Limb][X_he]

    std::cout << "Total Input Complex Elements: " << total_complex_elements << "\n";
    std::cout << "Total Packed Coefficients (re or im): " << total_pack_coeffs << "\n";
    std::cout << "Total HE Coefficients (embed): " << total_he_coeffs << "\n";

    // 2) init backend & key
    std::cout << ">>> Initializing HE Backend & Keys...\n";
    init_he_backend();

    SecretKey sk;
    generate_secret_key(sk, LIMBS);
    cuda_sync("Key Gen");

    // 3) input
    std::cout << ">>> Generating Input Data...\n";
    std::vector<cuDoubleComplex> h_in(total_complex_elements);
    for (int ell = 0; ell < PHI; ++ell) {
        for (int i = 0; i < n2; ++i) {
            double val_real = (double)ell + (double)i * 0.00001;
            double val_imag = (double)ell - (double)i * 0.00001;
            h_in[ell * n2 + i] = make_cuDoubleComplex(val_real, val_imag);
        }
    }

    cuDoubleComplex* d_in;
    cuda_check(cudaMalloc(&d_in, total_complex_elements * sizeof(cuDoubleComplex)), "Malloc Input");
    cuda_check(cudaMemcpy(d_in, h_in.data(),
                          total_complex_elements * sizeof(cuDoubleComplex),
                          cudaMemcpyHostToDevice),
               "H2D Input");

    // Step A: Batched Encode -> packed re/im (N_pack each)
    std::cout << ">>> Step A: Batched Encode...\n";

    uint64_t *d_packed_re, *d_packed_im;
    cuda_check(cudaMalloc(&d_packed_re, total_pack_coeffs * sizeof(uint64_t)), "Malloc Packed Re");
    cuda_check(cudaMalloc(&d_packed_im, total_pack_coeffs * sizeof(uint64_t)), "Malloc Packed Im");

    BatchedEncoder batched_enc(n);
    batched_enc.encode_packed_p17(d_in, d_packed_re, d_packed_im);
    cuda_sync("Batched Encode");

    // Route-A embedding: build one HE message (N_he) by concatenating [re|im]
    std::cout << ">>> Step A2: Route-A Embed [re|im] -> N_he...\n";

    uint64_t* d_packed_embed;
    cuda_check(cudaMalloc(&d_packed_embed, total_he_coeffs * sizeof(uint64_t)), "Malloc Packed Embed");

    {
        int thr = 256;
        int blk = (int)((total_pack_coeffs + thr - 1) / thr);
        pack_re_im_to_embed<<<blk, thr>>>(d_packed_embed,
                                          d_packed_re, d_packed_im,
                                          n, LIMBS, N_pack, N_he);
        cuda_sync("Pack Re/Im -> Embed");
    }

    // Step B: Encrypt embed (single ciphertext)
    std::cout << ">>> Step B: Encrypting (embedded)...\n";
    RLWECiphertext ct_embed;
    allocate_ciphertext(ct_embed, LIMBS);

    encrypt(d_packed_embed, sk, ct_embed);
    cuda_sync("Encrypt");

    // Step C: Decrypt embed
    std::cout << ">>> Step C: Decrypting (embedded)...\n";

    uint64_t* d_decrypted_embed;
    cuda_check(cudaMalloc(&d_decrypted_embed, total_he_coeffs * sizeof(uint64_t)), "Malloc Decrypt Embed");

    decrypt(ct_embed, sk, d_decrypted_embed);
    cuda_sync("Decrypt");

    // Step C2: split embed back to packed re/im (N_pack each)
    std::cout << ">>> Step C2: Un-embed -> packed re/im...\n";

    uint64_t *d_decrypted_re, *d_decrypted_im;
    cuda_check(cudaMalloc(&d_decrypted_re, total_pack_coeffs * sizeof(uint64_t)), "Malloc Decrypt Re");
    cuda_check(cudaMalloc(&d_decrypted_im, total_pack_coeffs * sizeof(uint64_t)), "Malloc Decrypt Im");

    {
        int thr = 256;
        int blk = (int)((total_pack_coeffs + thr - 1) / thr);
        unpack_embed_to_re_im<<<blk, thr>>>(d_decrypted_re, d_decrypted_im,
                                            d_decrypted_embed,
                                            n, LIMBS, N_pack, N_he);
        cuda_sync("Unpack Embed -> Re/Im");
    }

    // Step D: Decode (same as your original)
    std::cout << ">>> Step D: Decode...\n";

    size_t rns_eval_size = (size_t)PHI * (size_t)LIMBS * (size_t)n2;
    uint64_t *d_eval_re, *d_eval_im;
    cuda_check(cudaMalloc(&d_eval_re, rns_eval_size * sizeof(uint64_t)), "Malloc Eval Re");
    cuda_check(cudaMalloc(&d_eval_im, rns_eval_size * sizeof(uint64_t)), "Malloc Eval Im");

    batched_enc.unpack_eval_p17(d_decrypted_re, d_decrypted_im, d_eval_re, d_eval_im);
    cuda_sync("Unpack");

    cuDoubleComplex* d_out;
    cuda_check(cudaMalloc(&d_out, total_complex_elements * sizeof(cuDoubleComplex)), "Malloc Out");

    Encoder single_enc(n);
    size_t batch_rns_stride = (size_t)LIMBS * (size_t)n2;
    size_t batch_complex_stride = (size_t)n2;

    for (int ell = 0; ell < PHI; ++ell) {
        single_enc.decode(
            d_eval_re + (size_t)ell * batch_rns_stride,
            d_eval_im + (size_t)ell * batch_rns_stride,
            d_out     + (size_t)ell * batch_complex_stride
        );
    }
    cuda_sync("Single Decode");

    // Verification (same)
    std::cout << ">>> Verifying results...\n";
    std::vector<cuDoubleComplex> h_out(total_complex_elements);
    cuda_check(cudaMemcpy(h_out.data(), d_out,
                          total_complex_elements * sizeof(cuDoubleComplex),
                          cudaMemcpyDeviceToHost),
               "D2H Out");

    double max_err = 0.0;
    int err_batch = -1;
    int err_idx = -1;

    for (int i = 0; i < (int)total_complex_elements; ++i) {
        double got_r = h_out[i].x;
        double got_i = h_out[i].y;
        double ref_r = h_in[i].x;
        double ref_i = h_in[i].y;

        double err = std::hypot(got_r - ref_r, got_i - ref_i);
        if (err > max_err) {
            max_err = err;
            err_idx = i;
            err_batch = i / n2;
        }
    }

    std::cout << "Global Max Error: " << std::scientific << max_err << "\n";
    if (err_batch != -1) {
        int local_idx = err_idx % n2;
        std::cout << "Worst case at Batch " << err_batch << ", Index " << local_idx << "\n";
        std::cout << "  Exp: " << h_in[err_idx].x << " + " << h_in[err_idx].y << "i\n";
        std::cout << "  Got: " << h_out[err_idx].x << " + " << h_out[err_idx].y << "i\n";
    }

    // Cleanup
    cudaFree(d_in); cudaFree(d_out);
    cudaFree(d_packed_re); cudaFree(d_packed_im);
    cudaFree(d_packed_embed);
    cudaFree(d_decrypted_embed);
    cudaFree(d_decrypted_re); cudaFree(d_decrypted_im);
    cudaFree(d_eval_re); cudaFree(d_eval_im);
    free_ciphertext(ct_embed);

    if (max_err < 1e-4) {
        std::cout << ">>> [SUCCESS] Pipeline Verified! Data integrity preserved.\n";
        return 0;
    } else {
        std::cout << ">>> [FAILURE] Error too high. Check embedding/scale/noise logic.\n";
        return 1;
    }
}
