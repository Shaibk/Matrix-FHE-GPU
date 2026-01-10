#pragma once
#include <cstdint>

namespace matrix_fhe {

static constexpr int MAT_N = 256; 
static constexpr int RNS_NUM_LIMBS = 3; 
// Matrix Dimensions
// n = 256 (Power of 2)
#define MATRIX_N 256

// Batch Prime
// p = 17 (Fermat Prime, phi(p)=16)
#define BATCH_PRIME_P 17
#define PACK_N 4096

// ===== HE / RLWE layer =====
#define HE_N          (2 * PACK_N)   // 8192

// 140-bit Q, 40-bit Delta
static constexpr double SCALING_FACTOR = 1099511627776.0; // 2^40
// config.h
constexpr int POLY_N = MATRIX_N * (BATCH_PRIME_P - 1); // 256*16=4096

// 工业级 RNS 模数 (满足 1 mod 2N，约 47-bit)
// 总模数 Q ≈ 141 bits
static constexpr uint64_t RNS_MODULI[3] = {
    140737433174017ULL, // 0x7FFFFCB60001
    140737361870849ULL, // 0x7FFFF8760001
    140737355186177ULL // 0x7FFFF8100001
};
} // namespace matrix_fhe
