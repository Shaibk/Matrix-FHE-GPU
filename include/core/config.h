#pragma once
#include <cstdint>

namespace matrix_fhe {

// ===================== Core Parameters (v2) =====================
static constexpr int LOG_N = 16;                 // Log2(RLWE_N)
static constexpr int HE_N  = (1 << LOG_N);       // 65536

static constexpr int MATRIX_N = 64;              // n
static constexpr int BATCH_SIZE = 512;           // phi
static constexpr int BATCH_PRIME_P = 771; // for W-CRT: phi(p)=512

static constexpr int PACK_N = MATRIX_N * BATCH_SIZE;   // n * phi = 32768
static constexpr int POLY_N = PACK_N;

// RNS limbs
static constexpr int RNS_NUM_LIMBS = 11;
static constexpr int P_NUM_LIMBS   = 3;

// 35-bit scale (GL scheme)
static constexpr double SCALING_FACTOR = 34359738368.0; // 2^35

// Q primes (11)
// GL constraint: q ≡ 1 (mod 197376)
static constexpr uint64_t RNS_MODULI[RNS_NUM_LIMBS] = {
    17592186435073ULL, // bits=45
    17182765057ULL,    // bits=35
    17184541441ULL,    // bits=35
    17186120449ULL,    // bits=35
    17186515201ULL,    // bits=35
    17186909953ULL,    // bits=35
    17188883713ULL,    // bits=35
    17190462721ULL,    // bits=35
    17190857473ULL,    // bits=35
    17191844353ULL,    // bits=35
    17192831233ULL     // bits=35
};

// P primes (3) - reserved for future key switching
static constexpr uint64_t P_MODULI[P_NUM_LIMBS] = {
    18014398515156481ULL, // bits=55
    549757491457ULL,      // bits=40
    549759662593ULL       // bits=40
};

} // namespace matrix_fhe
