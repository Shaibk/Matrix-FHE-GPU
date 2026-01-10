#!/usr/bin/env python3
import sys

def miller_rabin_deterministic(n: int) -> bool:
    """Deterministic Miller–Rabin valid for all n < 2^64."""
    if n < 2:
        return False
    small_primes = [2, 3, 5, 7, 11, 13, 17]
    for p in small_primes:
        if n == p:
            return True
        if n % p == 0:
            return False

    d = n - 1
    s = 0
    while (d & 1) == 0:
        d >>= 1
        s += 1

    bases = [2, 325, 9375, 28178, 450775, 9780504, 1795265022]
    for a in bases:
        if a % n == 0:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False
    return True

def find_primes(count=3):
    # 2^47 = 140737488355328
    limit = 1 << 47
    
    # 条件 A: NTT (q = 1 mod 2N) -> q = 1 mod 131072
    mod_ntt = 131072
    
    # 条件 B: Batching (q = 1 mod 17)
    mod_batch = 17
    
    # 综合步长: 131072 * 17 = 2228224
    step = mod_ntt * mod_batch
    
    print(f"Searching for primes < 2^47 ({limit})")
    print(f"Condition: q % {step} == 1")
    print(f"  -> q % {mod_ntt} == 1 (NTT Friendly for N=65536)")
    print(f"  -> q % {mod_batch} == 1 (Batching Friendly for p=17)")
    print("-" * 60)

    # 从最大值开始向下找
    # 对齐到 q = 1 mod step
    curr = limit - 1
    rem = (curr - 1) % step
    q = curr - rem # q 现在是满足同余条件的最大候选数

    found = []
    while len(found) < count and q > 2:
        if miller_rabin_deterministic(q):
            found.append(q)
            print(f"FOUND q[{len(found)-1}]: {q} (Hex: {hex(q)})")
        q -= step

    return found

if __name__ == "__main__":
    primes = find_primes(3)
    
    print("\n请将以下数组复制到你的 include/core/config.h 中:")
    print("static constexpr uint64_t RNS_MODULI[3] = {")
    for i, p in enumerate(primes):
        comma = "," if i < len(primes)-1 else ""
        print(f"    {p}ULL{comma} // 0x{p:X}")
    print("};")