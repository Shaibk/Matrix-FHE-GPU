#!/usr/bin/env python3

def miller_rabin_deterministic(n: int) -> bool:
    """
    Deterministic Miller–Rabin primality test
    Valid for all n < 2^64
    """
    if n < 2:
        return False
    small_primes = [2, 3, 5, 7, 11, 13, 17]
    for p in small_primes:
        if n == p:
            return True
        if n % p == 0:
            return False

    # write n-1 = d * 2^s
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


def check_constraints(q: int, bits: int = 47, modA: int = 512, modB: int = 17) -> dict:
    return {
        "prime": miller_rabin_deterministic(q),
        "bit_length": q.bit_length(),
        f"<2^{bits}": q < (1 << bits),
        f"(q-1)%{modA}==0": ((q - 1) % modA == 0),
        f"(q-1)%{modB}==0": ((q - 1) % modB == 0),
        f"q%{modA}": q % modA,
        f"q%{modB}": q % modB,
    }


if __name__ == "__main__":
    primes = [
        140737433174017,
    140737361870849,
    140737355186177
    ]

    for q in primes:
        info = check_constraints(q, bits=47, modA=512, modB=17)
        print(f"q = {q}")
        print(f"  is prime: {info['prime']}")
        print(f"  bit_length: {info['bit_length']}  (<2^47: {info['<2^47']})")
        print(f"  (q-1)%512==0: {info['(q-1)%512==0']}, q%512={info['q%512']}")
        print(f"  (q-1)%17==0:  {info['(q-1)%17==0']},  q%17={info['q%17']}")
        print()
