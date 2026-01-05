
import random
import os
import hashlib
import hmac
from typing import Tuple, Iterable
import math

class AffineHash:
    """
    2-universal hashing for integer keys.
    h(x) = ((a*x + b) mod p) mod m
    Requirements: 0 <= x < p and p is prime.
    """
    def __init__(self, m: int, p: int):
        assert m >= 1 and p > 1
        self.m = m
        self.p = p
        self.a = random.randrange(1, p)   # a != 0
        self.b = random.randrange(0, p)

    def __call__(self, x: int) -> int:
        # Assume 0 <= x < p (or reduce x modulo p first).
        return ((self.a * x + self.b) % self.p) % self.m

# ---- Registration: hash and store (salt, hash, iterations) ----
def hash_password(password: str, iterations: int = 100_000) -> Tuple[bytes, bytes, int]:
    """
    Returns (salt, derived_key, iterations). Store all three.
    Uses PBKDF2-HMAC-SHA256 with the given iteration count.
    """
    salt = os.urandom(16)  # 16-byte random salt
    dk = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, iterations)
    return salt, dk, iterations

# ---- Login: verify a password attempt ----
def verify_password(password: str, salt: bytes, stored_dk: bytes, iterations: int) -> bool:
    """
    Recomputes PBKDF2 with the same salt + iterations and compares constant-time.
    """
    dk = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, iterations)
    return hmac.compare_digest(dk, stored_dk)


class BloomFilter:
    """
    Simple Bloom filter practices
    -m: number of bits in the filter
    - k: number of hash functions
    """
    def __init__(self, capacity: int, false_positive_rate: float=0.01):
        """
        Create a Bloom filter with given capacity and false positive rate.
        """
        if capacity <= 0:
            raise ValueError("Capacity must be > 0")
        if not (0 < false_positive_rate < 1):
            raise ValueError("False positive rate must be between 0 and 1")
        m = int(-capacity * math.log(false_positive_rate) / (math.log(2) ** 2))

        k = int((m / capacity) * math.log(2))

        self.m = m

        self.k = k
        self.bits = 0 # bitset stored in integer
        self.n = 0 # number of inserted elements

    def _hashes(self, item: bytes) -> Iterable[int]:
        """
        Generate k indices using double hashing.
        h1(x), h2(x), then h_i = (h1 + i * h2) mod m
        """
        h1 = int.from_bytes(hashlib.sha256(item).digest(), 'big')
        h2 = int.from_bytes(hashlib.blake2b(item, digest_size=16).digest(), 'big')
        for i in range(self.k):
            yield (h1 + i * h2) % self.m
    

    def add(self, item: str) -> None:
        b = item.encode('utf-8')
        changed = False
        for idx in self._hashes(b):
            mask = 1 << idx
            if self.bits & mask == 0:
                self.bits |= mask
                changed = True
        if changed:
            self.n += 1
    
    def __contains__(self, item: str) -> bool:
        b = item.encode('utf-8')
        for idx in self._hashes(b):
            if (self.bits >> idx) & 1 == 0:
                return False
        return True
    
    def fp_rate_estimate(self) -> float:
        """Estimate the current false positive rate."""
        return (1 - math.exp(-self.k * self.n / self.m)) ** self.k
    
    def info(self) -> dict:
        return {
            "m_bits": self.m,
            "k_hashes": self.k,
            "items": self.n,
            "fp_rate_estimate": round(self.fp_rate_estimate(), 6)
        }