
import random
import os
import hashlib
import hmac
from typing import Tuple

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

