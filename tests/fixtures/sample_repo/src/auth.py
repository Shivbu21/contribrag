"""Token-based authentication helpers for SampleLib."""
import hashlib
import hmac

SECRET = "sample-secret"


def hash_password(password: str) -> str:
    """Hash a password with SHA-256 and a static salt prefix."""
    return hashlib.sha256(f"salt:{password}".encode()).hexdigest()


def verify_token(token: str, expected: str) -> bool:
    """Verify a token against the expected value in constant time."""
    return hmac.compare_digest(token, expected)
