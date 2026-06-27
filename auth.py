"""
auth.py
───────
Authentication & authorization for the BCT RAG web application.

Deliberately dependency-free — Python standard library only — so the single
`python api.py` deployment stays trivial to reproduce for the defense (no JWT or
bcrypt wheels to compile on Windows).

  - Passwords : PBKDF2-HMAC-SHA256, 200k iterations, per-user random salt.
  - Tokens    : compact "payload.signature" tokens, HMAC-SHA256 signed, with an
                expiry claim. Self-contained and stateless (no server sessions).
  - Roles     : 'admin' (manage accounts + analytics) and 'bank' (query only).
"""

import base64
import hashlib
import hmac
import json
import logging
import os
import time

from fastapi import Depends, Header, HTTPException

import config
import database as db

log = logging.getLogger(__name__)

_PBKDF2_ITERATIONS = 200_000
TOKEN_TTL_SECONDS = 8 * 60 * 60  # 8 hours — comfortably covers a work session

DEFAULT_ADMIN_USERNAME = "admin"
DEFAULT_ADMIN_PASSWORD = "admin"


# ── Signing secret ────────────────────────────────────────────────────────────
def _get_secret() -> bytes:
    """Resolve the HMAC signing secret.

    Order of preference:
      1. BCT_SECRET_KEY env var (best for production).
      2. A generated secret persisted to a local file, so tokens survive server
         restarts during the demo and users are not forced to re-login.
    """
    env_secret = os.getenv("BCT_SECRET_KEY", "")
    if env_secret:
        return env_secret.encode("utf-8")

    secret_file = config.BASE_DIR / ".auth_secret"
    if secret_file.exists():
        return secret_file.read_bytes()

    generated = os.urandom(32)
    try:
        secret_file.write_bytes(generated)
    except Exception as e:  # pragma: no cover - filesystem edge case
        log.warning(f"Could not persist auth secret ({e}); using in-memory secret.")
    return generated


_SECRET = _get_secret()


# ── Password hashing ──────────────────────────────────────────────────────────
def hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    """Return (hex_hash, hex_salt). Generates a fresh salt when none is given."""
    if salt is None:
        salt = os.urandom(16).hex()
    dk = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt), _PBKDF2_ITERATIONS
    )
    return dk.hex(), salt


def verify_password(password: str, password_hash: str, salt: str) -> bool:
    candidate, _ = hash_password(password, salt)
    return hmac.compare_digest(candidate, password_hash)


# ── Tokens ────────────────────────────────────────────────────────────────────
def _b64e(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")


def _b64d(data: str) -> bytes:
    return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4))


def create_token(user_id: int, username: str, role: str) -> str:
    payload = {
        "uid": user_id,
        "username": username,
        "role": role,
        "exp": int(time.time()) + TOKEN_TTL_SECONDS,
    }
    payload_b64 = _b64e(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    sig = hmac.new(_SECRET, payload_b64.encode("utf-8"), hashlib.sha256).digest()
    return f"{payload_b64}.{_b64e(sig)}"


def decode_token(token: str) -> dict | None:
    """Return the payload if the signature is valid and not expired, else None."""
    try:
        payload_b64, sig_b64 = token.split(".")
        expected = hmac.new(_SECRET, payload_b64.encode("utf-8"), hashlib.sha256).digest()
        if not hmac.compare_digest(expected, _b64d(sig_b64)):
            return None
        payload = json.loads(_b64d(payload_b64))
        if int(payload.get("exp", 0)) < int(time.time()):
            return None
        return payload
    except Exception:
        return None


# ── High-level operations ─────────────────────────────────────────────────────
def authenticate(username: str, password: str) -> dict | None:
    """Validate credentials and return {token, user} on success, else None."""
    row = db.get_user_by_username(username)
    if not row:
        return None
    if row["status"] != "active":
        # Surface suspension distinctly so the API can return a clear message.
        return {"suspended": True}
    if not verify_password(password, row["password_hash"], row["salt"]):
        return None
    token = create_token(row["id"], row["username"], row["role"])
    return {"token": token, "user": db.user_to_public(row)}


def ensure_default_admin() -> None:
    """Create the undeletable default admin/admin account on first startup."""
    if db.get_user_by_username(DEFAULT_ADMIN_USERNAME):
        return
    pwd_hash, salt = hash_password(DEFAULT_ADMIN_PASSWORD)
    db.create_user(
        username=DEFAULT_ADMIN_USERNAME,
        password_hash=pwd_hash,
        salt=salt,
        role="admin",
        bank_name="Banque Centrale de Tunisie",
        status="active",
    )
    log.info("Default admin account created (admin / admin). Change the password after first login.")


def is_default_admin(row) -> bool:
    """The default admin is protected from deletion / suspension."""
    return row is not None and row["username"] == DEFAULT_ADMIN_USERNAME


# ── FastAPI dependencies ──────────────────────────────────────────────────────
def get_current_user(authorization: str = Header(default="")) -> dict:
    """Resolve and validate the bearer token into the live user record."""
    token = authorization[7:] if authorization.startswith("Bearer ") else ""
    payload = decode_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired session")

    row = db.get_user_by_id(payload["uid"])
    if not row or row["status"] != "active":
        raise HTTPException(status_code=401, detail="Account not found or suspended")

    return db.user_to_public(row)


def require_admin(user: dict = Depends(get_current_user)) -> dict:
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Administrator privileges required")
    return user
