"""API Key 加密存储（Fernet）。

主密钥来源优先级：
1. 环境变量 HUMANBOT_SECRET
2. data/.secret_key 文件（首次运行自动生成，权限 600）
"""
from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Optional

from ..settings import DATA_DIR

_KEY_FILE = DATA_DIR / ".secret_key"
_fernet = None


def _load_key() -> bytes:
    env = os.environ.get("HUMANBOT_SECRET", "").strip()
    if env:
        return _normalize(env)
    if _KEY_FILE.exists():
        raw = _KEY_FILE.read_text(encoding="utf-8").strip()
        if raw:
            return _normalize(raw)
    key = base64.urlsafe_b64encode(os.urandom(32)).decode()
    _KEY_FILE.write_text(key, encoding="utf-8")
    try:
        os.chmod(_KEY_FILE, 0o600)
    except OSError:
        pass
    return _normalize(key)


def _normalize(raw: str) -> bytes:
    b = raw.encode()
    try:
        base64.urlsafe_b64decode(b)
        return b
    except Exception:
        # 任意字符串 → 派生 32 字节
        import hashlib
        return base64.urlsafe_b64encode(hashlib.sha256(b).digest())


def _get_fernet():
    global _fernet
    if _fernet is None:
        try:
            from cryptography.fernet import Fernet
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "缺少 cryptography 依赖，请先 pip install cryptography"
            ) from exc
        _fernet = Fernet(_load_key())
    return _fernet


def encrypt(plain: str) -> str:
    """加密，返回 base64 密文。"""
    if not plain:
        return ""
    try:
        return _get_fernet().encrypt(plain.encode()).decode()
    except Exception:
        # 加密不可用时退化为明文（仍加前缀标识，便于排查）
        return "plain:" + base64.urlsafe_b64encode(plain.encode()).decode()


def decrypt(cipher: str) -> str:
    """解密。"""
    if not cipher:
        return ""
    if cipher.startswith("plain:"):
        try:
            return base64.urlsafe_b64decode(cipher[6:]).decode()
        except Exception:
            return ""
    try:
        return _get_fernet().decrypt(cipher.encode()).decode()
    except Exception:
        return ""


def gen_password(length: int = 20) -> str:
    """生成强密码（无歧义字符）。"""
    import secrets
    import string

    alphabet = string.ascii_letters + string.digits + "!@#$%^&*-_=+"
    # 排除易混字符
    for ch in "Il1O0o":
        alphabet = alphabet.replace(ch, "")
    while True:
        pwd = "".join(secrets.choice(alphabet) for _ in range(length))
        if (any(c.islower() for c in pwd) and any(c.isupper() for c in pwd)
                and any(c.isdigit() for c in pwd)
                and any(c in "!@#$%^&*-_=+" for c in pwd)):
            return pwd


def gen_token(length: int = 48) -> str:
    import secrets
    return secrets.token_urlsafe(length)


def hash_password(pwd: str, salt: Optional[str] = None) -> str:
    """PBKDF2 口令散列，格式 salt$hash。"""
    import hashlib
    import hmac
    import secrets as _s

    salt = salt or _s.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", pwd.encode(), salt.encode(), 200_000)
    return f"{salt}${dk.hex()}"


def verify_password(pwd: str, stored: str) -> bool:
    import hashlib
    import hmac

    if not stored or "$" not in stored:
        return hmac.compare_digest(pwd, stored)
    salt, _ = stored.split("$", 1)
    return hmac.compare_digest(hash_password(pwd, salt), stored)
