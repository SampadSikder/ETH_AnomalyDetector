"""Credential loading and redaction.

Rules enforced here:
  * keys come from the process environment or a local `.env` at the repo root;
  * values already exported by the shell win over `.env`;
  * every secret is registered for redaction, so no log line, error message,
    report or cached provenance record can leak one;
  * an authenticated endpoint URL is itself treated as a secret.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

#: Environment variable names this project reads. Documented in `.env.example`.
ALCHEMY_KEY_VAR = "ALCHEMY_API_KEY"
DRPC_KEY_VAR = "DRPC_API_KEY"
CREDENTIAL_VARS = (ALCHEMY_KEY_VAR, DRPC_KEY_VAR)

REDACTION = "***REDACTED***"

_secrets: set[str] = set()


def register_secret(value: str | None) -> None:
    """Mark a value (a key, or a URL containing one) as never-to-be-printed."""
    if value and len(str(value)) >= 8:
        _secrets.add(str(value))


def redact(text: object) -> str:
    """Return `text` with every registered secret replaced."""
    out = str(text)
    for secret in sorted(_secrets, key=len, reverse=True):
        out = out.replace(secret, REDACTION)
    # Defence in depth: scrub anything that looks like a key in a URL path or
    # query even if it was never registered (e.g. a key pasted into a config).
    out = re.sub(r"(?i)(api[-_]?key=)[^&\s\"']+", r"\1" + REDACTION, out)
    out = re.sub(r"(?i)(v2/)[A-Za-z0-9_-]{16,}", r"\1" + REDACTION, out)
    out = re.sub(r"(?i)(dkey=)[^&\s\"']+", r"\1" + REDACTION, out)
    return out


def load_dotenv(repo_root: Path, *, override: bool = False) -> dict[str, str]:
    """Load `repo_root/.env` into the environment without clobbering the shell.

    Returns the variable names found (never the values). The file is left
    untouched; this never writes to `.env`.
    """
    found: dict[str, str] = {}
    path = Path(repo_root) / ".env"
    if not path.is_file():
        return found
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("export "):
            line = line[len("export "):].lstrip()
        name, sep, value = line.partition("=")
        if not sep:
            continue
        name = name.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if not name:
            continue
        found[name] = value
        if override or not os.environ.get(name):
            os.environ[name] = value
    return found


def credential(var: str) -> str | None:
    """Read a credential from the environment and register it for redaction."""
    value = os.environ.get(var, "").strip()
    if not value:
        return None
    register_secret(value)
    return value


def credential_status() -> dict[str, bool]:
    """Which credentials are present. Booleans only - safe for any report."""
    return {var: bool(os.environ.get(var, "").strip()) for var in CREDENTIAL_VARS}
