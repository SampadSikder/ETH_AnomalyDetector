"""Identity rules for the cohort.

A transaction is keyed by (chain id, lower-case 0x-prefixed 32-byte hash).
Nothing downstream may key on a hash alone.
"""
from __future__ import annotations

import re

from .errors import ValidationError

ETHEREUM_MAINNET = 1

_HASH_RE = re.compile(r"\A0x[0-9a-f]{64}\Z")
_ADDRESS_RE = re.compile(r"\A0x[0-9a-f]{40}\Z")
_HEX_RE = re.compile(r"\A0x([0-9a-f][0-9a-f])*\Z")

# Chain slug -> EIP-155 id, for cross-checking the upstream `chain` column
# against the upstream `chain_id` column rather than trusting either alone.
CHAIN_SLUGS = {
    "eth": 1, "ethereum": 1, "mainnet": 1,
    "bsc": 56, "arbitrum": 42161, "base": 8453,
    "optimism": 10, "polygon": 137, "avalanche": 43114,
}


def normalize_chain_id(value: object) -> int:
    """Return a positive EIP-155 chain id, or raise."""
    if isinstance(value, bool):
        raise ValidationError("chain_id must be an integer, not a bool")
    if isinstance(value, int):
        cid = value
    else:
        text = str(value).strip()
        if not text:
            raise ValidationError("chain_id is empty")
        try:
            cid = int(text, 16) if text.lower().startswith("0x") else int(text, 10)
        except ValueError:
            raise ValidationError(f"chain_id is not an integer: {text!r}") from None
    if cid <= 0:
        raise ValidationError(f"chain_id must be positive, got {cid}")
    return cid


def normalize_tx_hash(value: object) -> str:
    """Return a lower-case 0x-prefixed 32-byte hash, or raise.

    A bare 64-hex string is accepted and prefixed; anything else is rejected so
    that a truncated or mistyped hash can never silently become a cohort key.
    """
    if value is None:
        raise ValidationError("tx_hash is missing")
    text = str(value).strip().lower()
    if not text:
        raise ValidationError("tx_hash is empty")
    if not text.startswith("0x"):
        text = "0x" + text
    if not _HASH_RE.match(text):
        raise ValidationError(f"tx_hash is not a 32-byte hex hash: {_elide(text)}")
    return text


def normalize_address(value: object) -> str | None:
    """Return a lower-case 20-byte address, None for an absent address, or raise."""
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text or text in {"none", "null"}:
        return None
    if not text.startswith("0x"):
        text = "0x" + text
    if not _ADDRESS_RE.match(text):
        raise ValidationError(f"not a 20-byte address: {_elide(text)}")
    return text


def normalize_hex_bytes(value: object) -> str | None:
    """Return lower-case 0x-prefixed even-length hex, None if absent, or raise.

    `'0x'` (empty byte string) is a valid, meaningful value and is preserved as
    distinct from None (absent).
    """
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if not text.startswith("0x"):
        text = "0x" + text
    if not _HEX_RE.match(text):
        raise ValidationError(f"not an even-length hex byte string: {_elide(text)}")
    return text


def tx_key(chain_id: int, tx_hash: str) -> str:
    """The cohort's primary key. Stable, sortable, and never a bare hash."""
    return f"{normalize_chain_id(chain_id)}:{normalize_tx_hash(tx_hash)}"


def split_tx_key(key: str) -> tuple[int, str]:
    chain, _, tx = str(key).partition(":")
    if not tx:
        raise ValidationError(f"not a tx key: {_elide(str(key))}")
    return normalize_chain_id(chain), normalize_tx_hash(tx)


def chain_id_for_slug(slug: object) -> int | None:
    if slug is None:
        return None
    return CHAIN_SLUGS.get(str(slug).strip().lower())


def _elide(text: str, keep: int = 12) -> str:
    return text if len(text) <= keep * 2 else f"{text[:keep]}...{text[-4:]}"
