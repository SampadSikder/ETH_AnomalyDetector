"""Checksums and deterministic digests.

Reproducibility rests on these: the same source files and config must produce
the same sorted hashes and labels, and a report must be able to prove it.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping

CHUNK = 1 << 20


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def md5_file(path: Path) -> str:
    """Only for comparing against a publisher's md5 (e.g. Zenodo's)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(obj: object) -> bytes:
    """Stable JSON bytes: sorted keys, no insignificant whitespace."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      default=str).encode("utf-8")


def digest_json(obj: object) -> str:
    return sha256_bytes(canonical_json(obj))


def cohort_digest(rows: Iterable[Mapping[str, object]], fields: tuple[str, ...]) -> str:
    """Digest of the cohort's identity: sorted (key, label, ...) tuples only.

    Deliberately narrow. It must not change when an unrelated column is added,
    and it must change when a hash or a label changes.
    """
    lines = sorted(canonical_json([row.get(f) for f in fields]) for row in rows)
    h = hashlib.sha256()
    h.update(canonical_json(list(fields)))
    for line in lines:
        h.update(line)
        h.update(b"\n")
    return h.hexdigest()
