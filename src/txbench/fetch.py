"""Fetch the pinned upstream release so a fresh checkout can rebuild.

Read-only access to the published artifact: the record's metadata is read, the
archive is downloaded with resume support, its md5 is checked against the one
the record publishes, and it is extracted into its own directory. Nothing here
writes to the upstream copy afterwards.
"""
from __future__ import annotations

import json
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from .errors import ReleaseError
from .hashutil import md5_file, sha256_file
from .logs import get_logger
from .paths import Layout

log = get_logger("fetch")

ZENODO_API = "https://zenodo.org/api/records/{record}"
ARCHIVE_SUFFIX = ".zip"


def record_metadata(record_id: str) -> dict[str, Any]:
    url = ZENODO_API.format(record=record_id)
    request = urllib.request.Request(url, headers={
        "Accept": "application/json",
        "User-Agent": "txbench-fetch/0.1 (research benchmark)"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def _pick_archive(meta: dict[str, Any]) -> dict[str, Any]:
    files = meta.get("files") or []
    archives = [f for f in files if str(f.get("key", "")).endswith(ARCHIVE_SUFFIX)]
    if not archives:
        raise ReleaseError("the record publishes no .zip archive")
    return max(archives, key=lambda f: int(f.get("size") or 0))


def download(url: str, target: Path, *, chunk: int = 1 << 20) -> int:
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={
        "User-Agent": "txbench-fetch/0.1 (research benchmark)"})
    written = 0
    tmp = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(request, timeout=120) as response, open(tmp, "wb") as f:
        while block := response.read(chunk):
            f.write(block)
            written += len(block)
    tmp.replace(target)
    return written


def fetch_release(layout: Layout, *, record_id: str = "23758032", keep_archive: bool = True,
                  expected_md5: str | None = None) -> dict[str, Any]:
    meta = record_metadata(record_id)
    archive = _pick_archive(meta)
    declared = str(archive.get("checksum") or "")
    declared_md5 = declared.split("md5:", 1)[1] if declared.startswith("md5:") else None
    if expected_md5 and declared_md5 and expected_md5 != declared_md5:
        raise ReleaseError("the record's published md5 does not match the expected one given "
                           "on the command line; refusing to download")
    target_dir = layout.upstream / "download"
    target = target_dir / str(archive["key"])
    out: dict[str, Any] = {
        "record_id": record_id,
        "doi": meta.get("doi"),
        "title": (meta.get("metadata") or {}).get("title"),
        "license": ((meta.get("metadata") or {}).get("license") or {}).get("id"),
        "archive": str(archive["key"]),
        "archive_bytes": int(archive.get("size") or 0),
        "published_md5": declared_md5,
        "archive_path": str(target),
    }
    if target.is_file() and target.stat().st_size == int(archive.get("size") or 0):
        log.info("archive already present: %s", target)
    else:
        log.info("downloading %s (%.1f MB)", archive["key"],
                 int(archive.get("size") or 0) / 1e6)
        download(archive["links"]["self"], target)
    actual_md5 = md5_file(target)
    out["actual_md5"] = actual_md5
    out["md5_verified"] = (declared_md5 is None) or (actual_md5 == declared_md5)
    if declared_md5 and actual_md5 != declared_md5:
        raise ReleaseError(f"md5 mismatch for {target.name}: the download does not match the "
                           "checksum the record publishes; delete it and retry")
    out["archive_sha256"] = sha256_file(target)

    destination = layout.upstream_release
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target) as zf:
        members = zf.namelist()
        roots = {name.split("/", 1)[0] for name in members}
        out["members"] = len(members)
        out["roots"] = sorted(roots)
        already = all((destination / root).exists() for root in roots)
        if already and not any((destination / root / ".incomplete").exists() for root in roots):
            log.info("already extracted under %s", destination)
            out["extracted"] = False
        else:
            log.info("extracting %d members into %s", len(members), destination)
            zf.extractall(destination)
            out["extracted"] = True
    out["release_dir"] = str(destination)
    if not keep_archive:
        target.unlink(missing_ok=True)
        out["archive_removed"] = True
    return out
