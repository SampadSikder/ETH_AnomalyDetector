"""Discovery, verification and provenance of an upstream TxBench release.

The release is treated as read-only evidence. Its files are never rewritten,
never mixed with model features, and every file this project reads is recorded
with its checksum so a second researcher can prove they read the same bytes.

A *fixture* release is a tiny, hand-written stand-in used so that the importer,
the extractors and the tests can run with no real labels present. A fixture is
marked by a `FIXTURE` marker file in its root, and the mark travels with every
row, report and metric derived from it.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .errors import ReleaseError
from .hashutil import sha256_file
from .logs import get_logger

log = get_logger("release")

FIXTURE_MARKER = "FIXTURE"

#: Canonical relative paths inside a release, by role. Each is resolved
#: against the release root; a missing optional file is reported, not fatal.
ROLES: dict[str, tuple[str, bool]] = {
    # role: (relative path, required)
    "combined": ("step4_benign/output/benchmark/combined.csv", True),
    "traces_dir": ("step4_benign/output/benchmark/traces", False),
    "benign_hard": ("step4_benign/output/benchmark/benign_hard.csv", False),
    "attacks": ("step4_benign/output/benchmark/attacks.csv", False),
    "accepted_attacks": ("step3_oracle/output/attack_txs_accepted.csv", False),
    "oracle": ("step3_oracle/output/oracle.csv", False),
    "splits_report": ("step4_benign/output/benchmark/splits_report.json", False),
    "traces_status_step2": ("step2_traces/output/traces_status.json", False),
    "traces_status_step4": ("step4_benign/output/traces_status.json", False),
    "evaluator": ("step4_benign/evaluate.py", False),
    "data_dictionary": ("step4_benign/docs/data_dictionary.md", False),
    "methodology": ("step4_benign/docs/methodology.md", False),
    "documentation": ("DOCUMENTATION.md", False),
    "readme": ("README.md", False),
    "checksums": ("SHA256SUMS", False),
    "code_license": ("LICENSE", False),
    "data_license": ("DATASET_LICENSE.md", False),
    "validator": ("validate.py", False),
}


@dataclass
class Release:
    root: Path
    release_id: str
    is_fixture: bool
    doi: str | None = None
    files: dict[str, Path] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    checksums: dict[str, str] = field(default_factory=dict)      # role -> sha256 of the file we read
    declared_checksums: dict[str, str] = field(default_factory=dict)  # role -> sha256 from SHA256SUMS
    checksum_status: dict[str, str] = field(default_factory=dict)      # role -> verified|undeclared|MISMATCH
    license_notice: dict[str, object] = field(default_factory=dict)

    def path(self, role: str) -> Path:
        if role not in self.files:
            raise ReleaseError(
                f"release {self.release_id!r} has no {role!r} file "
                f"(expected {ROLES.get(role, ('?',))[0]} under {self.root})")
        return self.files[role]

    def has(self, role: str) -> bool:
        return role in self.files

    def provenance(self) -> dict[str, object]:
        return {
            "release_id": self.release_id,
            "release_root": str(self.root),
            "doi": self.doi,
            "is_fixture": self.is_fixture,
            "files": {role: str(p.relative_to(self.root)) for role, p in sorted(self.files.items())},
            "missing_roles": sorted(self.missing),
            "sha256": dict(sorted(self.checksums.items())),
            "sha256_declared": dict(sorted(self.declared_checksums.items())),
            "checksum_status": dict(sorted(self.checksum_status.items())),
            "license_notice": self.license_notice,
        }


def _find_root(given: Path) -> Path:
    """Accept the release root, its parent, or a directory holding one copy."""
    given = Path(given)
    if not given.exists():
        raise ReleaseError(f"release directory not found: {given}")
    probe = Path(ROLES["combined"][0])
    if (given / probe).is_file() or (given / FIXTURE_MARKER).is_file():
        return given
    nested = sorted(p for p in given.iterdir() if p.is_dir() and (p / probe).is_file())
    if len(nested) == 1:
        return nested[0]
    if len(nested) > 1:
        raise ReleaseError(f"{given} holds {len(nested)} releases; name one explicitly: "
                           f"{[p.name for p in nested]}")
    raise ReleaseError(
        f"{given} does not look like a TxBench release: no {probe} and no {FIXTURE_MARKER} marker")


def _release_id(root: Path, override: str | None) -> str:
    if override:
        return override
    readme = root / "README.md"
    if readme.is_file():
        head = readme.read_text(encoding="utf-8", errors="replace")[:4000]
        m = re.search(r"version\s+(\d+\.\d+(?:\.\d+)?)", head, re.I)
        if m:
            return f"TxBench-v{m.group(1)}"
    name = root.name
    return name if name else "TxBench-unknown"


def _read_declared_checksums(root: Path) -> dict[str, str]:
    path = root / ROLES["checksums"][0]
    if not path.is_file():
        return {}
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and re.fullmatch(r"[0-9a-f]{64}", parts[0]):
            rel = parts[1].lstrip("*").strip()
            out[rel.removeprefix("./")] = parts[0]
    return out


def _license_notice(root: Path) -> dict[str, object]:
    """What the release says about reuse and redistribution.

    Recorded verbatim-ish (first lines) so packaging decisions are traceable;
    the full files stay in the release.
    """
    notice: dict[str, object] = {"files": {}, "redistribution_reviewed": False}
    for role in ("code_license", "data_license"):
        rel = ROLES[role][0]
        p = root / rel
        if p.is_file():
            text = p.read_text(encoding="utf-8", errors="replace")
            head = [ln.strip() for ln in text.splitlines() if ln.strip()][:12]
            notice["files"][rel] = {"sha256": sha256_file(p), "head": head}
    notice["redistribution_reviewed"] = bool(notice["files"])
    return notice


def load_release(release_dir: Path, *, release_id: str | None = None,
                 doi: str | None = None, verify_checksums: bool = True) -> Release:
    """Resolve a release directory into verified, role-keyed paths."""
    root = _find_root(Path(release_dir))
    is_fixture = (root / FIXTURE_MARKER).is_file()
    rel = Release(root=root, release_id=_release_id(root, release_id), is_fixture=is_fixture, doi=doi)
    if is_fixture:
        log.warning("release %s is FIXTURE data: no real upstream labels", rel.release_id)

    declared = _read_declared_checksums(root)
    for role, (relpath, required) in ROLES.items():
        target = root / relpath
        exists = target.is_dir() if role.endswith("_dir") else target.is_file()
        if not exists:
            rel.missing.append(role)
            if required:
                raise ReleaseError(f"release {rel.release_id!r} is missing required {role} "
                                   f"({relpath}) under {root}")
            continue
        rel.files[role] = target
        if target.is_file() and verify_checksums:
            actual = sha256_file(target)
            rel.checksums[role] = actual
            want = declared.get(relpath)
            if want:
                rel.declared_checksums[role] = want
                rel.checksum_status[role] = "verified" if want == actual else "MISMATCH"
            else:
                rel.checksum_status[role] = "undeclared"

    bad = [r for r, s in rel.checksum_status.items() if s == "MISMATCH"]
    if bad:
        raise ReleaseError(f"checksum mismatch against {ROLES['checksums'][0]} for: {sorted(bad)}. "
                           "The release copy does not match what it declares; do not import it.")
    rel.license_notice = _license_notice(root)
    log.info("release %s at %s (fixture=%s); %d files resolved, %d roles missing",
             rel.release_id, root, is_fixture, len(rel.files), len(rel.missing))
    return rel


def write_provenance(rel: Release, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rel.provenance(), indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    return out_path
