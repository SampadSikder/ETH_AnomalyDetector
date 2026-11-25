"""The on-disk layout. One place, so a report can name every artefact."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path:
    """Walk up for a marker file; fall back to the current directory."""
    here = Path(start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").is_file() or (candidate / ".git").exists():
            return candidate
    return here


@dataclass(frozen=True)
class Layout:
    root: Path

    # --- upstream (never modified, never mixed with model features) ---
    @property
    def upstream(self) -> Path: return self.root / "data" / "upstream"
    @property
    def upstream_release(self) -> Path: return self.upstream / "txbench"

    # --- derived ---
    @property
    def data(self) -> Path: return self.root / "data"
    @property
    def manifest_dir(self) -> Path: return self.data / "manifest"
    @property
    def cache_dir(self) -> Path: return self.data / "cache"
    @property
    def cache_index(self) -> Path: return self.cache_dir / "index.sqlite"
    @property
    def features_dir(self) -> Path: return self.data / "features"
    @property
    def splits_dir(self) -> Path: return self.data / "splits"
    @property
    def adapters_dir(self) -> Path: return self.data / "adapters"
    @property
    def eval_dir(self) -> Path: return self.data / "evaluation"
    @property
    def reports_dir(self) -> Path: return self.root / "reports"
    @property
    def fixtures_dir(self) -> Path: return self.root / "fixtures"

    # --- named artefacts ---
    @property
    def manifest(self) -> Path: return self.manifest_dir / "manifest.parquet"
    @property
    def manifest_meta(self) -> Path: return self.manifest_dir / "manifest_meta.json"
    @property
    def schema_mapping(self) -> Path: return self.manifest_dir / "schema_mapping.json"
    @property
    def quarantine(self) -> Path: return self.manifest_dir / "quarantine.parquet"
    @property
    def cohort_report(self) -> Path: return self.reports_dir / "cohort.md"

    def ensure(self) -> "Layout":
        for p in (self.upstream, self.manifest_dir, self.cache_dir, self.features_dir,
                  self.splits_dir, self.adapters_dir, self.eval_dir, self.reports_dir):
            p.mkdir(parents=True, exist_ok=True)
        return self


def layout(root: Path | None = None) -> Layout:
    return Layout(find_repo_root(root))
