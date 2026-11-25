"""Shared fixtures. Everything here runs offline, with no credentials."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
FIXTURE_RELEASE = REPO / "fixtures" / "txbench_release"

# Hash helpers mirroring the fixture generator.
def H(n: int) -> str:
    return "0x" + f"{n:02x}" * 32


def ADDR(n: int) -> str:
    return "0x" + f"{n:02x}" * 20


@pytest.fixture(autouse=True)
def no_credentials(monkeypatch):
    """Every test runs as if no key were configured."""
    for var in ("ALCHEMY_API_KEY", "DRPC_API_KEY"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def release():
    from txbench.release import load_release
    return load_release(FIXTURE_RELEASE)


@pytest.fixture
def imported(release):
    from txbench.importer import import_labels
    return import_labels(release, chain_id=1)


@pytest.fixture
def manifest(imported):
    return imported.manifest


@pytest.fixture
def traces_dir(release):
    return release.path("traces_dir")


@pytest.fixture
def trace_status(release):
    from txbench.upstream_status import load_trace_status
    return load_trace_status(release)


@pytest.fixture
def records(manifest, traces_dir, trace_status):
    from txbench.assemble import records as build
    return dict(build(manifest, traces_dir=traces_dir, store=None, trace_status=trace_status))


@pytest.fixture
def tx_of(records):
    def get(n: int):
        return records[f"1:{H(n)}"]
    return get
