"""Configuration: code defaults, overridable by a TOML file, logged in full.

Nothing secret lives here. A provider holds the *name* of the environment
variable that carries its key, never the key.
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from .errors import ConfigError
from .ids import ETHEREUM_MAINNET

# Methods this project may ever ask a provider for.
M_TX = "eth_getTransactionByHash"
M_RECEIPT = "eth_getTransactionReceipt"
M_BLOCK_BY_NUMBER = "eth_getBlockByNumber"
M_BLOCK_BY_HASH = "eth_getBlockByHash"
M_CHAIN_ID = "eth_chainId"
M_DEBUG_TRACE = "debug_traceTransaction"
M_TRACE_TX = "trace_transaction"

BASE_METHODS = (M_TX, M_RECEIPT, M_BLOCK_BY_NUMBER, M_BLOCK_BY_HASH)
TRACE_METHODS = (M_DEBUG_TRACE, M_TRACE_TX)


@dataclass
class ProviderConfig:
    name: str
    enabled: bool = True
    key_var: str = ""
    #: `{key}` is substituted at call time and never logged or stored.
    url_template: str = ""
    #: Unauthenticated endpoint, used only for a capability probe if given.
    public_url: str = ""
    concurrency: int = 2
    #: Provider-side throughput ceiling, in cost units per second.
    rate_limit_units_per_s: float = 300.0
    #: Name of the unit the provider bills in ("CU", "request", ...).
    unit: str = "CU"
    #: Cap for a single run. A run that would exceed it stops and says so.
    run_unit_budget: int = 200_000
    #: The plan's documented allowance, for the cost report only.
    plan_unit_allowance: int | None = None
    #: Per-method unit cost. Defaults are the published list prices for the
    #: free plan; override per account, because plans differ.
    method_units: dict[str, int] = field(default_factory=dict)
    #: Methods known to be unavailable on this plan. Disabled for the run
    #: rather than retried: a 'not supported' answer is not a failure to retry.
    disabled_methods: list[str] = field(default_factory=list)
    #: Methods this provider may serve as a fallback for.
    fallback_methods: list[str] = field(default_factory=lambda: list(BASE_METHODS))
    units_source: str = "configured default; verify against the account's plan"

    def unit_cost(self, method: str) -> int:
        return int(self.method_units.get(method, self.method_units.get("_default", 20)))

    def supports(self, method: str) -> bool:
        return self.enabled and method not in self.disabled_methods


def _alchemy_default() -> ProviderConfig:
    return ProviderConfig(
        name="alchemy",
        key_var="ALCHEMY_API_KEY",
        url_template="https://eth-mainnet.g.alchemy.com/v2/{key}",
        concurrency=4,
        rate_limit_units_per_s=300.0,          # Free: 300 CU/s (plan docs)
        unit="CU",
        run_unit_budget=400_000,
        plan_unit_allowance=30_000_000,        # Free: 30M CU/month (plan docs)
        method_units={M_TX: 20, M_RECEIPT: 20, M_BLOCK_BY_NUMBER: 20,
                      M_BLOCK_BY_HASH: 20, M_CHAIN_ID: 0, "_default": 20},
        # Debug and Trace APIs are excluded from Alchemy's Free plan.
        disabled_methods=[M_DEBUG_TRACE, M_TRACE_TX],
        units_source="Alchemy pricing / compute-unit-cost docs for the Free plan",
    )


def _drpc_default() -> ProviderConfig:
    return ProviderConfig(
        name="drpc",
        key_var="DRPC_API_KEY",
        url_template="https://lb.drpc.org/ogrpc?network=ethereum&dkey={key}",
        enabled=True,
        concurrency=2,
        rate_limit_units_per_s=30.0,           # conservative; PAYG lists 100 RPS
        unit="request",
        run_unit_budget=20_000,
        plan_unit_allowance=10_000_000,        # Free: ~10M equivalent requests
        method_units={"_default": 1, M_CHAIN_ID: 0},
        # dRPC's Ethereum debug/trace methods are paid, so they stay off.
        disabled_methods=[M_DEBUG_TRACE, M_TRACE_TX],
        fallback_methods=list(BASE_METHODS),
        units_source="dRPC plan docs (Free tier, multichain equivalent requests)",
    )


@dataclass
class CollectConfig:
    pilot_size: int = 20
    seed: int = 42
    #: Hard caps for a single run. Reaching one stops the run cleanly.
    max_calls: int = 20_000
    max_retries: int = 4
    retry_base_delay_s: float = 0.5
    retry_max_delay_s: float = 30.0
    request_timeout_s: float = 30.0
    #: Per-payload ceiling for a stored trace. Larger payloads are truncated
    #: and recorded as truncated rather than silently clipped.
    max_stored_payload_bytes: int = 32 * 1024 * 1024
    #: Cross-check the upstream envelope/receipt against an RPC read even when
    #: the upstream record already carries the field.
    verify_upstream: bool = False


@dataclass
class FeatureConfig:
    #: Minimum training-set document frequency for a token to enter the
    #: vocabulary. 1 keeps every training token.
    min_train_frequency: int = 1
    #: Split whose train rows fit the vocabulary and scalers.
    fit_split: str = "temporal"


@dataclass
class SplitConfig:
    primary: str = "temporal"
    secondary: tuple[str, ...] = ("protocol",)
    metadata_only: tuple[str, ...] = ("cross_chain",)
    #: Share of training rows held out as validation, taken from the latest
    #: training blocks so that validation stays in the past relative to test.
    validation_fraction: float = 0.2
    validation_seed: int = 42


@dataclass
class BigQueryConfig:
    """The warehouse backfill. Nothing here is a credential.

    `total_byte_budget` is the ceiling for *all* executed queries together. It
    is deliberately `None` by default: with no budget configured the collector
    dry-runs, prints the command that would spend bytes, and stops.
    """
    table: str = "bigquery-public-data.goog_blockchain_ethereum_mainnet_us.traces"
    #: Total bytes this project may bill across runs. None = nothing executes.
    total_byte_budget: int | None = None
    #: Environment variable naming the billing project (never a key).
    project_var: str = "GOOGLE_CLOUD_PROJECT"


@dataclass
class Config:
    chain_id: int = ETHEREUM_MAINNET
    release_dir: Path = Path("data/upstream/txbench")
    release_doi: str | None = None
    collect: CollectConfig = field(default_factory=CollectConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    splits: SplitConfig = field(default_factory=SplitConfig)
    bigquery: BigQueryConfig = field(default_factory=BigQueryConfig)
    providers: dict[str, ProviderConfig] = field(
        default_factory=lambda: {"alchemy": _alchemy_default(), "drpc": _drpc_default()})
    source_file: Path | None = None

    def provider(self, name: str) -> ProviderConfig:
        try:
            return self.providers[name]
        except KeyError:
            raise ConfigError(f"unknown provider {name!r}; configured: "
                              f"{sorted(self.providers)}") from None

    def effective_settings(self) -> dict[str, Any]:
        """Everything that affects a run, for the log and the cost report."""
        out = {
            "chain_id": self.chain_id,
            "release_dir": str(self.release_dir),
            "release_doi": self.release_doi,
            "config_file": str(self.source_file) if self.source_file else None,
            "collect": asdict(self.collect),
            "features": asdict(self.features),
            "splits": {**asdict(self.splits)},
            "bigquery": asdict(self.bigquery),
            "providers": {},
        }
        for name, p in self.providers.items():
            d = asdict(p)
            d.pop("url_template", None)       # may contain a key placeholder
            d.pop("public_url", None)
            out["providers"][name] = d
        return out


_SECTION_TYPES = {"collect": CollectConfig, "features": FeatureConfig, "splits": SplitConfig,
                  "bigquery": BigQueryConfig}


def load_config(path: Path | None = None, root: Path | None = None) -> Config:
    """Load defaults, then overlay a TOML file if one is given or discovered."""
    cfg = Config()
    candidate = Path(path) if path else None
    if candidate is None and root is not None:
        discovered = Path(root) / "config" / "benchmark.toml"
        candidate = discovered if discovered.is_file() else None
    if candidate is None:
        return cfg
    if not candidate.is_file():
        raise ConfigError(f"config file not found: {candidate}")
    with open(candidate, "rb") as f:
        raw = tomllib.load(f)
    cfg.source_file = candidate

    cohort = raw.get("cohort", {})
    if "chain_id" in cohort:
        cfg.chain_id = int(cohort["chain_id"])
    if "release_dir" in cohort:
        cfg.release_dir = Path(cohort["release_dir"])
    if "release_doi" in cohort:
        cfg.release_doi = str(cohort["release_doi"]) or None

    for section, kind in _SECTION_TYPES.items():
        values = raw.get(section) or {}
        if not isinstance(values, dict):
            raise ConfigError(f"[{section}] must be a table")
        current = getattr(cfg, section)
        for key, value in values.items():
            if not hasattr(current, key):
                raise ConfigError(f"unknown option [{section}].{key}")
            existing = getattr(current, key)
            if isinstance(existing, tuple):
                value = tuple(value)
            setattr(current, key, value)

    for name, values in (raw.get("providers") or {}).items():
        base = cfg.providers.get(name) or ProviderConfig(name=name)
        for key, value in (values or {}).items():
            if key == "method_units":
                base.method_units = {**base.method_units, **{k: int(v) for k, v in value.items()}}
            elif hasattr(base, key):
                setattr(base, key, value)
            else:
                raise ConfigError(f"unknown option [providers.{name}].{key}")
        cfg.providers[name] = base
    return cfg
