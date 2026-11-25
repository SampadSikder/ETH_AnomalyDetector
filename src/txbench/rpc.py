"""JSON-RPC providers with budgets, rate limits, bounded retries and no leaks.

What this module guarantees:
  * an authenticated endpoint URL is built at call time, registered as a
    secret, and never logged, stored or reported;
  * every call is charged against a per-provider unit budget and a global call
    cap, and reaching a cap raises `BudgetExceeded` instead of quietly
    continuing;
  * batch members are counted individually;
  * a throttled response honours `Retry-After`; retries are bounded and
    jittered;
  * "method not supported" is a permanent answer, not something to retry: the
    method is disabled for the rest of the run;
  * no paid plan is ever activated: methods a plan excludes are configured off.
"""
from __future__ import annotations

import json
import random
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Iterable

from .config import ProviderConfig
from .errors import BudgetExceeded, ProviderError
from .logs import get_logger
from .credentials import credential, redact, register_secret

log = get_logger("rpc")

#: JSON-RPC error codes that mean "this endpoint will never serve this method".
UNSUPPORTED_CODES = {-32601, -32004}
UNSUPPORTED_MARKERS = ("method not found", "not supported", "unsupported method",
                       "not available on", "requires a paid", "upgrade your plan",
                       "method does not exist")
THROTTLE_MARKERS = ("rate limit", "too many requests", "capacity", "throughput", "429")


@dataclass
class Budget:
    """Caps for one run. Nothing here is per-month accounting; it is a fuse."""
    max_calls: int
    per_provider_units: dict[str, int] = field(default_factory=dict)
    calls: int = 0
    units: dict[str, int] = field(default_factory=dict)
    retries: int = 0
    stored_bytes: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def charge(self, provider: str, units: int, *, calls: int = 1) -> None:
        with self._lock:
            if self.calls + calls > self.max_calls:
                raise BudgetExceeded(
                    f"call cap reached: {self.calls} calls made, cap {self.max_calls}")
            limit = self.per_provider_units.get(provider)
            spent = self.units.get(provider, 0)
            if limit is not None and spent + units > limit:
                raise BudgetExceeded(
                    f"{provider}: unit budget reached ({spent} spent + {units} requested "
                    f"> {limit}); raise [providers.{provider}].run_unit_budget to continue")
            self.calls += calls
            self.units[provider] = spent + units

    def would_exceed(self, provider: str, units: int) -> bool:
        with self._lock:
            limit = self.per_provider_units.get(provider)
            return (self.calls + 1 > self.max_calls or
                    (limit is not None and self.units.get(provider, 0) + units > limit))

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {"calls": self.calls, "max_calls": self.max_calls, "retries": self.retries,
                    "units_by_provider": dict(sorted(self.units.items())),
                    "unit_budget_by_provider": dict(sorted(self.per_provider_units.items()))}


class RateLimiter:
    """Token bucket in provider units per second, shared across threads."""

    def __init__(self, units_per_second: float):
        self.rate = max(float(units_per_second), 0.001)
        self._tokens = self.rate
        self._updated = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, units: int) -> float:
        """Block until `units` are available. Returns the seconds waited."""
        need = max(float(units), 0.0)
        waited = 0.0
        while True:
            with self._lock:
                now = time.monotonic()
                self._tokens = min(self.rate, self._tokens + (now - self._updated) * self.rate)
                self._updated = now
                if need <= self._tokens or need == 0.0:
                    self._tokens -= need
                    return waited
                deficit = need - self._tokens
                delay = min(deficit / self.rate, 5.0)
            time.sleep(delay)
            waited += delay


@dataclass
class RpcResult:
    payload: Any
    raw_bytes: bytes
    units: int
    provider: str
    attempts: int
    waited_s: float


class Provider:
    """One JSON-RPC endpoint."""

    def __init__(self, cfg: ProviderConfig, *, timeout_s: float = 30.0,
                 max_retries: int = 4, base_delay_s: float = 0.5, max_delay_s: float = 30.0,
                 budget: Budget | None = None, opener=None):
        self.cfg = cfg
        self.name = cfg.name
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.base_delay_s = base_delay_s
        self.max_delay_s = max_delay_s
        self.budget = budget
        self.limiter = RateLimiter(cfg.rate_limit_units_per_s)
        self.semaphore = threading.Semaphore(max(int(cfg.concurrency), 1))
        self._opener = opener or urllib.request.urlopen
        self._disabled_at_runtime: set[str] = set()
        self._id = 0
        self._id_lock = threading.Lock()
        self._key = credential(cfg.key_var) if cfg.key_var else None
        self._url = self._build_url()
        register_secret(self._url)

    # --- availability --------------------------------------------------------
    def _build_url(self) -> str | None:
        if self.cfg.url_template and "{key}" in self.cfg.url_template:
            if not self._key:
                return self.cfg.public_url or None
            return self.cfg.url_template.replace("{key}", self._key)
        return self.cfg.url_template or self.cfg.public_url or None

    @property
    def has_credential(self) -> bool:
        return bool(self._key)

    @property
    def available(self) -> bool:
        return bool(self.cfg.enabled and self._url)

    def supports(self, method: str) -> bool:
        return (self.cfg.supports(method) and method not in self._disabled_at_runtime
                and self.available)

    def disable_method(self, method: str, reason: str) -> None:
        if method not in self._disabled_at_runtime:
            self._disabled_at_runtime.add(method)
            log.warning("%s: disabling %s for this run: %s", self.name, method, reason)

    def unit_cost(self, method: str) -> int:
        return self.cfg.unit_cost(method)

    def status(self) -> dict[str, Any]:
        return {"provider": self.name, "enabled": self.cfg.enabled,
                "credential_present": self.has_credential,
                "endpoint_configured": bool(self._url),
                "unit": self.cfg.unit,
                "units_source": self.cfg.units_source,
                "rate_limit_units_per_s": self.cfg.rate_limit_units_per_s,
                "concurrency": self.cfg.concurrency,
                "plan_unit_allowance": self.cfg.plan_unit_allowance,
                "run_unit_budget": self.cfg.run_unit_budget,
                "disabled_methods": sorted(set(self.cfg.disabled_methods) |
                                           self._disabled_at_runtime)}

    # --- calling -------------------------------------------------------------
    def call(self, method: str, params: list[Any]) -> RpcResult:
        if not self.supports(method):
            raise ProviderError(f"{self.name}: {method} is disabled for this run",
                                unsupported=True)
        units = self.unit_cost(method)
        body = self._envelope(method, params)
        return self._send(body, units=units, calls=1, label=method)

    def call_batch(self, calls: Iterable[tuple[str, list[Any]]]) -> list[Any]:
        """A JSON-RPC batch. Members are counted and charged individually."""
        items = list(calls)
        if not items:
            return []
        for method, _ in items:
            if not self.supports(method):
                raise ProviderError(f"{self.name}: {method} is disabled for this run",
                                    unsupported=True)
        units = sum(self.unit_cost(m) for m, _ in items)
        body = [self._envelope(m, p) for m, p in items]
        result = self._send(body, units=units, calls=len(items),
                            label=f"batch[{len(items)}]")
        payload = result.payload
        if not isinstance(payload, list):
            raise ProviderError(f"{self.name}: batch reply was not a list")
        by_id = {item.get("id"): item for item in payload if isinstance(item, dict)}
        return [self._unwrap(by_id.get(env["id"]), env["method"]) for env in body]

    def _envelope(self, method: str, params: list[Any]) -> dict[str, Any]:
        with self._id_lock:
            self._id += 1
            rpc_id = self._id
        return {"jsonrpc": "2.0", "id": rpc_id, "method": method, "params": params}

    def _send(self, body: Any, *, units: int, calls: int, label: str) -> RpcResult:
        if self.budget is not None:
            self.budget.charge(self.name, units, calls=calls)
        payload_bytes = json.dumps(body).encode("utf-8")
        attempt = 0
        waited = 0.0
        while True:
            attempt += 1
            waited += self.limiter.acquire(units)
            try:
                with self.semaphore:
                    raw = self._http_post(payload_bytes)
                parsed = json.loads(raw.decode("utf-8"))
                if isinstance(body, dict):
                    self._unwrap(parsed, body["method"])
                return RpcResult(payload=parsed, raw_bytes=raw, units=units, provider=self.name,
                                 attempts=attempt, waited_s=waited)
            except ProviderError as exc:
                if exc.unsupported:
                    raise
                if not exc.retryable or attempt > self.max_retries:
                    raise
                delay = exc.retry_after if exc.retry_after is not None else min(
                    self.max_delay_s, self.base_delay_s * (2 ** (attempt - 1)))
                delay = min(self.max_delay_s, delay + random.uniform(0, self.base_delay_s))
                if self.budget is not None:
                    with self.budget._lock:
                        self.budget.retries += 1
                log.info("%s: %s attempt %d failed (%s); retrying in %.2fs",
                         self.name, label, attempt, redact(exc), delay)
                time.sleep(delay)
                waited += delay

    def _http_post(self, payload: bytes) -> bytes:
        request = urllib.request.Request(
            self._url, data=payload, method="POST",
            headers={"Content-Type": "application/json", "Accept": "application/json",
                     "User-Agent": "txbench-collector/0.1 (research benchmark)"})
        try:
            with self._opener(request, timeout=self.timeout_s) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            retry_after = _parse_retry_after(exc.headers.get("Retry-After") if exc.headers else None)
            detail = ""
            try:
                detail = exc.read()[:400].decode("utf-8", "replace")
            except Exception:                                 # noqa: BLE001 - best effort
                pass
            message = f"{self.name}: HTTP {exc.code} {redact(detail)}"
            if exc.code == 429 or any(m in detail.lower() for m in THROTTLE_MARKERS):
                raise ProviderError(message, retryable=True, status=exc.code,
                                    retry_after=retry_after) from None
            if exc.code in (402, 403) and any(m in detail.lower() for m in UNSUPPORTED_MARKERS):
                raise ProviderError(message, unsupported=True, status=exc.code) from None
            if 500 <= exc.code < 600:
                raise ProviderError(message, retryable=True, status=exc.code,
                                    retry_after=retry_after) from None
            raise ProviderError(message, status=exc.code) from None
        except urllib.error.URLError as exc:
            raise ProviderError(f"{self.name}: transport error: {redact(exc.reason)}",
                                retryable=True) from None
        except TimeoutError:
            raise ProviderError(f"{self.name}: request timed out after {self.timeout_s}s",
                                retryable=True) from None

    def _unwrap(self, item: Any, method: str) -> Any:
        if item is None:
            raise ProviderError(f"{self.name}: no reply for {method} in batch", retryable=True)
        if not isinstance(item, dict):
            raise ProviderError(f"{self.name}: malformed reply for {method}")
        error = item.get("error")
        if error:
            code = error.get("code") if isinstance(error, dict) else None
            text = str(error.get("message") if isinstance(error, dict) else error)
            lowered = text.lower()
            if code in UNSUPPORTED_CODES or any(m in lowered for m in UNSUPPORTED_MARKERS):
                self.disable_method(method, f"provider reports: {redact(text)[:120]}")
                raise ProviderError(f"{self.name}: {method} unsupported: {redact(text)[:200]}",
                                    unsupported=True)
            if any(m in lowered for m in THROTTLE_MARKERS):
                raise ProviderError(f"{self.name}: throttled on {method}: {redact(text)[:200]}",
                                    retryable=True)
            raise ProviderError(f"{self.name}: {method} error {code}: {redact(text)[:200]}",
                                retryable=True)
        if "result" not in item:
            raise ProviderError(f"{self.name}: reply for {method} has neither result nor error")
        return item["result"]

    def result_of(self, outcome: RpcResult) -> Any:
        payload = outcome.payload
        if isinstance(payload, dict):
            return payload.get("result")
        return payload


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


def build_providers(cfgs: dict[str, ProviderConfig], *, budget: Budget | None = None,
                    timeout_s: float = 30.0, max_retries: int = 4,
                    base_delay_s: float = 0.5, max_delay_s: float = 30.0,
                    opener=None) -> dict[str, Provider]:
    return {name: Provider(cfg, timeout_s=timeout_s, max_retries=max_retries,
                           base_delay_s=base_delay_s, max_delay_s=max_delay_s,
                           budget=budget, opener=opener)
            for name, cfg in cfgs.items()}
