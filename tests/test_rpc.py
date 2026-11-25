"""Provider behaviour: throttling, unsupported methods, partial replies, budgets.

No network: a fake opener returns exactly the HTTP conditions we want to prove
the collector survives.
"""
from __future__ import annotations

import io
import json
import urllib.error
from email.message import Message

import pytest

from txbench.config import M_RECEIPT, M_TX, ProviderConfig
from txbench.errors import BudgetExceeded, ProviderError
from txbench.rpc import Budget, Provider, RateLimiter


def _headers(**values) -> Message:
    message = Message()
    for key, value in values.items():
        message[key.replace("_", "-")] = str(value)
    return message


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def cfg(**overrides) -> ProviderConfig:
    base = dict(name="fake", key_var="FAKE_KEY", url_template="https://example.invalid/v2/{key}",
                concurrency=1, rate_limit_units_per_s=10_000.0,
                method_units={M_TX: 20, M_RECEIPT: 20, "_default": 20},
                disabled_methods=["debug_traceTransaction"])
    base.update(overrides)
    return ProviderConfig(**base)


def provider(opener, *, budget=None, max_retries=3, **overrides) -> Provider:
    import os
    os.environ["FAKE_KEY"] = "secret-key-value-1234567890"
    try:
        return Provider(cfg(**overrides), opener=opener, budget=budget, max_retries=max_retries,
                        base_delay_s=0.0, max_delay_s=0.0)
    finally:
        os.environ.pop("FAKE_KEY", None)


def ok_reply(result):
    return lambda request, timeout=None: FakeResponse(
        json.dumps({"jsonrpc": "2.0", "id": 1, "result": result}).encode())


def test_throttling_retries_and_honours_retry_after():
    calls = {"n": 0}

    def opener(request, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.HTTPError(
                "https://example.invalid", 429, "Too Many Requests",
                _headers(retry_after="0"), io.BytesIO(b"rate limit exceeded"))
        return FakeResponse(json.dumps({"jsonrpc": "2.0", "id": 1,
                                        "result": {"hash": "0x1"}}).encode())

    p = provider(opener)
    result = p.call(M_TX, ["0x1"])
    assert calls["n"] == 2, "a 429 is retried once"
    assert result.attempts == 2
    assert p.result_of(result) == {"hash": "0x1"}


def test_throttling_gives_up_after_the_retry_bound():
    def opener(request, timeout=None):
        raise urllib.error.HTTPError("https://example.invalid", 429, "Too Many Requests",
                                     _headers(retry_after="0"), io.BytesIO(b"rate limit"))

    p = provider(opener, max_retries=2)
    with pytest.raises(ProviderError) as exc:
        p.call(M_TX, ["0x1"])
    assert exc.value.retryable is True


def test_unsupported_method_is_permanent_and_disables_itself():
    def opener(request, timeout=None):
        return FakeResponse(json.dumps({
            "jsonrpc": "2.0", "id": 1,
            "error": {"code": -32601, "message": "the method debug_traceTransaction "
                                                 "does not exist/is not available"}}).encode())

    p = provider(opener)
    with pytest.raises(ProviderError) as exc:
        p.call(M_TX, ["0x1"])
    assert exc.value.unsupported is True
    assert exc.value.retryable is False
    assert not p.supports(M_TX), "an unsupported method is disabled for the rest of the run"


def test_method_disabled_by_config_is_never_called():
    def opener(request, timeout=None):
        raise AssertionError("a disabled method must not reach the network")

    p = provider(opener)
    assert p.supports("debug_traceTransaction") is False
    with pytest.raises(ProviderError) as exc:
        p.call("debug_traceTransaction", ["0x1"])
    assert exc.value.unsupported is True


def test_partial_reply_without_result_or_error_is_rejected():
    def opener(request, timeout=None):
        return FakeResponse(json.dumps({"jsonrpc": "2.0", "id": 1}).encode())

    with pytest.raises(ProviderError, match="neither result nor error"):
        provider(opener).call(M_TX, ["0x1"])


def test_truncated_body_is_a_transport_level_failure():
    def opener(request, timeout=None):
        return FakeResponse(b'{"jsonrpc": "2.0", "id": 1, "resu')

    with pytest.raises(json.JSONDecodeError):
        provider(opener).call(M_TX, ["0x1"])


def test_batch_members_are_counted_individually():
    def opener(request, timeout=None):
        payload = json.loads(request.data.decode())
        return FakeResponse(json.dumps(
            [{"jsonrpc": "2.0", "id": item["id"], "result": {"hash": "0x1"}}
             for item in payload]).encode())

    budget = Budget(max_calls=10, per_provider_units={"fake": 1000})
    p = provider(opener, budget=budget)
    results = p.call_batch([(M_TX, ["0x1"]), (M_RECEIPT, ["0x1"]), (M_TX, ["0x2"])])
    assert len(results) == 3
    assert budget.calls == 3, "three members count as three calls, not one"
    assert budget.units["fake"] == 60


def test_budget_fuse_stops_the_run():
    budget = Budget(max_calls=2, per_provider_units={"fake": 1000})
    p = provider(ok_reply({"hash": "0x1"}), budget=budget)
    p.call(M_TX, ["0x1"])
    p.call(M_TX, ["0x2"])
    with pytest.raises(BudgetExceeded, match="call cap"):
        p.call(M_TX, ["0x3"])


def test_unit_budget_fuse_mentions_the_setting_to_raise():
    budget = Budget(max_calls=100, per_provider_units={"fake": 30})
    p = provider(ok_reply({"hash": "0x1"}), budget=budget)
    p.call(M_TX, ["0x1"])
    with pytest.raises(BudgetExceeded, match="run_unit_budget"):
        p.call(M_TX, ["0x2"])


def test_no_credential_means_no_endpoint():
    import os
    os.environ.pop("FAKE_KEY", None)
    p = Provider(cfg(), opener=ok_reply({}))
    assert p.has_credential is False
    assert p.available is False
    assert p.status()["credential_present"] is False


def test_status_never_exposes_the_key_or_the_url():
    p = provider(ok_reply({}))
    rendered = json.dumps(p.status())
    assert "secret-key-value" not in rendered
    assert "example.invalid" not in rendered


def test_error_messages_are_redacted():
    from txbench.credentials import redact, register_secret
    register_secret("secret-key-value-1234567890")
    text = redact("failed calling https://example.invalid/v2/secret-key-value-1234567890")
    assert "secret-key-value" not in text
    assert "REDACTED" in text


def test_rate_limiter_blocks_until_tokens_are_available():
    limiter = RateLimiter(units_per_second=1000.0)
    waited = limiter.acquire(10)
    assert waited >= 0.0
    big = RateLimiter(units_per_second=100.0)
    big.acquire(100)
    assert big.acquire(50) > 0.0, "a drained bucket must make the caller wait"
