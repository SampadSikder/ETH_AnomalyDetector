"""No credential may reach version control, a log, a report or a cache record."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from conftest import REPO

# The variable names this project reads, taken from the package rather than
# written out here: a literal "NAME=" in this file would make the scan below
# flag itself.
def _assignment_markers() -> tuple[str, ...]:
    from txbench.credentials import CREDENTIAL_VARS
    return tuple(f"{name}{chr(61)}" for name in CREDENTIAL_VARS)


def _tracked_files() -> list[str]:
    result = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True,
                            check=False)
    return [line for line in result.stdout.splitlines() if line.strip()]


def test_dotenv_is_not_tracked():
    assert ".env" not in _tracked_files()


def test_dotenv_is_ignored_by_git():
    result = subprocess.run(["git", "check-ignore", ".env"], cwd=REPO, capture_output=True,
                            text=True, check=False)
    assert result.returncode == 0, ".env must be git-ignored"


def test_example_env_documents_the_variables_with_blank_values():
    from txbench.credentials import CREDENTIAL_VARS
    text = (REPO / ".env.example").read_text(encoding="utf-8")
    for name in CREDENTIAL_VARS:
        assert f"{name}=" in text
        value = [line for line in text.splitlines() if line.startswith(f"{name}=")][0]
        assert value.strip() == f"{name}=", "the example file must carry no value"


def test_no_tracked_file_assigns_a_credential():
    markers = _assignment_markers()
    offenders = []
    for relative in _tracked_files():
        path = REPO / relative
        if not path.is_file() or path.suffix in {".parquet", ".sqlite", ".gz", ".zip"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for marker in markers:
            for line in text.splitlines():
                if marker in line and not line.strip().startswith("#"):
                    value = line.split(marker, 1)[1].strip().strip("\"'")
                    # A blank value, a placeholder or a variable reference is fine.
                    if value and not value.startswith(("{", "$", "<")) and \
                            relative not in {".env.example"}:
                        offenders.append((relative, line.strip()[:80]))
    assert offenders == [], f"a credential assignment is committed: {offenders}"


def test_reports_and_json_artifacts_carry_no_key():
    import os
    key = os.environ.get("ALCHEMY_API_KEY") or ""
    suspicious = []
    for folder in ("reports", "data/manifest", "data/collection", "data/evaluation"):
        base = REPO / folder
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if path.suffix not in {".json", ".md"} or not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            if key and len(key) >= 8 and key in text:
                suspicious.append(str(path))
            if "g.alchemy.com/v2/" in text and "REDACTED" not in text:
                # An endpoint URL with a key segment must never be written out.
                for line in text.splitlines():
                    if "g.alchemy.com/v2/" in line and "{key}" not in line:
                        suspicious.append(f"{path}: {line.strip()[:60]}")
    assert suspicious == [], f"an artefact may contain a credential: {suspicious}"


def test_provider_config_stores_only_the_variable_name():
    from txbench.config import load_config
    settings = load_config().effective_settings()
    rendered = json.dumps(settings)
    assert "url_template" not in rendered, "a URL template never reaches a report"
    assert settings["providers"]["alchemy"]["key_var"] == "ALCHEMY_API_KEY"


def test_redaction_covers_an_unregistered_key_in_a_url():
    from txbench.credentials import redact
    text = redact("https://eth-mainnet.g.alchemy.com/v2/AbCdEf0123456789XYZ")
    assert "AbCdEf0123456789XYZ" not in text


def test_dotenv_loading_does_not_override_the_shell(tmp_path, monkeypatch):
    from txbench.credentials import ALCHEMY_KEY_VAR, load_dotenv
    # Built at runtime, so this file never contains a literal key assignment
    # for the scan above to find.
    (tmp_path / ".env").write_text(f"{ALCHEMY_KEY_VAR}{chr(61)}from-file\n", encoding="utf-8")
    monkeypatch.setenv(ALCHEMY_KEY_VAR, "from-shell")
    found = load_dotenv(tmp_path)
    assert ALCHEMY_KEY_VAR in found
    import os
    assert os.environ[ALCHEMY_KEY_VAR] == "from-shell"


def test_dotenv_file_is_not_rewritten(tmp_path):
    from txbench.credentials import ALCHEMY_KEY_VAR, load_dotenv
    path = tmp_path / ".env"
    original = f"# keep me\n{ALCHEMY_KEY_VAR}{chr(61)}abc123xyz\n"
    path.write_text(original, encoding="utf-8")
    load_dotenv(tmp_path)
    assert path.read_text(encoding="utf-8") == original, "an existing .env is preserved"
