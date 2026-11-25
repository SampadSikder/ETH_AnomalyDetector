"""Logging with mandatory credential redaction."""
from __future__ import annotations

import logging
import sys

from .credentials import redact

_CONFIGURED = False


class _RedactingFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return redact(super().format(record))


def setup_logging(verbosity: int = 0, *, stream=None) -> None:
    global _CONFIGURED
    level = logging.WARNING if verbosity < 0 else (logging.INFO if verbosity == 0 else logging.DEBUG)
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(_RedactingFormatter("%(asctime)s %(levelname)-7s %(name)s  %(message)s",
                                             datefmt="%H:%M:%S"))
    root = logging.getLogger("txbench")
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    if not _CONFIGURED:
        setup_logging()
    return logging.getLogger(f"txbench.{name}")
