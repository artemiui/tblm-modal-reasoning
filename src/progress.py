from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

from tqdm.auto import tqdm


def resolve_log_path(output_path: Path) -> Path:
    """Derive a log file path from an output artifact path."""
    return output_path.with_suffix(".log")


def setup_file_logger(name: str, log_path: Path) -> logging.Logger:
    """Create a file logger that writes JSON-lines entries."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not any(isinstance(h, logging.FileHandler) and getattr(h, "_log_path", None) == str(log_path) for h in logger.handlers):
        handler = logging.FileHandler(str(log_path), mode="a", encoding="utf-8")
        handler._log_path = str(log_path)  # type: ignore[attr-defined]
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    return logger


def log_event(logger: logging.Logger, data: Any) -> None:
    """Log a structured event as a JSON line."""
    if isinstance(data, str):
        payload = data
    else:
        payload = json.dumps(
            {**data, "_ts": time.strftime("%Y-%m-%dT%H:%M:%S%z")},
            ensure_ascii=True,
            default=str,
        )
    logger.info(payload)


def make_tqdm(iterable=None, **kwargs) -> tqdm:
    """Create a tqdm progress bar with sensible defaults."""
    return tqdm(iterable, **kwargs)
