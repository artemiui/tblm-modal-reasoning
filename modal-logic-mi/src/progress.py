from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable


def setup_file_logger(name: str, log_file: Path | str | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter("[%(levelname)s] %(asctime)s - %(name)s: %(message)s"))
        logger.addHandler(console)
        if log_file:
            path = Path(log_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(str(path), encoding="utf-8")
            file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
            logger.addHandler(file_handler)
    return logger


def log_event(logger: logging.Logger, payload: Mapping[str, Any] | str) -> None:
    if isinstance(payload, str):
        logger.info(payload)
    else:
        logger.info(json.dumps(dict(payload), ensure_ascii=True, default=str))


def make_tqdm(iterable: Any, **kwargs: Any) -> Any:
    return tqdm(iterable, **kwargs)


def resolve_log_path(output_dir: Path | str | None = None, output_path: Path | str | None = None, filename: str = "run.log") -> Path:
    if output_path is not None:
        p = Path(output_path)
        return p.parent / filename if p.suffix else p / filename
    if output_dir is not None:
        return Path(output_dir) / filename
    return Path("logs") / filename
