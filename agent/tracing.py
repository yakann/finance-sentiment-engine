"""Day 20: structlog JSON configuration + per-run trace writer."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import structlog


def configure_logging(level: int = logging.INFO) -> None:
    """Call once at startup — configures structlog to emit newline-delimited JSON."""
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.ExceptionRenderer(),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def write_trace(run_id: str, data: dict[str, Any], traces_dir: str = "traces") -> str:
    """Persist per-run trace as JSON, return absolute path."""
    Path(traces_dir).mkdir(parents=True, exist_ok=True)
    path = Path(traces_dir) / f"run_{run_id}.json"
    path.write_text(json.dumps(data, indent=2, default=str))
    return str(path.resolve())
