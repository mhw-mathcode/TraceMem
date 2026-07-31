from __future__ import annotations

import json
import logging
import re
import sys
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from time import perf_counter
from typing import TextIO, TypeAlias


Scalar: TypeAlias = str | int | float | bool | None | Path | Enum
_SAFE_TOKEN = re.compile(r"^[A-Za-z0-9._:/-]+$")
_FORMATTER = logging.Formatter(
    fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S%z",
)


def _render(value: Scalar) -> str:
    if isinstance(value, Enum):
        value = value.value
    if isinstance(value, Path):
        value = str(value)
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return str(value)
    if not isinstance(value, str):
        raise TypeError("log fields must be scalar values")
    return value if _SAFE_TOKEN.fullmatch(value) else json.dumps(value)


def log_event(
    logger: logging.Logger,
    level: int,
    event: str,
    **fields: Scalar,
) -> None:
    parts = [f"event={_render(event)}"]
    parts.extend(f"{key}={_render(value)}" for key, value in fields.items())
    logger.log(level, " ".join(parts))


def duration_ms(
    started: float,
    *,
    now: Callable[[], float] = perf_counter,
) -> int:
    return max(0, round((now() - started) * 1000))


def configure_logging(
    level: str,
    *,
    stream: TextIO | None = None,
) -> logging.Logger:
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"unsupported log level: {level}")

    logger = logging.getLogger("tracemem")
    tagged = [
        handler
        for handler in logger.handlers
        if getattr(handler, "_tracemem_handler", False)
    ]
    if tagged:
        handler = tagged[0]
        if stream is not None and handler.stream is not stream:
            handler.setStream(stream)
    else:
        handler = logging.StreamHandler(stream or sys.stderr)
        handler._tracemem_handler = True  # type: ignore[attr-defined]
        logger.addHandler(handler)

    handler.setFormatter(_FORMATTER)
    handler.setLevel(numeric_level)
    logger.setLevel(numeric_level)
    logger.propagate = False
    return logger
