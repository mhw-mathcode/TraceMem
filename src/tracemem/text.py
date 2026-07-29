from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any

import numpy as np
from pydantic import BaseModel


_TOKEN_PATTERN = re.compile(r"[a-z0-9]+|[\u3400-\u4dbf\u4e00-\u9fff]+")


def lexical_text(value: str) -> str:
    tokens: list[str] = []
    for token in _TOKEN_PATTERN.findall(value.lower()):
        if re.fullmatch(r"[\u3400-\u4dbf\u4e00-\u9fff]+", token):
            if len(token) == 1:
                tokens.append(token)
            else:
                tokens.extend(token[index : index + 2] for index in range(len(token) - 1))
        else:
            tokens.append(token)
    return " ".join(tokens)


def payload_hash(value: BaseModel | dict[str, Any]) -> str:
    if isinstance(value, BaseModel):
        serializable = value.model_dump(mode="json")
    else:
        serializable = value
    encoded = json.dumps(
        serializable,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:24]
    return f"{prefix}_{digest}"


def timestamp_to_datetime(value: int | float | str | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        seconds = float(value)
        if abs(seconds) > 10_000_000_000:
            seconds /= 1000.0
        return datetime.fromtimestamp(seconds, tz=timezone.utc)
    normalized = value.strip()
    if not normalized:
        return None
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def pack_vector(vector: np.ndarray) -> tuple[bytes, int]:
    normalized = np.asarray(vector, dtype=np.float32).reshape(-1)
    return normalized.tobytes(order="C"), int(normalized.size)


def unpack_vector(blob: bytes, dimensions: int) -> np.ndarray:
    vector = np.frombuffer(blob, dtype=np.float32, count=dimensions)
    return vector.copy()
