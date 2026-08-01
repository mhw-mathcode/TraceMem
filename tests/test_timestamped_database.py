from __future__ import annotations

import re
import logging
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tracemem.app import _timestamped_database_path, create_app
from tracemem.config import Settings


def test_timestamped_database_path_inserts_utc_timestamp(
    tmp_path: Path,
) -> None:
    result = _timestamped_database_path(
        tmp_path / "tracemem.db",
        now=lambda: datetime(
            2026,
            8,
            1,
            6,
            30,
            52,
            123456,
            tzinfo=timezone.utc,
        ),
    )

    assert result == tmp_path / "tracemem-20260801T063052123456Z.db"


def test_timestamped_database_path_appends_timestamp_without_extension(
    tmp_path: Path,
) -> None:
    result = _timestamped_database_path(
        tmp_path / "tracemem",
        now=lambda: datetime(
            2026,
            8,
            1,
            6,
            30,
            52,
            123456,
            tzinfo=timezone.utc,
        ),
    )

    assert result == tmp_path / "tracemem-20260801T063052123456Z"


def test_timestamped_database_path_preserves_memory_database() -> None:
    assert _timestamped_database_path(Path(":memory:")) == Path(":memory:")


def test_timestamped_database_path_uses_process_id_on_collision(
    tmp_path: Path,
) -> None:
    primary = tmp_path / "tracemem-20260801T063052123456Z.db"
    primary.write_bytes(b"existing")

    result = _timestamped_database_path(
        tmp_path / "tracemem.db",
        now=lambda: datetime(
            2026,
            8,
            1,
            6,
            30,
            52,
            123456,
            tzinfo=timezone.utc,
        ),
        process_id=4321,
    )

    assert result == tmp_path / "tracemem-20260801T063052123456Z-p4321.db"
    assert result != primary


def _settings(database_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        api_key="service-secret",
        database_path=database_path,
        profile="baseline",
        embedding_mode="hash",
        extraction_mode="disabled",
        rerank_mode="disabled",
    )


@pytest.mark.asyncio
async def test_each_lifespan_gets_one_shared_timestamped_database(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "tracemem.app.configure_logging",
        lambda _level: logging.getLogger("tracemem"),
    )
    base_path = tmp_path / "runtime.db"
    first_app = create_app(_settings(base_path))
    async with first_app.router.lifespan_context(first_app):
        first_database = first_app.state.add_service.database
        assert (
            first_database
            is first_app.state.search_service.retriever.database
        )
        first_path = first_database.path

    second_app = create_app(_settings(base_path))
    async with second_app.router.lifespan_context(second_app):
        second_database = second_app.state.add_service.database
        assert (
            second_database
            is second_app.state.search_service.retriever.database
        )
        second_path = second_database.path

    name_pattern = re.compile(
        r"runtime-\d{8}T\d{12}Z(?:-p\d+)?\.db"
    )
    assert first_path.is_file()
    assert second_path.is_file()
    assert first_path != second_path
    assert name_pattern.fullmatch(first_path.name)
    assert name_pattern.fullmatch(second_path.name)
    assert not base_path.exists()
