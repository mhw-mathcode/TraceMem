from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

from tracemem.domain import (
    Candidate,
    ClaimResult,
    ClaimStatus,
    EpisodeDraft,
    ExistingStateVersion,
    MemoryCardDraft,
    StateVersionDraft,
    VectorRecord,
)
from tracemem.text import pack_vector, unpack_vector


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime | None) -> str | None:
    return value.astimezone(timezone.utc).isoformat() if value else None


def _datetime(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


class Database:
    def __init__(
        self,
        path: str | Path,
        *,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self.path = Path(path)
        self.clock = clock

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    @contextmanager
    def _session(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._session() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS add_requests (
                    request_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    status TEXT NOT NULL,
                    lease_until TEXT,
                    raw_only INTEGER NOT NULL DEFAULT 0,
                    error_code TEXT,
                    created_at TEXT NOT NULL,
                    completed_at TEXT
                );

                CREATE TABLE IF NOT EXISTS episodes (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    request_id TEXT NOT NULL REFERENCES add_requests(request_id),
                    message_index INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    lexical_text TEXT NOT NULL,
                    event_time TEXT,
                    ingested_at TEXT NOT NULL,
                    importance REAL NOT NULL,
                    embedding BLOB NOT NULL,
                    embedding_dimensions INTEGER NOT NULL,
                    UNIQUE(request_id, message_index)
                );
                CREATE INDEX IF NOT EXISTS episodes_user_idx
                    ON episodes(user_id, ingested_at);

                CREATE VIRTUAL TABLE IF NOT EXISTS episode_fts USING fts5(
                    user_id UNINDEXED,
                    memory_id UNINDEXED,
                    lexical_text
                );

                CREATE TABLE IF NOT EXISTS memory_cards (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    object_value TEXT NOT NULL,
                    event_time TEXT,
                    valid_from TEXT,
                    valid_to TEXT,
                    ingested_at TEXT NOT NULL,
                    polarity TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    importance REAL NOT NULL,
                    status TEXT NOT NULL,
                    source_episode_ids TEXT NOT NULL,
                    search_text TEXT NOT NULL,
                    lexical_text TEXT NOT NULL,
                    relation_hint TEXT,
                    embedding BLOB NOT NULL,
                    embedding_dimensions INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS memory_cards_user_idx
                    ON memory_cards(user_id, ingested_at);

                CREATE VIRTUAL TABLE IF NOT EXISTS card_fts USING fts5(
                    user_id UNINDEXED,
                    memory_id UNINDEXED,
                    lexical_text
                );

                CREATE TABLE IF NOT EXISTS state_versions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    state_key TEXT NOT NULL,
                    memory_card_id TEXT NOT NULL REFERENCES memory_cards(id),
                    relation_to_previous TEXT NOT NULL,
                    previous_version_id TEXT,
                    valid_from TEXT,
                    valid_to TEXT,
                    is_current INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS state_versions_lookup_idx
                    ON state_versions(user_id, state_key, is_current);

                CREATE TABLE IF NOT EXISTS entities (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    canonical_name TEXT NOT NULL,
                    entity_type TEXT,
                    UNIQUE(user_id, canonical_name)
                );

                CREATE TABLE IF NOT EXISTS entity_aliases (
                    user_id TEXT NOT NULL,
                    alias TEXT NOT NULL,
                    entity_id TEXT NOT NULL REFERENCES entities(id),
                    PRIMARY KEY(user_id, alias)
                );
                """
            )

    def claim_add(
        self,
        request_id: str,
        user_id: str,
        session_id: str,
        payload_hash: str,
        lease_seconds: int,
    ) -> ClaimResult:
        now = self.clock().astimezone(timezone.utc)
        lease_until = now + timedelta(seconds=lease_seconds)
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM add_requests WHERE request_id = ?",
                (request_id,),
            ).fetchone()
            if row is None:
                connection.execute(
                    """
                    INSERT INTO add_requests(
                        request_id, user_id, session_id, payload_hash, status,
                        lease_until, created_at
                    ) VALUES (?, ?, ?, ?, 'processing', ?, ?)
                    """,
                    (
                        request_id,
                        user_id,
                        session_id,
                        payload_hash,
                        _iso(lease_until),
                        _iso(now),
                    ),
                )
                connection.commit()
                return ClaimResult(
                    ClaimStatus.CLAIMED,
                    request_id,
                    user_id,
                    session_id,
                    lease_until,
                )
            if row["payload_hash"] != payload_hash:
                connection.rollback()
                return ClaimResult(
                    ClaimStatus.CONFLICT,
                    request_id,
                    row["user_id"],
                    row["session_id"],
                )
            if row["status"] == "completed":
                connection.rollback()
                return ClaimResult(
                    ClaimStatus.COMPLETED,
                    request_id,
                    row["user_id"],
                    row["session_id"],
                )
            existing_lease = _datetime(row["lease_until"])
            if existing_lease and existing_lease > now:
                connection.rollback()
                return ClaimResult(
                    ClaimStatus.WAIT,
                    request_id,
                    row["user_id"],
                    row["session_id"],
                    existing_lease,
                )
            connection.execute(
                """
                UPDATE add_requests
                SET status = 'processing', lease_until = ?, error_code = NULL
                WHERE request_id = ?
                """,
                (_iso(lease_until), request_id),
            )
            connection.commit()
            return ClaimResult(
                ClaimStatus.CLAIMED,
                request_id,
                row["user_id"],
                row["session_id"],
                lease_until,
            )
        finally:
            connection.close()

    def commit_add(
        self,
        *,
        request_id: str,
        episodes: Sequence[EpisodeDraft],
        cards: Sequence[MemoryCardDraft] = (),
        state_versions: Sequence[StateVersionDraft] = (),
        close_version_ids: Sequence[str] = (),
        raw_only: bool,
    ) -> None:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            request_row = connection.execute(
                "SELECT status FROM add_requests WHERE request_id = ?",
                (request_id,),
            ).fetchone()
            if request_row is None or request_row["status"] != "processing":
                raise RuntimeError("Add request is not claimed for processing")

            for item in episodes:
                vector_blob, dimensions = pack_vector(item.embedding)
                connection.execute(
                    """
                    INSERT INTO episodes(
                        id, user_id, session_id, request_id, message_index, role,
                        content, lexical_text, event_time, ingested_at, importance,
                        embedding, embedding_dimensions
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        item.id,
                        item.user_id,
                        item.session_id,
                        item.request_id,
                        item.message_index,
                        item.role,
                        item.content,
                        item.lexical_text,
                        _iso(item.event_time),
                        _iso(item.ingested_at),
                        item.importance,
                        vector_blob,
                        dimensions,
                    ),
                )
                connection.execute(
                    "INSERT INTO episode_fts(user_id, memory_id, lexical_text) VALUES (?, ?, ?)",
                    (item.user_id, item.id, item.lexical_text),
                )

            for card in cards:
                vector_blob, dimensions = pack_vector(card.embedding)
                connection.execute(
                    """
                    INSERT INTO memory_cards(
                        id, user_id, kind, subject, predicate, object_value,
                        event_time, valid_from, valid_to, ingested_at, polarity,
                        confidence, importance, status, source_episode_ids,
                        search_text, lexical_text, relation_hint, embedding,
                        embedding_dimensions
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        card.id,
                        card.user_id,
                        card.kind,
                        card.subject,
                        card.predicate,
                        card.object,
                        _iso(card.event_time),
                        _iso(card.valid_from),
                        _iso(card.valid_to),
                        _iso(card.ingested_at),
                        card.polarity,
                        card.confidence,
                        card.importance,
                        card.status,
                        json.dumps(card.source_episode_ids),
                        card.search_text,
                        card.lexical_text,
                        card.relation_hint,
                        vector_blob,
                        dimensions,
                    ),
                )
                connection.execute(
                    "INSERT INTO card_fts(user_id, memory_id, lexical_text) VALUES (?, ?, ?)",
                    (card.user_id, card.id, card.lexical_text),
                )

            if close_version_ids:
                placeholders = ",".join("?" for _ in close_version_ids)
                connection.execute(
                    f"UPDATE state_versions SET is_current = 0 WHERE id IN ({placeholders})",
                    tuple(close_version_ids),
                )
            for version in state_versions:
                connection.execute(
                    """
                    INSERT INTO state_versions(
                        id, user_id, state_key, memory_card_id,
                        relation_to_previous, previous_version_id, valid_from,
                        valid_to, is_current
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        version.id,
                        version.user_id,
                        version.state_key,
                        version.memory_card_id,
                        version.relation_to_previous,
                        version.previous_version_id,
                        _iso(version.valid_from),
                        _iso(version.valid_to),
                        int(version.is_current),
                    ),
                )

            connection.execute(
                """
                UPDATE add_requests
                SET status = 'completed', lease_until = NULL, raw_only = ?,
                    completed_at = ?
                WHERE request_id = ?
                """,
                (int(raw_only), _iso(self.clock()), request_id),
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def mark_add_failed(self, request_id: str, error_code: str) -> None:
        with self._session() as connection:
            connection.execute(
                """
                UPDATE add_requests
                SET status = 'failed', lease_until = NULL, error_code = ?
                WHERE request_id = ?
                """,
                (error_code, request_id),
            )

    def count_episodes(self, user_id: str) -> int:
        with self._session() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM episodes WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        return int(row["count"])

    def count_cards(self, user_id: str) -> int:
        with self._session() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM memory_cards WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        return int(row["count"])

    def search_episode_fts(
        self,
        user_id: str,
        query: str,
        limit: int,
    ) -> list[Candidate]:
        with self._session() as connection:
            rows = connection.execute(
                """
                SELECT e.*, -bm25(episode_fts) AS relevance
                FROM episode_fts
                JOIN episodes e ON e.id = episode_fts.memory_id
                WHERE episode_fts.user_id = ? AND episode_fts MATCH ?
                ORDER BY relevance DESC, e.id
                LIMIT ?
                """,
                (user_id, query, limit),
            ).fetchall()
        return [self._episode_candidate(row, float(row["relevance"])) for row in rows]

    def search_card_fts(
        self,
        user_id: str,
        query: str,
        limit: int,
    ) -> list[Candidate]:
        with self._session() as connection:
            rows = connection.execute(
                """
                SELECT c.*, -bm25(card_fts) AS relevance
                FROM card_fts
                JOIN memory_cards c ON c.id = card_fts.memory_id
                WHERE card_fts.user_id = ? AND card_fts MATCH ?
                ORDER BY relevance DESC, c.id
                LIMIT ?
                """,
                (user_id, query, limit),
            ).fetchall()
        return [self._card_candidate(row, float(row["relevance"])) for row in rows]

    def load_episode_vectors(self, user_id: str) -> list[VectorRecord]:
        with self._session() as connection:
            rows = connection.execute(
                "SELECT * FROM episodes WHERE user_id = ? ORDER BY id",
                (user_id,),
            ).fetchall()
        return [
            VectorRecord(
                candidate=self._episode_candidate(row, 0.0),
                vector=unpack_vector(row["embedding"], row["embedding_dimensions"]),
            )
            for row in rows
        ]

    def load_card_vectors(self, user_id: str) -> list[VectorRecord]:
        with self._session() as connection:
            rows = connection.execute(
                "SELECT * FROM memory_cards WHERE user_id = ? ORDER BY id",
                (user_id,),
            ).fetchall()
        return [
            VectorRecord(
                candidate=self._card_candidate(row, 0.0),
                vector=unpack_vector(row["embedding"], row["embedding_dimensions"]),
            )
            for row in rows
        ]

    def load_current_versions(
        self,
        user_id: str,
        state_keys: Sequence[str] = (),
    ) -> list[ExistingStateVersion]:
        sql = """
            SELECT sv.*, c.object_value, c.event_time
            FROM state_versions sv
            JOIN memory_cards c ON c.id = sv.memory_card_id
            WHERE sv.user_id = ? AND sv.is_current = 1
        """
        parameters: list[object] = [user_id]
        if state_keys:
            placeholders = ",".join("?" for _ in state_keys)
            sql += f" AND sv.state_key IN ({placeholders})"
            parameters.extend(state_keys)
        with self._session() as connection:
            rows = connection.execute(sql, tuple(parameters)).fetchall()
        return [
            ExistingStateVersion(
                id=row["id"],
                user_id=row["user_id"],
                state_key=row["state_key"],
                memory_card_id=row["memory_card_id"],
                object=row["object_value"],
                event_time=_datetime(row["event_time"]),
                valid_from=_datetime(row["valid_from"]),
                valid_to=_datetime(row["valid_to"]),
                is_current=bool(row["is_current"]),
            )
            for row in rows
        ]

    def load_state_candidates(
        self,
        user_id: str,
        state_keys: Sequence[str] = (),
    ) -> list[Candidate]:
        sql = """
            SELECT c.*, sv.state_key, sv.is_current
            FROM state_versions sv
            JOIN memory_cards c ON c.id = sv.memory_card_id
            WHERE sv.user_id = ?
        """
        parameters: list[object] = [user_id]
        if state_keys:
            placeholders = ",".join("?" for _ in state_keys)
            sql += f" AND sv.state_key IN ({placeholders})"
            parameters.extend(state_keys)
        sql += " ORDER BY sv.is_current DESC, c.event_time DESC, c.id"
        with self._session() as connection:
            rows = connection.execute(sql, tuple(parameters)).fetchall()
        return [
            self._card_candidate(
                row,
                1.0 if row["is_current"] else 0.5,
                state_key=row["state_key"],
                is_current=bool(row["is_current"]),
            )
            for row in rows
        ]

    @staticmethod
    def _episode_candidate(row: sqlite3.Row, score: float) -> Candidate:
        return Candidate(
            id=row["id"],
            source_type="episode",
            content=row["content"],
            created_at=_datetime(row["ingested_at"]) or _utc_now(),
            score=score,
            event_time=_datetime(row["event_time"]),
            source_episode_ids=(row["id"],),
        )

    @staticmethod
    def _card_candidate(
        row: sqlite3.Row,
        score: float,
        *,
        state_key: str | None = None,
        is_current: bool | None = None,
    ) -> Candidate:
        sources = tuple(json.loads(row["source_episode_ids"]))
        return Candidate(
            id=row["id"],
            source_type="card",
            content=row["search_text"],
            created_at=_datetime(row["ingested_at"]) or _utc_now(),
            score=score,
            event_time=_datetime(row["event_time"]),
            state_key=state_key,
            is_current=is_current,
            memory_kind=row["kind"],
            source_episode_ids=sources,
            metadata={
                "subject": row["subject"],
                "predicate": row["predicate"],
                "object": row["object_value"],
            },
        )
