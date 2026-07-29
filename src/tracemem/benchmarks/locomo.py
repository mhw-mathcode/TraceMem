from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Sequence


@dataclass(frozen=True)
class LocomoMessage:
    dia_id: str
    role: str
    content: str


@dataclass(frozen=True)
class LocomoSession:
    index: int
    timestamp: str | None
    messages: tuple[LocomoMessage, ...]


@dataclass(frozen=True)
class LocomoQuestion:
    question: str
    answers: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    category: str


@dataclass(frozen=True)
class LocomoSample:
    sample_id: str
    sessions: tuple[LocomoSession, ...]
    questions: tuple[LocomoQuestion, ...]
    evidence_text: dict[str, str]


_SESSION_PATTERN = re.compile(r"session_(\d+)$")


def parse_samples(payload: Sequence[dict[str, Any]]) -> list[LocomoSample]:
    samples: list[LocomoSample] = []
    for raw_sample in payload:
        conversation = raw_sample["conversation"]
        speaker_a = str(conversation.get("speaker_a", "speaker_a"))
        sessions: list[LocomoSession] = []
        evidence_text: dict[str, str] = {}
        session_fields: list[tuple[int, list[dict[str, Any]]]] = []
        for key, value in conversation.items():
            match = _SESSION_PATTERN.fullmatch(key)
            if match and isinstance(value, list):
                session_fields.append((int(match.group(1)), value))
        for index, raw_messages in sorted(session_fields):
            timestamp = _session_timestamp(
                conversation.get(f"session_{index}_date_time")
            )
            messages: list[LocomoMessage] = []
            for raw_message in raw_messages:
                speaker = str(raw_message.get("speaker", "unknown"))
                text = str(raw_message.get("text", "")).strip()
                dia_id = str(raw_message.get("dia_id", "")).strip()
                content = f"{speaker}: {text}"
                message = LocomoMessage(
                    dia_id=dia_id,
                    role="user" if speaker == speaker_a else "assistant",
                    content=content,
                )
                messages.append(message)
                if dia_id:
                    evidence_text[dia_id] = content
            if messages:
                sessions.append(
                    LocomoSession(
                        index=index,
                        timestamp=timestamp,
                        messages=tuple(messages),
                    )
                )

        questions = tuple(_parse_question(item) for item in raw_sample.get("qa", []))
        samples.append(
            LocomoSample(
                sample_id=str(raw_sample["sample_id"]),
                sessions=tuple(sessions),
                questions=questions,
                evidence_text=evidence_text,
            )
        )
    return samples


def _parse_question(raw: dict[str, Any]) -> LocomoQuestion:
    answer = raw.get("answer", "")
    if isinstance(answer, list):
        answers = tuple(str(item) for item in answer)
    else:
        answers = (str(answer),)
    evidence_ids: list[str] = []
    for evidence in raw.get("evidence", []):
        evidence_ids.extend(
            part.strip()
            for part in re.split(r"[;,]", str(evidence))
            if part.strip()
        )
    return LocomoQuestion(
        question=str(raw.get("question", "")),
        answers=answers,
        evidence_ids=tuple(dict.fromkeys(evidence_ids)),
        category=str(raw.get("category", "unknown")),
    )


def _session_timestamp(value: Any) -> str | None:
    if not value:
        return None
    text = str(value).strip()
    for pattern in (
        "%I:%M %p on %d %B, %Y",
        "%I:%M%p on %d %B, %Y",
        "%d %B, %Y",
    ):
        try:
            parsed = datetime.strptime(text, pattern).replace(tzinfo=timezone.utc)
            return parsed.isoformat()
        except ValueError:
            continue
    return None
