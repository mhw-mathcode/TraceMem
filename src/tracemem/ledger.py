from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

from tracemem.domain import (
    ExistingStateVersion,
    MemoryCardDraft,
    StateVersionDraft,
)
from tracemem.text import stable_id


_NORMALIZE = re.compile(r"[^\w\u3400-\u4dbf\u4e00-\u9fff]+", re.UNICODE)


def normalize_state_part(value: str) -> str:
    return _NORMALIZE.sub(" ", value.casefold()).strip()


def state_key(subject: str, predicate: str) -> str:
    return f"{normalize_state_part(subject)}|{normalize_state_part(predicate)}"


@dataclass(frozen=True)
class LedgerPlan:
    new_versions: tuple[StateVersionDraft, ...]
    close_version_ids: tuple[str, ...]


class Ledger:
    def plan(
        self,
        *,
        existing_versions: Sequence[ExistingStateVersion],
        cards: Sequence[MemoryCardDraft],
    ) -> LedgerPlan:
        current_by_key = {
            version.state_key: version
            for version in existing_versions
            if version.is_current
        }
        new_versions: list[StateVersionDraft] = []
        close_ids: list[str] = []

        for card in cards:
            key = state_key(card.subject, card.predicate)
            previous = current_by_key.get(key)
            relation, is_current, close_previous = self._relation(previous, card)
            version = StateVersionDraft(
                id=stable_id("state", card.user_id, key, card.id),
                user_id=card.user_id,
                state_key=key,
                memory_card_id=card.id,
                relation_to_previous=relation,
                previous_version_id=previous.id if previous else None,
                valid_from=card.valid_from or card.event_time,
                valid_to=None,
                is_current=is_current,
            )
            new_versions.append(version)
            if previous and close_previous:
                close_ids.append(previous.id)
            if is_current:
                current_by_key[key] = ExistingStateVersion(
                    id=version.id,
                    user_id=version.user_id,
                    state_key=version.state_key,
                    memory_card_id=version.memory_card_id,
                    object=card.object,
                    event_time=card.event_time,
                    valid_from=version.valid_from,
                    valid_to=None,
                    is_current=True,
                )

        return LedgerPlan(tuple(new_versions), tuple(dict.fromkeys(close_ids)))

    @staticmethod
    def _relation(
        previous: ExistingStateVersion | None,
        card: MemoryCardDraft,
    ) -> tuple[str, bool, bool]:
        if previous is None:
            return "continue", True, False
        if card.relation_hint == "correction":
            return "correction", True, True
        if normalize_state_part(previous.object) == normalize_state_part(card.object):
            return "continue", True, True
        if (
            card.event_time is not None
            and previous.event_time is not None
            and card.event_time > previous.event_time
        ):
            return "update", True, True
        if card.relation_hint == "update" and card.event_time != previous.event_time:
            return "update", True, True
        return "conflict", False, False
