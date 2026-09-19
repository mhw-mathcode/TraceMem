import base64
from datetime import datetime, timezone
from io import BytesIO

import numpy as np
from PIL import Image

from tracemem.api_models import AddRequest
from tracemem.db import Database
from tracemem.domain import EpisodeDraft


def original_parts():
    output = BytesIO()
    Image.new("RGB", (7, 1), "red").save(output, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()
    request = AddRequest.model_validate({
        "request_id": "r1", "user_id": "u1", "session_id": "s1",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "photo"},
            {"type": "image_url", "image_url": {"url": url}},
        ]}],
    })
    return request.messages[0].content


def test_original_parts_roundtrip_without_polluting_search_index(tmp_path):
    database = Database(tmp_path / "memory.db")
    database.initialize()
    database.claim_add("r1", "u1", "s1", "hash", 120)
    now = datetime.now(timezone.utc)
    parts = original_parts()
    episode = EpisodeDraft(
        id="episode-1", user_id="u1", session_id="s1", request_id="r1",
        message_index=0, role="user", content="photo\n[Image: red bicycle]",
        lexical_text="photo red bicycle", event_time=None, ingested_at=now,
        importance=1.0, embedding=np.array([1.0, 0.0], dtype=np.float32),
        original_content=parts,
    )
    database.commit_add(request_id="r1", episodes=[episode], raw_only=True)

    assert database.load_original_contents(["episode-1"])["episode-1"] == parts
    assert "base64," not in database.load_episode_vectors("u1")[0].candidate.content
    database.initialize()
    assert database.load_original_contents(["episode-1"])["episode-1"] == parts
