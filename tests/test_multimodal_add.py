import base64
from io import BytesIO

import pytest
from PIL import Image

from tracemem.add_service import AddDependencyError, AddService
from tracemem.api_models import AddRequest
from tracemem.db import Database
from tracemem.model_clients import DisabledExtractor, HashEmbedder
from tracemem.vision import VisionError


def make_request(content):
    return AddRequest.model_validate({
        "request_id": "r1", "user_id": "u1", "session_id": "s1",
        "messages": [{"role": "user", "content": content}],
    })


def image_part():
    output = BytesIO()
    Image.new("RGB", (7, 1), "red").save(output, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()
    return {"type": "image_url", "image_url": {"url": url}}


class FakeVision:
    def __init__(self, failures=0):
        self.failures = failures
        self.calls = 0

    async def describe(self, image_url, context):
        self.calls += 1
        if self.calls <= self.failures:
            raise VisionError("vision failed")
        return "a red bicycle beside a tree"


@pytest.mark.asyncio
async def test_image_only_add_indexes_visible_content(tmp_path):
    database = Database(tmp_path / "memory.db")
    database.initialize()
    vision = FakeVision()
    service = AddService(database=database, embedder=HashEmbedder(),
                         extractor=DisabledExtractor(), vision=vision)
    await service.add(make_request([image_part()]))

    assert vision.calls == 1
    assert database.count_episodes("u1") == 1
    assert database.search_episode_fts("u1", '"bicycle"', 5)
    assert "base64," not in database.load_episode_vectors("u1")[0].candidate.content


@pytest.mark.asyncio
async def test_failed_vision_add_can_retry_same_request_id(tmp_path):
    database = Database(tmp_path / "memory.db")
    database.initialize()
    vision = FakeVision(failures=1)
    service = AddService(database=database, embedder=HashEmbedder(),
                         extractor=DisabledExtractor(), vision=vision)
    request = make_request([image_part()])
    with pytest.raises(AddDependencyError):
        await service.add(request)
    assert database.count_episodes("u1") == 0
    await service.add(request)
    assert database.count_episodes("u1") == 1


@pytest.mark.asyncio
async def test_text_only_add_does_not_call_vision(tmp_path):
    database = Database(tmp_path / "memory.db")
    database.initialize()
    vision = FakeVision()
    service = AddService(database=database, embedder=HashEmbedder(),
                         extractor=DisabledExtractor(), vision=vision)
    await service.add(make_request("ordinary text"))
    assert vision.calls == 0
