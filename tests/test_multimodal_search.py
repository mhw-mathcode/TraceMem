import base64
from datetime import datetime, timezone
from io import BytesIO

import pytest
from PIL import Image

from tracemem.api_models import SearchRequest
from tracemem.domain import Candidate
from tracemem.multimodal import ImagePart
from tracemem.search_service import SearchService


def image_part():
    output = BytesIO()
    Image.new("RGB", (7, 1), "red").save(output, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()
    return ImagePart.model_validate({"type": "image_url", "image_url": {"url": url}})


class FakeDatabase:
    def __init__(self, payloads):
        self.payloads = payloads

    def load_original_contents(self, ids):
        return {key: value for key, value in self.payloads.items() if key in ids}


class FakeRetriever:
    def __init__(self, candidates, payloads):
        self.candidates = candidates
        self.database = FakeDatabase(payloads)

    async def retrieve(self, **kwargs):
        return self.candidates


def candidate(id, content, score):
    return Candidate(id=id, source_type="episode", content=content,
                     created_at=datetime.now(timezone.utc), score=score)


@pytest.mark.asyncio
async def test_search_returns_original_image_parts_in_order():
    parts = [image_part()]
    service = SearchService(retriever=FakeRetriever(
        [candidate("e1", "[Image: red bicycle]", 3)], {"e1": parts}
    ))
    result = await service.search(SearchRequest(query="bicycle", user_id="u1", top_k=5))
    assert result.data[0].content == parts


@pytest.mark.asyncio
async def test_search_skips_whole_image_result_that_exceeds_remaining_budget(monkeypatch):
    import tracemem.search_service as search_module

    parts = [image_part()]
    monkeypatch.setattr(search_module, "MAX_RESPONSE_IMAGE_BYTES", 100, raising=False)
    service = SearchService(retriever=FakeRetriever([
        candidate("e1", "red bicycle", 3),
        candidate("e2", "blue boat", 2),
        candidate("e3", "green tree", 1),
    ], {"e1": parts, "e2": parts}))
    result = await service.search(SearchRequest(query="object", user_id="u1", top_k=2))
    assert [item.id for item in result.data] == ["e1", "e3"]
    assert isinstance(result.data[1].content, str)
