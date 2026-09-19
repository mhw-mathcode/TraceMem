import base64
import importlib
from io import BytesIO

import httpx
import pytest
from PIL import Image

from tracemem.config import Settings
from tracemem.model_clients import DisabledExtractor, DisabledReranker, HashEmbedder
from tracemem.runtime import ModelBundle


def image_part():
    output = BytesIO()
    Image.new("RGB", (7, 1), "red").save(output, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()
    return {"type": "image_url", "image_url": {"url": url}}


class FakeVision:
    async def describe(self, image_url, context):
        return "a red bicycle beside a tree"


@pytest.mark.asyncio
async def test_http_image_only_add_then_text_search_returns_original_part(tmp_path, monkeypatch):
    app_module = importlib.import_module("tracemem.app")
    monkeypatch.setattr(app_module, "build_model_bundle", lambda settings, client: ModelBundle(
        embedder=HashEmbedder(), extractor=DisabledExtractor(),
        reranker=DisabledReranker(), vision=FakeVision(),
    ))
    settings = Settings(api_key="secret", database_path=tmp_path / "memory.db", profile="baseline")
    parts = [image_part()]
    app = app_module.create_app(settings)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            add = await client.post("/add", headers={"X-API-Key": "secret"}, json={
                "request_id": "r1", "user_id": "u1", "session_id": "s1",
                "messages": [{"role": "user", "content": parts}],
            })
            search = await client.post("/search", headers={"X-API-Key": "secret"}, json={
                "query": "red bicycle", "user_id": "u1", "top_k": 5,
            })
            invalid = await client.post("/add", headers={"X-API-Key": "secret"}, json={
                "request_id": "r2", "user_id": "u1", "session_id": "s1",
                "messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}
                ]}],
            })
    assert add.status_code == 200
    assert search.status_code == 200
    assert search.json()["data"][0]["content"] == parts
    assert invalid.status_code == 422
