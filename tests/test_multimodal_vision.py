import json

import httpx
import pytest

from tracemem.vision import OpenAIVisionDescriber, VisionError, VisionUnavailable


PNG_URL = "data:image/png;base64,aGVsbG8="


@pytest.mark.asyncio
async def test_vision_sends_inline_data_uri_to_chat_completions():
    captured = {}

    def respond(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers["Authorization"]
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "A red bicycle beside a tree."}}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        describer = OpenAIVisionDescriber(
            client=client, url="https://api.openai.com/v1", api_key="key",
            model="gpt-4o-mini", max_retries=0,
        )
        result = await describer.describe(PNG_URL, "a trip photo")
    assert result == "A red bicycle beside a tree."
    assert captured["url"] == "https://api.openai.com/v1/chat/completions"
    assert captured["auth"] == "Bearer key"
    assert captured["body"]["model"] == "gpt-4o-mini"
    assert captured["body"]["messages"][1]["content"][1] == {
        "type": "image_url", "image_url": {"url": PNG_URL}
    }


@pytest.mark.asyncio
async def test_vision_rejects_missing_key_without_network_call():
    def fail_if_called(request: httpx.Request) -> httpx.Response:
        raise AssertionError("network was called")

    async with httpx.AsyncClient(transport=httpx.MockTransport(fail_if_called)) as client:
        describer = OpenAIVisionDescriber(
            client=client, url="https://api.openai.com/v1", api_key="", model="gpt-4o-mini"
        )
        with pytest.raises(VisionUnavailable):
            await describer.describe(PNG_URL, "")


@pytest.mark.asyncio
async def test_vision_rejects_empty_response_without_exposing_image():
    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": " "}}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        describer = OpenAIVisionDescriber(
            client=client, url="https://api.openai.com/v1", api_key="key",
            model="gpt-4o-mini", max_retries=0,
        )
        with pytest.raises(VisionError) as error:
            await describer.describe(PNG_URL, "")
    assert "aGVsbG8=" not in str(error.value)
