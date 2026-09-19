import base64
from io import BytesIO

import pytest
from PIL import Image
from pydantic import ValidationError

from tracemem.api_models import AddRequest, SearchResult
from tracemem.multimodal import image_bytes


def image_url() -> str:
    output = BytesIO()
    Image.new("RGB", (7, 1), "red").save(output, format="PNG")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()


def request_with_content(content):
    return {
        "request_id": "r1",
        "messages": [{"role": "user", "content": content}],
        "user_id": "u1",
        "session_id": "s1",
    }


def test_add_accepts_ordered_text_and_image_parts():
    parts = [
        {"type": "text", "text": "first"},
        {"type": "image_url", "image_url": {"url": image_url()}},
        {"type": "text", "text": "last"},
    ]
    request = AddRequest.model_validate(request_with_content(parts))
    assert request.messages[0].model_dump(mode="json")["content"] == parts


def test_image_only_message_is_valid():
    request = AddRequest.model_validate(request_with_content([
        {"type": "image_url", "image_url": {"url": image_url()}}
    ]))
    assert request.messages[0].content[0].type == "image_url"


def test_image_bytes_counts_decoded_bytes_not_base64_characters():
    url = image_url()
    request = AddRequest.model_validate(request_with_content([
        {"type": "image_url", "image_url": {"url": url}}
    ]))
    assert image_bytes(request.messages[0].content) == len(base64.b64decode(url.split(",")[1]))


def test_search_result_accepts_ordered_parts():
    parts = [{"type": "image_url", "image_url": {"url": image_url()}}]
    result = SearchResult.model_validate({"id": "e1", "content": parts})
    assert result.model_dump(mode="json")["content"] == parts


@pytest.mark.parametrize("url", ["https://example.com/a.png", "data:image/png;base64,%%%"])
def test_add_rejects_remote_or_invalid_base64(url):
    with pytest.raises(ValidationError):
        AddRequest.model_validate(request_with_content([
            {"type": "image_url", "image_url": {"url": url}}
        ]))


def test_add_rejects_empty_text_part():
    with pytest.raises(ValidationError):
        AddRequest.model_validate(request_with_content([{"type": "text", "text": ""}]))


def test_add_rejects_whitespace_only_text_part():
    with pytest.raises(ValidationError):
        AddRequest.model_validate(request_with_content([{"type": "text", "text": "   "}]))


def test_add_rejects_mismatched_mime():
    url = image_url().replace("data:image/png;", "data:image/jpeg;")
    with pytest.raises(ValidationError):
        AddRequest.model_validate(request_with_content([
            {"type": "image_url", "image_url": {"url": url}}
        ]))


def test_add_rejects_aggregate_limit_across_messages(monkeypatch):
    import tracemem.api_models as api_models

    size = len(base64.b64decode(image_url().split(",")[1]))
    monkeypatch.setattr(api_models, "MAX_REQUEST_IMAGE_BYTES", size)
    payload = request_with_content([{"type": "image_url", "image_url": {"url": image_url()}}])
    payload["messages"].append(payload["messages"][0].copy())
    with pytest.raises(ValidationError):
        AddRequest.model_validate(payload)


def test_add_rejects_image_over_single_image_limit(monkeypatch):
    import tracemem.multimodal as multimodal

    size = len(base64.b64decode(image_url().split(",")[1]))
    monkeypatch.setattr(multimodal, "MAX_IMAGE_BYTES", size - 1)
    with pytest.raises(ValidationError):
        AddRequest.model_validate(request_with_content([
            {"type": "image_url", "image_url": {"url": image_url()}}
        ]))
