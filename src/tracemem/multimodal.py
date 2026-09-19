from __future__ import annotations

import base64
import binascii
from io import BytesIO
from typing import Annotated, Literal

from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, field_validator


MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_REQUEST_IMAGE_BYTES = 30 * 1024 * 1024
MAX_RESPONSE_IMAGE_BYTES = 30 * 1024 * 1024
_FORMATS = {"jpeg": "JPEG", "png": "PNG", "webp": "WEBP"}


def decoded_image_size(url: str) -> int:
    if not url.startswith("data:image/") or ";base64," not in url:
        raise ValueError("image must be an inline Base64 Data URI")
    header, payload = url.split(",", 1)
    mime = header.removeprefix("data:image/").removesuffix(";base64")
    if header != f"data:image/{mime};base64" or mime not in _FORMATS:
        raise ValueError("image must be JPEG, PNG, or WebP")
    if not payload or len(payload) > 4 * ((MAX_IMAGE_BYTES + 2) // 3):
        raise ValueError("image exceeds 10 MiB or is empty")
    try:
        data = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("image contains invalid Base64") from error
    if not data or len(data) > MAX_IMAGE_BYTES:
        raise ValueError("image exceeds 10 MiB or is empty")
    try:
        with Image.open(BytesIO(data)) as picture:
            if picture.format != _FORMATS[mime]:
                raise ValueError("image MIME type does not match its data")
            picture.verify()
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError("image data is not a valid JPEG, PNG, or WebP") from error
    return len(data)


class TextPart(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["text"]
    text: str = Field(min_length=1)


class ImageURL(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str

    @field_validator("url")
    @classmethod
    def validate_url(cls, url: str) -> str:
        decoded_image_size(url)
        return url


class ImagePart(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["image_url"]
    image_url: ImageURL


ContentPart = Annotated[TextPart | ImagePart, Field(discriminator="type")]
Content = Annotated[
    Annotated[str, Field(min_length=1)] | Annotated[list[ContentPart], Field(min_length=1)],
    Field(union_mode="left_to_right"),
]


def image_bytes(content: Content) -> int:
    if isinstance(content, str):
        return 0
    total = 0
    for part in content:
        if isinstance(part, ImagePart):
            payload = part.image_url.url.split(",", 1)[1]
            total += len(payload) * 3 // 4 - (len(payload) - len(payload.rstrip("=")))
    return total


def context_text(content: Content) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(part.text for part in content if isinstance(part, TextPart))


def index_text(content: Content, descriptions: list[str]) -> str:
    if isinstance(content, str):
        return content
    output: list[str] = []
    image_index = 0
    for part in content:
        if isinstance(part, TextPart):
            output.append(part.text)
        else:
            output.append(f"[Image: {descriptions[image_index]}]")
            image_index += 1
    if image_index != len(descriptions):
        raise ValueError("description count does not match images")
    return "\n".join(output)
