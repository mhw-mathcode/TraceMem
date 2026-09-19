from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tracemem.multimodal import Content, MAX_REQUEST_IMAGE_BYTES, image_bytes


class MemoryMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant", "system", "tool"]
    timestamp: int | float | str | None = None
    content: Content


class AddRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=1)
    messages: list[MemoryMessage] = Field(min_length=1)
    user_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_total_image_bytes(self) -> "AddRequest":
        if sum(image_bytes(message.content) for message in self.messages) > MAX_REQUEST_IMAGE_BYTES:
            raise ValueError("Add images exceed 30 MiB")
        return self


class AddResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    success: Literal[True] = True
    request_id: str
    user_id: str
    session_id: str


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    options: list[str] | None = None
    user_id: str = Field(min_length=1)
    top_k: int = Field(ge=1, le=120)


class SearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    content: Content
    score: float | None = None
    created_at: datetime | None = None


class SearchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data: list[SearchResult]
