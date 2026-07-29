from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class MemoryMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant", "system", "tool"]
    timestamp: int | float | str | None = None
    content: str = Field(min_length=1)


class AddRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=1)
    messages: list[MemoryMessage] = Field(min_length=1)
    user_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)


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
    top_k: int = Field(ge=1, le=90)


class SearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    content: str = Field(min_length=1)
    score: float | None = None
    created_at: datetime | None = None


class SearchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data: list[SearchResult]
