from __future__ import annotations

from secrets import compare_digest

from fastapi import HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader


API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(
    request: Request,
    supplied_key: str | None = Security(API_KEY_HEADER),
) -> None:
    expected_key: str = request.app.state.api_key
    if supplied_key is None or not compare_digest(supplied_key, expected_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
