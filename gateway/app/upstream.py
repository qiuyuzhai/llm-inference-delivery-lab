from collections.abc import AsyncIterator
from typing import Any

import httpx
from fastapi import HTTPException
from starlette.responses import StreamingResponse


def build_chat_payload(body: dict[str, Any], default_model: str) -> dict[str, Any]:
    messages = body.get("messages")
    if not messages:
        raise ValueError("messages must not be empty")

    payload = dict(body)
    payload["model"] = payload.get("model") or default_model
    payload["messages"] = messages
    payload["stream"] = bool(payload.get("stream", False))
    return payload


async def post_chat_completion(
    *,
    base_url: str,
    payload: dict[str, Any],
    timeout_seconds: int,
) -> dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            response = await client.post(f"{base_url}/v1/chat/completions", json=payload)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text) from exc
    except httpx.TimeoutException as exc:
        raise HTTPException(status_code=504, detail="upstream request timed out") from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail="upstream request failed") from exc


async def stream_chat_completion(
    *,
    base_url: str,
    payload: dict[str, Any],
    timeout_seconds: int,
) -> StreamingResponse:
    async def event_stream() -> AsyncIterator[bytes]:
        try:
            async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                async with client.stream(
                    "POST",
                    f"{base_url}/v1/chat/completions",
                    json=payload,
                ) as response:
                    response.raise_for_status()
                    async for chunk in response.aiter_bytes():
                        yield chunk
        except httpx.HTTPError as exc:
            yield f'data: {{"error":"{str(exc)}"}}\n\n'.encode()

    return StreamingResponse(event_stream(), media_type="text/event-stream")
