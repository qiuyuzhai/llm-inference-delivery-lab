import time
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException

from gateway.app.config import Settings, get_settings
from gateway.app.metrics import REQUEST_LATENCY_SECONDS, REQUESTS_TOTAL, metrics_response
from gateway.app.upstream import build_chat_payload, post_chat_completion, stream_chat_completion

app = FastAPI(title="LLM Inference Gateway")

SettingsDependency = Annotated[Settings, Depends(get_settings)]
AuthorizationHeader = Annotated[str | None, Header()]


@app.get("/health")
def health() -> dict[str, str]:
    REQUESTS_TOTAL.labels(route="/health", status="ok").inc()
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return metrics_response()


def require_api_key(
    settings: SettingsDependency,
    authorization: AuthorizationHeader = None,
) -> None:
    expected = f"Bearer {settings.gateway_api_key}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="missing or invalid API key")


@app.post("/v1/chat/completions")
async def chat_completions(
    body: dict[str, Any],
    _: Annotated[None, Depends(require_api_key)],
    settings: SettingsDependency,
):
    started = time.perf_counter()
    route = "/v1/chat/completions"
    try:
        payload = build_chat_payload(body, default_model=settings.default_model)
        if payload["stream"]:
            REQUESTS_TOTAL.labels(route=route, status="stream").inc()
            return await stream_chat_completion(
                base_url=settings.vllm_base_url,
                payload=payload,
                timeout_seconds=settings.stream_timeout_seconds,
            )
        result = await post_chat_completion(
            base_url=settings.vllm_base_url,
            payload=payload,
            timeout_seconds=settings.request_timeout_seconds,
        )
        REQUESTS_TOTAL.labels(route=route, status="ok").inc()
        return result
    except ValueError as exc:
        REQUESTS_TOTAL.labels(route=route, status="bad_request").inc()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        REQUEST_LATENCY_SECONDS.labels(route=route).observe(time.perf_counter() - started)
