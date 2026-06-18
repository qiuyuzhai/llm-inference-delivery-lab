import pytest
from fastapi.testclient import TestClient

from gateway.app.main import app
from gateway.app.upstream import build_chat_payload


def test_build_chat_payload_uses_default_model_when_missing():
    payload = build_chat_payload(
        body={
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
        },
        default_model="Qwen/Qwen2.5-14B-Instruct",
    )

    assert payload["model"] == "Qwen/Qwen2.5-14B-Instruct"
    assert payload["messages"] == [{"role": "user", "content": "hello"}]
    assert payload["stream"] is False


def test_gateway_requires_api_key():
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hello"}]},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "missing or invalid API key"


@pytest.mark.asyncio
async def test_build_chat_payload_rejects_empty_messages():
    with pytest.raises(ValueError, match="messages must not be empty"):
        build_chat_payload(body={"messages": []}, default_model="model")
