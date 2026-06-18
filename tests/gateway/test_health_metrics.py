from fastapi.testclient import TestClient

from gateway.app.main import app


def test_health_returns_ok():
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_metrics_endpoint_exposes_prometheus_text():
    client = TestClient(app)

    response = client.get("/metrics")

    assert response.status_code == 200
    assert "llm_gateway_requests_total" in response.text
