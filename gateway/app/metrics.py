from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.responses import Response

REQUESTS_TOTAL = Counter(
    "llm_gateway_requests_total",
    "Total requests handled by the LLM gateway.",
    ["route", "status"],
)

REQUEST_LATENCY_SECONDS = Histogram(
    "llm_gateway_request_latency_seconds",
    "Gateway request latency in seconds.",
    ["route"],
)


def metrics_response() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
