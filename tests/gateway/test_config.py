from gateway.app.config import Settings


def test_settings_reads_gateway_api_key():
    settings = Settings(
        vllm_base_url="http://vllm:8000",
        gateway_api_key="secret",
        request_timeout_seconds=30,
        stream_timeout_seconds=60,
    )

    assert settings.vllm_base_url == "http://vllm:8000"
    assert settings.gateway_api_key == "secret"
    assert settings.request_timeout_seconds == 30
    assert settings.stream_timeout_seconds == 60


def test_settings_strips_trailing_slash_from_vllm_url():
    settings = Settings(vllm_base_url="http://vllm:8000/")

    assert settings.vllm_base_url == "http://vllm:8000"
