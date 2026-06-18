from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    vllm_base_url: str = "http://localhost:8000"
    gateway_host: str = "0.0.0.0"
    gateway_port: int = 8080
    gateway_api_key: str = "change-me"
    request_timeout_seconds: int = 120
    stream_timeout_seconds: int = 300
    log_level: str = "INFO"
    default_model: str = "Qwen/Qwen2.5-14B-Instruct"

    @field_validator("vllm_base_url")
    @classmethod
    def strip_trailing_slash(cls, value: str) -> str:
        return value.rstrip("/")


def get_settings() -> Settings:
    return Settings()
