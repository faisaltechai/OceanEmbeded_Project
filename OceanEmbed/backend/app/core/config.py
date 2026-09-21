import os
from functools import lru_cache
from pydantic import BaseModel


class Settings(BaseModel):
    bundle_path: str = os.getenv(
        "OCEANEMBED_BUNDLE_PATH",
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "demo", "backend_bundle.json"),
    )
    cors_allow_origins: list[str] = os.getenv("OCEANEMBED_CORS_ORIGINS", "http://localhost:5173").split(",")
    environment: str = os.getenv("OCEANEMBED_ENV", "development")  # development | demo | production
    enable_scheduler: bool = os.getenv("OCEANEMBED_ENABLE_SCHEDULER", "true").lower() == "true"


@lru_cache
def get_settings() -> Settings:
    return Settings()
