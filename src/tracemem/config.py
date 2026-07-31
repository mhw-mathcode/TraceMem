from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AblationProfile(str, Enum):
    BASELINE = "baseline"
    CARDS = "cards"
    TEMPORAL = "temporal"
    ADAPTIVE = "adaptive"
    FULL = "full"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="TRACEMEM_",
        env_file=".env",
        extra="ignore",
        protected_namespaces=("settings_",),
    )

    api_key: str = ""
    database_path: Path = Path("data/tracemem.db")
    profile: AblationProfile = AblationProfile.FULL
    embedding_mode: Literal["hash", "openai"] = "hash"
    extraction_mode: Literal["disabled", "openai"] = "disabled"
    rerank_mode: Literal["disabled", "openai", "dashscope"] = "disabled"
    embedding_url: str = (
        "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"
    )
    embedding_api_key: str = ""
    embedding_extra_key: str = ""
    embedding_model: str = "text-embedding-v4"
    llm_url: str = (
        "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    )
    llm_api_key: str = ""
    llm_model: str = "qwen-plus"
    rerank_url: str = (
        "https://dashscope.aliyuncs.com/api/v1/services/"
        "rerank/text-rerank/text-rerank"
    )
    rerank_model: str = "qwen3-rerank"
    model_timeout_seconds: float = Field(default=45.0, gt=0)
    model_max_retries: int = Field(default=2, ge=0, le=10)
    embedding_batch_size: int = Field(default=10, ge=1, le=100)
    embedding_max_concurrency: int = Field(default=3, ge=1, le=100)
    embedding_dimensions: int = Field(default=256, ge=8)
    add_lease_seconds: int = Field(default=120, ge=1)
    final_result_limit: int = Field(default=30, ge=1, le=90)

    def require_api_key(self) -> None:
        if not self.api_key.strip():
            raise ValueError(
                "TRACEMEM_API_KEY is required to start the HTTP service"
            )

    def require_remote_credentials(self) -> None:
        if (
            self.embedding_mode == "openai"
            and not self.embedding_api_key.strip()
        ):
            raise ValueError(
                "TRACEMEM_EMBEDDING_API_KEY is required when "
                "TRACEMEM_EMBEDDING_MODE=openai"
            )
        if (
            self.extraction_mode == "openai"
            or self.rerank_mode in {"openai", "dashscope"}
        ) and not self.llm_api_key.strip():
            raise ValueError(
                "TRACEMEM_LLM_API_KEY is required for remote LLM "
                "or Rerank modes"
            )

    @property
    def cards_enabled(self) -> bool:
        return self.profile is not AblationProfile.BASELINE

    @property
    def temporal_enabled(self) -> bool:
        return self.profile in {
            AblationProfile.TEMPORAL,
            AblationProfile.ADAPTIVE,
            AblationProfile.FULL,
        }

    @property
    def adaptive_enabled(self) -> bool:
        return self.profile in {
            AblationProfile.ADAPTIVE,
            AblationProfile.FULL,
        }

    @property
    def evidence_pairing_enabled(self) -> bool:
        return self.profile is AblationProfile.FULL
