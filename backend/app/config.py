from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # DRES
    dres_base_url: str = "http://dres:8080"
    dres_evaluation_id: str | None = None
    dres_timeout_s: float = 5.0

    # Index / data
    index_path: Path = REPO_ROOT / "data" / "index.faiss"
    ids_path: Path = REPO_ROOT / "data" / "ids.json"
    images_dir: Path = REPO_ROOT / "data" / "images"
    thumbs_dir: Path = REPO_ROOT / "data" / "thumbs"
    metadata_path: Path = REPO_ROOT / "data" / "metadata.csv"

    region_index_dir: Path = REPO_ROOT / "data" / "regions"
    search_detections_path: Path = REPO_ROOT / "data" / "search-detections.json"

    # Embedding model (must match tools/build_index.py)
    embed_model_name: str = "ViT-B-32"
    embed_pretrained: str = "laion2b_s34b_b79k"
    encode_concurrency: int = 4

    # Session / security
    session_secret: str = "dev-only-insecure-secret-change-me"
    session_cookie_name: str = "retrieval_session"
    session_max_age_s: int = 60 * 60 * 8
    # Must be True in production (served over HTTPS via Caddy). Set False
    # only for local dev over plain http://127.0.0.1, otherwise browsers
    # silently drop the cookie.
    session_cookie_secure: bool = True

    # Request limits
    max_k: int = 100
    default_k: int = 50
    query_max_len: int = 1200
    rate_limit_per_min: int = 30
    search_cache_size: int = 256
    search_candidate_k: int = Field(default=100, ge=1, le=500)
    search_region_threshold: float = Field(default=0.20, ge=-1.0, le=1.0)
    search_region_weight: float = Field(default=0.25, ge=0.0, le=1.0)
    search_plan_api_key: str = ""
    search_plan_model: str = "gemini-3.5-flash-lite"
    search_plan_timeout_s: float = Field(default=10.0, gt=0)
    search_plan_cache_size: int = Field(default=128, ge=1)

    # Question Answering (QA)
    qa_local_mode: bool = False
    qa_enabled: bool = False
    qa_retrieval_enabled: bool = False
    qa_concurrency: int = 1
    qa_queue_limit: int = 8
    qa_queue_timeout_s: float = 3.0
    qa_request_timeout_s: float = 60.0
    qa_sequence_path: Path = REPO_ROOT / "data" / "qa_sequences.json"
    qa_people_catalog_path: Path = REPO_ROOT / "data" / "qa_people_reviewed.json"
    qa_rate_limit_per_min: int = 6
    qa_retrieval_k: int = 10
    qa_max_images: int = 5
    qa_corpus_max_images: int = 128
    qa_retrieval_strategy: Literal["verified_early_stop", "consensus"] = "verified_early_stop"
    qa_confirm_candidates: int = 1
    qa_cache_size: int = 128
    qa_cache_ttl_s: float = 600.0
    qa_max_output_tokens: int = 256
    qa_question_max_len: int = 1200
    qa_image_max_side: int = 1024
    qa_max_crops: int = 2
    qa_provider: str = "openai_compatible"
    qa_base_url: str = ""
    qa_api_key: str = ""
    qa_model: str = ""
    qa_prompt_version: str = "qa-v1"
    qa_policy_version: str = "policy-v1"

    @property
    def using_default_secret(self) -> bool:
        return self.session_secret == "dev-only-insecure-secret-change-me"


@lru_cache
def get_settings() -> Settings:
    return Settings()
