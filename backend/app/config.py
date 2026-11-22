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
    # Offer a "Demo" choice at login that uses the built-in mock DRES (no real submissions).
    dres_demo_enabled: bool = True

    # Index / data
    index_path: Path = REPO_ROOT / "data" / "index.faiss"
    ids_path: Path = REPO_ROOT / "data" / "ids.json"
    images_dir: Path = REPO_ROOT / "data" / "images"
    thumbs_dir: Path = REPO_ROOT / "data" / "thumbs"
    metadata_path: Path = REPO_ROOT / "data" / "metadata.csv"

    region_index_dir: Path = REPO_ROOT / "data" / "regions"
    search_detections_path: Path = REPO_ROOT / "data" / "search-detections.json"
    detection_model_dir: Path = REPO_ROOT / ".cache" / "huggingface" / "hub" / "models--IDEA-Research--grounding-dino-tiny" / "snapshots" / "a2bb814dd30d776dcf7e30523b00659f4f141c71"
    # Grounding DINO for count/absence conditions in image search (slow on CPU). QA always uses it.
    search_detection_enabled: bool = False
    detection_max_images: int = Field(default=50, ge=1, le=500)
    detection_timeout_s: float = Field(default=45, gt=0, le=300)
    # Torch threads used only while Grounding DINO runs; 0 = physical cores (max 8).
    detection_threads: int = Field(default=0, ge=0, le=64)

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
    search_plan_api_key_backup: str = ""
    search_plan_model: str = "gemini-3.5-flash-lite"
    search_plan_timeout_s: float = Field(default=20.0, gt=0)
    search_plan_cache_size: int = Field(default=128, ge=1)
    # Vision check of the top results after a search (uses the QA Gemini client).
    # Images checked per request: 4 keeps a search at 1 check call + at most 1 close-up call.
    search_verify_images: int = Field(default=4, ge=1, le=16)
    search_verify_timeout_s: float = Field(default=40.0, gt=0, le=120)

    # Question Answering (QA)
    qa_local_mode: bool = False
    qa_enabled: bool = False
    qa_retrieval_enabled: bool = False
    qa_concurrency: int = 1
    qa_queue_limit: int = 8
    qa_queue_timeout_s: float = 3.0
    qa_request_timeout_s: float = 60.0
    qa_auto_timeout_s: float = Field(default=180.0, gt=0, le=300)
    # Stop reading retrieved images once the evidence already settles the answer.
    qa_early_stop: bool = True
    # rules: per-question-type evidence rules. stable_answer: after each batch ask for a short draft answer
    # and stop when two consecutive batches give the same one (reads 2 images per batch, one extra call each).
    qa_early_stop_mode: Literal["rules", "stable_answer"] = "rules"
    qa_sequence_path: Path = REPO_ROOT / "data" / "qa_sequences.json"
    qa_people_catalog_path: Path = REPO_ROOT / "data" / "qa_people_reviewed.json"
    qa_rate_limit_per_min: int = 6
    qa_retrieval_k: int = 10
    qa_max_images: int = 5
    qa_corpus_max_images: int = 128
    # Calibrated with tools/calibrate_distinct_threshold.py: 0.85 over-counted by ~18%, 0.76 by ~6%.
    qa_distinct_similarity_threshold: float = Field(default=0.76, ge=-1.0, le=1.0)
    qa_distinct_top_k: int = Field(default=10, ge=1, le=100)
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
    # Optional second Gemini key (same model) used when the first is rate limited or overloaded.
    qa_api_key_backup: str = ""
    qa_model: str = ""
    # Independent local settings let QA_PROVIDER switch without changing Gemini credentials.
    qa_ollama_base_url: str = "http://127.0.0.1:11434/v1"
    qa_ollama_model: str = "qwen2.5vl:7b"
    qa_prompt_version: str = "qa-v1"
    qa_policy_version: str = "policy-v1"

    @property
    def using_default_secret(self) -> bool:
        return self.session_secret == "dev-only-insecure-secret-change-me"


@lru_cache
def get_settings() -> Settings:
    return Settings()
