from __future__ import annotations

import math
import unicodedata
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .config import get_settings

_settings = get_settings()

QAMode = Literal["selected_image", "retrieval_qa", "auto"]
QALanguage = Literal["vi", "en"]
QAStatus = Literal["answered", "insufficient_evidence", "ambiguous"]
QAEVidenceSource = Literal["image", "region", "metadata"]


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=128)
    password: str = Field(min_length=1, max_length=256)
    # "real": DRES_BASE_URL (competition); "demo": built-in mock DRES (student / student).
    mode: Literal["real", "demo"] = "real"


class LoginResponse(BaseModel):
    username: str
    role: str
    mode: str = "real"


SpatialCondition = Literal["left_of", "right_of", "above", "below", "near"]
ConditionKind = Literal["absent", "exactly", "at_least", "at_most", "action", "other", SpatialCondition]
SPATIAL_CONDITIONS = frozenset({"left_of", "right_of", "above", "below", "near"})
COUNT_CONDITIONS = frozenset({"exactly", "at_least", "at_most"})


class SearchCondition(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid", frozen=True)
    kind: ConditionKind
    subject: str = Field(min_length=1, max_length=220)
    target: str | None = Field(default=None, min_length=1, max_length=220)
    count: int | None = Field(default=None, ge=0, le=99)

    @model_validator(mode="after")
    def check_operands(self):
        if (self.target is not None) != (self.kind in SPATIAL_CONDITIONS):
            raise ValueError("A spatial condition needs two objects")
        if (self.count is not None) != (self.kind in COUNT_CONDITIONS):
            raise ValueError("A count condition needs a number")
        return self


class SearchRequest(BaseModel):
    query: str = Field(default="", max_length=_settings.query_max_len)
    detail_queries: list[str] = Field(default_factory=list, max_length=5)
    conditions: list[SearchCondition] = Field(default_factory=list, max_length=5)
    auto_split: bool = False
    mode: Literal["legacy", "clip", "clip_region"] = "legacy"
    k: int = Field(default=_settings.default_k, ge=1, le=max(500, _settings.max_k))

    @model_validator(mode='after')
    def clean_queries(self):
        self.query = self.query.strip()
        self.detail_queries = [q.strip() for q in self.detail_queries if q.strip()]
        if not self.query and not self.detail_queries:
            raise ValueError('Enter at least one description')
        if any(len(q) > _settings.query_max_len for q in self.detail_queries):
            raise ValueError('Detail query too long')
        if self.mode == "legacy" and self.k > _settings.max_k:
            raise ValueError('Too many results')
        return self


class SearchVerifyRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    query: str = Field(min_length=1, max_length=_settings.query_max_len)
    detail_queries: list[str] = Field(default_factory=list, max_length=5)
    conditions: list[SearchCondition] = Field(default_factory=list, max_length=5)
    image_ids: list[str] = Field(min_length=1, max_length=50)
    language: Literal["vi", "en"] = "vi"


class SearchPlanRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    query: str = Field(min_length=1, max_length=_settings.query_max_len)
    language: Literal["auto", "vi", "en"] = "auto"


class SearchPlan(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    global_query_en: str = Field(min_length=1, max_length=_settings.query_max_len)
    detail_queries_en: list[str] = Field(max_length=5)
    conditions: list[SearchCondition] = Field(default_factory=list, max_length=5)

    @field_validator("detail_queries_en")
    @classmethod
    def validate_phrases(cls, phrases):
        if any(not phrase or len(phrase) > _settings.query_max_len for phrase in phrases):
            raise ValueError("Invalid search phrase")
        return list(dict.fromkeys(phrases))

    @field_validator("conditions")
    @classmethod
    def unique_conditions(cls, conditions):
        return list(dict.fromkeys(conditions))


class ParseQueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=_settings.query_max_len)


class DetectedObject(BaseModel):
    label: str
    score: float
    box: list[float]


class RegionMatch(BaseModel):
    query: str
    region_id: str | None = None
    score: float | None
    source: str
    box: list[float] | None = None
    kind: str | None = None
    objects: list[DetectedObject] = Field(default_factory=list)
    model: str | None = None
    supported: bool | None = None
    source_view: int | None = None
    source_window: list[float] | None = None


class SearchConditionResult(BaseModel):
    kind: Literal["present", ConditionKind]
    subject: str
    target: str | None = None
    count: int | None = None
    observed_count: int | None = None
    status: Literal["met", "violated", "unknown"]
    reason: str
    evidence: list[RegionMatch] = Field(default_factory=list)


class SearchResultItem(BaseModel):
    id: str
    score: float
    thumb: str
    matches: list[RegionMatch] = Field(default_factory=list)
    conditions: list[SearchConditionResult] = Field(default_factory=list)
    ranking: dict | None = None
    clip_rank: int | None = None
    final_rank: int | None = None
    region_evaluated: bool | None = None


class SearchResponse(BaseModel):
    results: list[SearchResultItem]
    took_ms: int
    detail_queries: list[str] = Field(default_factory=list)
    region_ready: bool = False
    notice: str | None = None
    mode: Literal["legacy", "clip", "clip_region"] = "legacy"
    candidate_k: int | None = None
    region_threshold: float | None = None
    index_version: str | None = None


class SubmitRequest(BaseModel):
    id: str = Field(min_length=1, max_length=256)


class SubmitTextRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    text: str = Field(min_length=1, max_length=2000)


class SubmitResponse(BaseModel):
    verdict: str
    message: str


class ImageMetadataResponse(BaseModel):
    owner: str
    day: str
    hour: str
    frame: str
    classes: list[str]
    class_count: dict[str, int]
    description: str


class TaskResponse(BaseModel):
    running: bool
    name: str | None = None
    task_group: str | None = None
    task_type: str | None = None
    duration_s: int | None = None
    server_time_ms: int | None = None
    # Evaluation this session submits to (chosen Run ID, DRES_EVALUATION_ID, or the demo evaluation).
    evaluation_id: str | None = None


class ErrorResponse(BaseModel):
    code: str
    message: str


class QAScope(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sequence_id: str | None = Field(default=None, max_length=200)
    start_order: int | None = Field(default=None, ge=0)
    end_order: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def valid_range(self):
        if self.start_order is not None and self.end_order is not None and self.start_order > self.end_order:
            raise ValueError("Phạm vi thứ tự ảnh không hợp lệ")
        return self


class QARequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: QAMode = "auto"
    conversation_id: str | None = Field(default=None, pattern=r"^[a-zA-Z0-9_-]{1,80}$")
    scope: QAScope | None = None
    question: str
    image_id: str | None = None
    retrieval_query: str | None = None
    answer_language: QALanguage = "vi"

    @model_validator(mode="after")
    def validate_qa_request(self):
        max_len = _settings.qa_question_max_len
        self.question = unicodedata.normalize("NFC", self.question).strip()
        if not self.question:
            raise ValueError("Câu hỏi không được để trống")
        if len(self.question) > max_len:
            raise ValueError(f"Câu hỏi vượt quá {max_len} ký tự")

        if self.retrieval_query is not None:
            cleaned_rq = unicodedata.normalize("NFC", self.retrieval_query).strip()
            if len(cleaned_rq) > max_len:
                raise ValueError(f"Mô tả tìm kiếm vượt quá {max_len} ký tự")
            self.retrieval_query = cleaned_rq or None

        if self.mode == "selected_image":
            if self.retrieval_query is not None:
                raise ValueError("Chế độ selected_image không nhận retrieval_query")
            if self.image_id is None:
                raise ValueError("Chế độ selected_image bắt buộc có image_id")
            self.image_id = self.image_id.strip()
            if not self.image_id or len(self.image_id) > 256:
                raise ValueError("image_id không hợp lệ")
            if any(token in self.image_id for token in ("/", "\\", "..", ":")):
                raise ValueError("image_id không được chứa đường dẫn hoặc URL")
        elif self.mode in {"retrieval_qa", "auto"}:
            if self.image_id is not None:
                raise ValueError("Chế độ retrieval_qa không nhận image_id")

        return self


def validate_qa_box(box: list[float]) -> list[float]:
    if len(box) != 4 or not all(math.isfinite(value) for value in box):
        raise ValueError("box phải gồm 4 số hữu hạn [x1, y1, x2, y2]")
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise ValueError("box phải chuẩn hóa trong khoảng [0, 1] với x1 < x2 và y1 < y2")
    return box


class QARegion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: str = Field(min_length=1, max_length=120)
    box: list[float]

    _validate_box = field_validator("box")(validate_qa_box)


class QAEvidenceItem(BaseModel):
    image_id: str
    caption: str | None = None
    observation: str | None = None
    verification_status: Literal['verified', 'unverified'] | None = None
    image_url: str | None = None
    thumbnail_url: str | None = None
    timestamp: str | None = None
    box: list[float] | None = None
    regions: list[QARegion] = Field(default_factory=list)
    unverified_proposals: list[QARegion] = Field(default_factory=list)
    source: QAEVidenceSource = "image"

    @field_validator("box")
    @classmethod
    def validate_box(cls, v: list[float] | None) -> list[float] | None:
        return [round(x, 6) for x in validate_qa_box(v)] if v is not None else None


class QAAlternativeItem(BaseModel):
    answer: str
    evidence: list[QAEvidenceItem] = Field(default_factory=list)
    clarification: str | None = None


class QAVersions(BaseModel):
    model: str
    prompt: str
    corpus: str
    region: str | None = None


class QAClaim(BaseModel):
    text: str
    evidence_ids: list[str] = Field(default_factory=list)


class QAResponse(BaseModel):
    request_id: str
    conversation_id: str | None = None
    question_type: str | None = None
    operation: str | None = None
    examined_scope: dict = Field(default_factory=dict)
    claims: list[QAClaim] = Field(default_factory=list)
    identity_match_candidates: list[dict] = Field(default_factory=list)
    distinct_estimate: dict | None = None
    rounds: int = 0
    api_calls: int = 0
    status: QAStatus
    answer: str | None = None
    # Bare answer to submit to DRES ('4 người', 'màu nâu'); None when the answer has no short form.
    short_answer: str | None = None
    # Distinct people found by the roster: label, appearance, the frame and box where each one is shown.
    people_roster: list[dict] = Field(default_factory=list)
    reason_code: str | None = None
    message: str | None = None
    evidence: list[QAEvidenceItem] = Field(default_factory=list)
    alternatives: list[QAAlternativeItem] = Field(default_factory=list)
    note: str | None = None
    note_evidence: list[QAEvidenceItem] = Field(default_factory=list)
    partial: bool = False
    searched_count: int = 0
    examined_count: int = 0
    took_ms: int
    cache_hit: bool = False
    versions: QAVersions


class QAStatusResponse(BaseModel):
    enabled: bool
    ready: bool
    supported_modes: list[QAMode]
    max_question_length: int
