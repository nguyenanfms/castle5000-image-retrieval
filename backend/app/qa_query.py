"""Query planner for retrieval_qa: extracts visual search context from questions without guessing answers."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from .query_parser import split_query

# Interrogative patterns to strip while preserving the subject/scene context.
_VI_INTERROGATIVE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bđang\s+cầm\s+(?:cái\s+|vật\s+)?gì\b", re.I), ""),
    (re.compile(r"\bcầm\s+(?:cái\s+|vật\s+)?gì\b", re.I), ""),
    (re.compile(r"\bđang\s+làm\s+gì\b", re.I), ""),
    (re.compile(r"\blàm\s+gì\b", re.I), ""),
    (re.compile(r"\bcó\s+màu\s+gì\b", re.I), ""),
    (re.compile(r"\bmàu\s+gì\b", re.I), ""),
    (re.compile(r"\blà\s+(?:cái\s+|vật\s+|ai\s+)?gì\b", re.I), ""),
    (re.compile(r"\bở\s+đâu\b", re.I), ""),
    (re.compile(r"\bnằm\s+ở\s+đâu\b", re.I), ""),
    (re.compile(r"\bcó\s+bao\s+nhiêu\b", re.I), ""),
    (re.compile(r"\bbao\s+nhiêu\b", re.I), ""),
    (re.compile(r"\bcó\s+mấy\b", re.I), ""),
    (re.compile(r"\bmấy\b", re.I), ""),
    (re.compile(r"\bnhư\s+thế\s+nào\b", re.I), ""),
    (re.compile(r"\bnào\b(?=\s*[?.,!]*$)", re.I), ""),
    (re.compile(r"\bgì\b(?=\s*[?.,!]*$)", re.I), ""),
]

_EN_INTERROGATIVE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # "what is the man wearing a red shirt holding?" -> "the man wearing a red shirt"
    (re.compile(r"^\s*what\s+(?:is|are|was|were)\s+(.+?)\s+(?:holding|carrying|doing|wearing|looking\s+at|using|eating|drinking)\s*[?.!]*$", re.I), r"\1"),
    # "what color is the shirt of the man in the kitchen?" -> "the shirt of the man in the kitchen"
    (re.compile(r"^\s*what\s+(?:color|colour|kind|type)\s+(?:is|are|was|were)\s+(.+?)\s*[?.!]*$", re.I), r"\1"),
    # "where is the red cup in the kitchen?" -> "the red cup in the kitchen"
    (re.compile(r"^\s*where\s+(?:is|are|was|were)\s+(.+?)\s*[?.!]*$", re.I), r"\1"),
    # "how many cups are on the table in the kitchen?" -> "cups on the table in the kitchen"
    (re.compile(r"^\s*how\s+many\s+(.+?)\s+(?:are\s+there|are|is|do\s+you\s+see)\b", re.I), r"\1 "),
    (re.compile(r"^\s*how\s+many\s+", re.I), ""),
    (re.compile(r"^\s*(?:what|who|which|where|when|how)\s+(?:is|are|was|were|does|do|did)\s+", re.I), ""),
    (re.compile(r"\b(?:holding|carrying|doing)\s+what\b", re.I), ""),
    (re.compile(r"\bwhat\s+(?:color|colour)\b", re.I), ""),
]

# Boilerplate framing phrases that do not identify a specific image.
_BOILERPLATE_RE = re.compile(
    r"\b(?:"
    r"trong\s+ảnh\s+này|trong\s+bức\s+ảnh\s+này|trong\s+ảnh|ở\s+trong\s+ảnh|bức\s+ảnh|hình\s+này|"
    r"in\s+this\s+image|in\s+the\s+image|in\s+this\s+photo|in\s+the\s+photo|in\s+this\s+picture|in\s+the\s+picture|"
    r"cho\s+biết|hãy\s+cho\s+biết|tell\s+me"
    r")\b",
    re.I,
)

# Generic nouns/pronouns that alone are insufficient to search across 5,000 images.
_GENERIC_ONLY_TOKENS: set[str] = {
    "a", "an", "the", "this", "that", "these", "those", "there", "here",
    "is", "are", "was", "were", "be", "been", "being", "has", "have", "had",
    "of", "in", "on", "at", "by", "with", "from", "to", "for",
    "person", "people", "man", "woman", "someone", "somebody", "he", "she", "it", "they",
    "object", "thing", "item", "items", "objects", "things",
    "người", "người_đó", "người_này", "anh_ấy", "cô_ấy", "họ", "nó", "vật", "đồ_vật", "thứ",
    "có", "đang", "của", "và", "với", "trên", "dưới", "cạnh", "bên", "gần",
}

# Ordered phrase translation table (Vietnamese -> English) for CLIP text encoder.
# Preserves negations ("không", "without", "not") and attributes (colors, clothing, locations, objects).
_VI_TO_EN_PHRASES: list[tuple[str, str]] = [
    ("không đội mũ", "not wearing a hat"),
    ("không đội nón", "not wearing a hat"),
    ("không đeo kính", "not wearing glasses"),
    ("không đeo khẩu trang", "not wearing a mask"),
    ("không mặc", "not wearing"),
    ("không đội", "not wearing"),
    ("không đeo", "not wearing"),
    ("không cầm", "not holding"),
    ("không có", "without"),
    ("không", "not"),
    ("người đàn ông", "a man"),
    ("người phụ nữ", "a woman"),
    ("cô gái", "a woman"),
    ("chàng trai", "a man"),
    ("nhiều người", "several people"),
    ("hai người", "two people"),
    ("ba người", "three people"),
    ("một người", "a person"),
    ("người", "a person"),
    ("mặc áo khoác", "wearing a jacket"),
    ("mặc áo sơ mi", "wearing a shirt"),
    ("mặc áo thun", "wearing a t-shirt"),
    ("mặc áo phông", "wearing a t-shirt"),
    ("mặc áo len", "wearing a sweater"),
    ("mặc áo", "wearing a shirt"),
    ("mặc quần", "wearing pants"),
    ("áo khoác", "jacket"),
    ("áo sơ mi", "shirt"),
    ("áo thun", "t-shirt"),
    ("áo phông", "t-shirt"),
    ("áo len", "sweater"),
    ("áo", "shirt"),
    ("quần", "pants"),
    ("đội mũ", "wearing a hat"),
    ("đội nón", "wearing a hat"),
    ("đeo kính", "wearing glasses"),
    ("đeo khẩu trang", "wearing a mask"),
    ("đeo ba lô", "wearing a backpack"),
    ("đeo túi", "carrying a bag"),
    ("mũ", "a hat"),
    ("nón", "a hat"),
    ("kính", "glasses"),
    ("khẩu trang", "a mask"),
    ("màu đỏ", "red"),
    ("màu xanh lá", "green"),
    ("màu xanh dương", "blue"),
    ("màu xanh", "blue"),
    ("màu vàng", "yellow"),
    ("màu trắng", "white"),
    ("màu đen", "black"),
    ("màu xám", "gray"),
    ("màu hồng", "pink"),
    ("màu tím", "purple"),
    ("màu cam", "orange"),
    ("màu nâu", "brown"),
    ("đỏ", "red"),
    ("xanh lá", "green"),
    ("xanh dương", "blue"),
    ("trắng", "white"),
    ("đen", "black"),
    ("vàng", "yellow"),
    ("xám", "gray"),
    ("hồng", "pink"),
    ("tím", "purple"),
    ("nâu", "brown"),
    ("nhà bếp", "in the kitchen"),
    ("phòng bếp", "in the kitchen"),
    ("bếp", "kitchen"),
    ("phòng khách", "in the living room"),
    ("phòng ngủ", "in the bedroom"),
    ("phòng làm việc", "in the office"),
    ("văn phòng", "in the office"),
    ("hành lang", "in the hallway"),
    ("cầu thang", "stairs"),
    ("ngoài trời", "outdoors"),
    ("cửa hàng", "in a store"),
    ("siêu thị", "in a supermarket"),
    ("nhà hàng", "in a restaurant"),
    ("quán cà phê", "in a cafe"),
    ("trên bàn", "on the table"),
    ("bàn ăn", "dining table"),
    ("bàn làm việc", "desk"),
    ("cái bàn", "table"),
    ("bàn", "table"),
    ("cái ghế", "chair"),
    ("ghế sofa", "sofa"),
    ("ghế", "chair"),
    ("tủ lạnh", "refrigerator"),
    ("lò vi sóng", "microwave"),
    ("bồn rửa", "sink"),
    ("máy tính xách tay", "laptop"),
    ("máy tính", "computer"),
    ("màn hình", "monitor"),
    ("điện thoại", "phone"),
    ("cốc", "cup"),
    ("ly", "cup"),
    ("chai nước", "bottle"),
    ("chai", "bottle"),
    ("bát", "bowl"),
    ("đĩa", "plate"),
    ("dao", "knife"),
    ("kéo", "scissors"),
    ("sách", "book"),
    ("túi", "bag"),
    ("ba lô", "backpack"),
    ("đang ngồi", "sitting"),
    ("ngồi", "sitting"),
    ("đang đứng", "standing"),
    ("đứng", "standing"),
    ("đang đi bộ", "walking"),
    ("đi bộ", "walking"),
    ("đang nấu ăn", "cooking"),
    ("nấu ăn", "cooking"),
    ("đang rửa", "washing"),
    ("đang đọc", "reading"),
    ("đang nhìn", "looking at"),
    ("đang sử dụng", "using"),
    ("đang dùng", "using"),
    ("đang uống", "drinking"),
    ("đang ăn", "eating"),
]


def _contains_vietnamese(text: str) -> bool:
    return bool(
        re.search(
            r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]",
            text,
            re.I,
        )
    )


def _translate_vi_context_to_en(text: str) -> str:
    """Deterministic phrase-level Vietnamese-to-English conversion for CLIP retrieval."""
    working = " " + text.strip() + " "
    # Handle Vietnamese post-nominal color adjectives: "shirt red" -> "red shirt"
    for vi_phrase, en_phrase in _VI_TO_EN_PHRASES:
        pattern = re.compile(r"(?<!\w)" + re.escape(vi_phrase) + r"(?!\w)", re.I)
        working = pattern.sub(en_phrase, working)
    working = re.sub(r"\s+", " ", working).strip(" ,;.!?")
    # Reorder common noun + color patterns produced from Vietnamese word order
    working = re.sub(
        r"\b(shirt|t-shirt|jacket|sweater|pants|hat|bag|backpack|cup|bottle|bowl|plate|chair|table|car)\s+"
        r"(red|blue|green|yellow|white|black|gray|pink|purple|orange|brown)\b",
        r"\2 \1",
        working,
        flags=re.I,
    )
    return working


def _has_specific_visual_context(cleaned_context: str) -> bool:
    """Return True only if cleaned_context contains at least one specific visual attribute/object/scene token."""
    if not cleaned_context:
        return False
    tokens = re.findall(r"[\w]+", cleaned_context.casefold())
    meaningful = [t for t in tokens if t not in _GENERIC_ONLY_TOKENS and len(t) > 1]
    return len(meaningful) >= 1


@dataclass(frozen=True)
class QueryPlan:
    original_question: str
    search_query: str
    detail_queries: list[str] = field(default_factory=list)
    sufficient_context: bool = True
    method: str = "user_retrieval_query"


class QAQueryPlanner:
    """Prepares retrieval search queries from a user question and optional retrieval description."""

    def plan(self, question: str, retrieval_query: str | None = None) -> QueryPlan:
        orig_q = unicodedata.normalize("NFC", question).strip()

        if retrieval_query is not None and retrieval_query.strip():
            rq = unicodedata.normalize("NFC", retrieval_query).strip()
            rq_no_boiler = _BOILERPLATE_RE.sub(" ", rq)
            rq_no_boiler = re.sub(r"\s+", " ", rq_no_boiler).strip(" ,;.!?")
            if not _has_specific_visual_context(rq_no_boiler):
                return QueryPlan(
                    original_question=orig_q,
                    search_query="",
                    detail_queries=[],
                    sufficient_context=False,
                    method="insufficient_user_query",
                )
            en_query = _translate_vi_context_to_en(rq_no_boiler) if _contains_vietnamese(rq_no_boiler) else rq_no_boiler
            details = split_query(en_query)
            return QueryPlan(
                original_question=orig_q,
                search_query=en_query,
                detail_queries=details if len(details) > 1 else [],
                sufficient_context=True,
                method="user_retrieval_query",
            )

        # Extract context from the question itself without guessing the target answer
        working = _BOILERPLATE_RE.sub(" ", orig_q)
        for pat, repl in _EN_INTERROGATIVE_PATTERNS:
            working = pat.sub(repl, working)
        for pat, repl in _VI_INTERROGATIVE_PATTERNS:
            working = pat.sub(repl, working)

        # Remove trailing/leading punctuation and leftover connectors
        working = re.sub(r"[?？！!]+", " ", working)
        working = re.sub(r"\b(?:người\s+đó|người\s+này|anh\s+ấy|cô\s+ấy|vật\s+đó|vật\s+này)\b", " ", working, flags=re.I)
        working = re.sub(r"^\s*(?:có|và|thì|là|mà|ở|trong|với)\b", " ", working, flags=re.I)
        working = re.sub(r"\b(?:có|và|thì|là|mà)\s*$", " ", working, flags=re.I)
        working = re.sub(r"\s+", " ", working).strip(" ,;.!?")

        if not _has_specific_visual_context(working):
            return QueryPlan(
                original_question=orig_q,
                search_query="",
                detail_queries=[],
                sufficient_context=False,
                method="missing_context",
            )

        if _contains_vietnamese(working):
            en_query = _translate_vi_context_to_en(working)
            method = "rule_vi_to_en_context"
        else:
            en_query = working
            method = "rule_en_context"

        if not _has_specific_visual_context(en_query):
            return QueryPlan(
                original_question=orig_q,
                search_query="",
                detail_queries=[],
                sufficient_context=False,
                method="missing_context",
            )

        details = split_query(en_query)
        return QueryPlan(
            original_question=orig_q,
            search_query=en_query,
            detail_queries=details if len(details) > 1 else [],
            sufficient_context=True,
            method=method,
        )
