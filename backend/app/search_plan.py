"""Normalize an image search with one text-only Gemini request."""
import asyncio
import json
import re
from collections import OrderedDict
from unicodedata import normalize

import httpx

from .schemas import SearchPlan
from .understanding import SEARCH_INSTRUCTIONS


PROMPT_VERSION = "castle-search-v13"
VIETNAMESE_MARKS = frozenset("đơưảạấầẩẫậắằẳẵặẻẽẹếềểễệỉịỏọốồổỗộớờởỡợủụứừửữựỷỹỵ")
STRIPE_DIRECTIONS = r"horizontal|vertical|diagonal"
STRIPE_ORIENTATION = re.compile(rf"\b({STRIPE_DIRECTIONS})(?:ly)?\s+(?:stripes|striped)\b", re.I)
ORIENTED_STRIPED = re.compile(rf"\b(?:{STRIPE_DIRECTIONS})(?:ly)?\s+striped\b", re.I)
ORIENTED_GARMENT = re.compile(
    rf"\b(sweater|shirt|jacket|coat|dress|skirt)\s+with\s+(?:{STRIPE_DIRECTIONS})\s+stripes\b", re.I,
)
# The caption, region-phrase and condition rules are shared with QA planning (understanding.py).
INSTRUCTIONS = SEARCH_INSTRUCTIONS


class SearchPlanError(Exception):
    pass


PERSON_WORDS = re.compile(r"\b(woman|women|man|men|person|people|girl|boy|child|kid|lady|guy)\b", re.I)
ATTRIBUTE_ONLY = re.compile(
    r"^(?:an?\s+|the\s+)?(?:[\w-]+\s+){0,4}(shirt|t-shirt|top|blouse|sweater|jumper|hoodie|jacket|coat|dress|skirt|"
    r"shorts|pants|trousers|jeans|leggings|apron|hat|cap|glasses|hair|ponytail|beard|scarf|shoes|sneakers)$", re.I)


def bind_person_attributes(plan):
    """Keep clothing and hair with the one person they describe.

    'woman with long hair' + 'white shirt' + 'dark shorts' becomes one phrase, so a white
    shirt worn by someone else cannot satisfy the requirement. With several people the
    owner is ambiguous, so the phrases are left as the planner wrote them.
    """
    details = list(plan.detail_queries_en)
    people = [i for i, phrase in enumerate(details) if PERSON_WORDS.search(phrase)]
    attributes = [i for i, phrase in enumerate(details)
                  if i not in people and ATTRIBUTE_ONLY.match(phrase.strip())]
    if len(people) != 1 or not attributes:
        return plan
    owner = details[people[0]]
    worn = [details[i].strip() for i in attributes]
    joined = " and ".join(worn)
    merged = f"{owner} in {joined}" if not any(w.lower().endswith(("hair", "ponytail", "beard")) for w in worn) \
        else f"{owner} with {joined}"
    kept = [merged if i == people[0] else phrase for i, phrase in enumerate(details) if i not in attributes]
    return plan.model_copy(update={"detail_queries_en": kept})


def compact_region_details(plan):
    # The caption retains direction. Keep it in the region phrases as well
    # when it distinguishes different subjects in the same description.
    description = " ".join([plan.global_query_en, *plan.detail_queries_en])
    if len({value.casefold() for value in STRIPE_ORIENTATION.findall(description)}) > 1:
        return plan
    details = [ORIENTED_STRIPED.sub("striped", ORIENTED_GARMENT.sub(r"striped \1", phrase))
               for phrase in plan.detail_queries_en]
    return plan.model_copy(update={"detail_queries_en": list(dict.fromkeys(details))})


class SearchPlanner:
    def __init__(self, settings, *, transport=None):
        gemini_qa = settings.qa_provider == 'gemini'
        self.key = settings.search_plan_api_key.strip() or (settings.qa_api_key.strip() if gemini_qa else '')
        own_backup = getattr(settings, 'search_plan_api_key_backup', '').strip()
        if own_backup:
            backups = [own_backup]
        elif gemini_qa:
            backups = getattr(settings, 'qa_backup_keys', None) or [getattr(settings, 'qa_api_key_backup', '').strip()]
        else:
            backups = []
        # Same model, further keys: tried in order when the active one is rate limited or overloaded.
        self.keys = [k for k in dict.fromkeys([self.key, *backups]) if k]
        self.model = settings.search_plan_model.strip()
        self.timeout = settings.search_plan_timeout_s
        self.cache_size = settings.search_plan_cache_size
        self.cache = OrderedDict()
        self.pending = {}
        self.client = httpx.AsyncClient(
            base_url="https://generativelanguage.googleapis.com/v1beta",
            transport=transport,
        )

    async def aclose(self):
        tasks = list(self.pending.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await self.client.aclose()

    async def plan(self, query, language):
        if not self.key or not self.model:
            raise SearchPlanError("Gemini chưa được cấu hình. Bạn có thể nhập mô tả tiếng Anh.")
        # Accented Vietnamese already identifies the language. Reuse the same
        # request in explicit Vietnamese and automatic mode.
        if language == "vi" and VIETNAMESE_MARKS.intersection(normalize("NFC", query).casefold()):
            language = "auto"
        key = (query, language, self.model, PROMPT_VERSION)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        if key not in self.pending:
            self.pending[key] = asyncio.create_task(self._generate(key, query, language))
        return await asyncio.shield(self.pending[key])

    async def _generate(self, key, query, language):
        payload = {
            "systemInstruction": {"parts": [{"text": INSTRUCTIONS}]},
            "contents": [{"role": "user", "parts": [{"text": json.dumps(
                {"query": query, "language": language}, ensure_ascii=False,
            )}]}],
            "generationConfig": {
                "temperature": 0,
                "responseMimeType": "application/json",
                "responseSchema": {
                    "type": "OBJECT",
                    "required": ["global_query_en", "detail_queries_en", "conditions"],
                    "properties": {
                        "global_query_en": {"type": "STRING", "description": "Natural English caption with all stated visual facts."},
                        "detail_queries_en": {
                            "type": "ARRAY", "maxItems": 5,
                            "description": "At most one short appearance phrase per visible subject. Put actions only in the caption.",
                            "items": {"type": "STRING", "description": (
                                "A visible entity with its distinguishing attributes, such as 'woman in a red polka-dot dress' "
                                "or 'open cabinet door'. Preserve a stated clothing pattern in the phrase."
                            )},
                        },
                        "conditions": {
                            "type": "ARRAY", "maxItems": 5,
                            "items": {
                                "type": "OBJECT", "required": ["kind", "subject"],
                                "properties": {
                                    "kind": {"type": "STRING", "enum": [
                                        "absent", "exactly", "at_least", "at_most", "action", "other",
                                        "left_of", "right_of", "above", "below", "near",
                                    ]},
                                    "subject": {"type": "STRING"},
                                    "target": {"type": "STRING"},
                                    "count": {"type": "INTEGER"},
                                },
                            },
                        },
                    },
                },
                "maxOutputTokens": 1024,
                "thinkingConfig": {"thinkingLevel": "MINIMAL"},
            },
        }
        try:
            async with asyncio.timeout(self.timeout):
                for index, key in enumerate(self.keys):
                    response = await self.client.post(
                        f"models/{self.model}:generateContent",
                        headers={"x-goog-api-key": key}, json=payload, timeout=self.timeout,
                    )
                    if response.status_code not in (429, 503) or index == len(self.keys) - 1:
                        break
            if response.status_code == 429:
                raise SearchPlanError("Gemini đã hết lượt gọi. Bạn có thể nhập mô tả tiếng Anh.")
            if not response.is_success:
                raise SearchPlanError(f"Gemini trả lỗi HTTP {response.status_code}. Bạn có thể nhập mô tả tiếng Anh.")
            parts = response.json()["candidates"][0]["content"]["parts"]
            plan = SearchPlan.model_validate_json("".join(
                part.get("text", "") for part in parts if not part.get("thought")
            ))
            plan = bind_person_attributes(compact_region_details(plan))
            self.cache[key] = plan
            while len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
            return plan
        except (TimeoutError, httpx.TimeoutException):
            raise SearchPlanError("Gemini phản hồi quá chậm. Bạn có thể nhập mô tả tiếng Anh.") from None
        except httpx.RequestError:
            raise SearchPlanError("Không kết nối được Gemini. Bạn có thể nhập mô tả tiếng Anh.") from None
        except (ValueError, KeyError, IndexError, TypeError, AttributeError):
            raise SearchPlanError("Gemini trả dữ liệu không hợp lệ. Bạn có thể nhập mô tả tiếng Anh.") from None
        finally:
            self.pending.pop(key, None)
