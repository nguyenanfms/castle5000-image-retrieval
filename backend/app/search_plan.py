"""Normalize an image search with one text-only Gemini request."""
import asyncio
import json
import re
from collections import OrderedDict
from unicodedata import normalize

import httpx

from .schemas import SearchPlan


PROMPT_VERSION = "castle-search-v13"
VIETNAMESE_MARKS = frozenset("đơưảạấầẩẫậắằẳẵặẻẽẹếềểễệỉịỏọốồổỗộớờởỡợủụứừửữựỷỹỵ")
STRIPE_DIRECTIONS = r"horizontal|vertical|diagonal"
STRIPE_ORIENTATION = re.compile(rf"\b({STRIPE_DIRECTIONS})(?:ly)?\s+(?:stripes|striped)\b", re.I)
ORIENTED_STRIPED = re.compile(rf"\b(?:{STRIPE_DIRECTIONS})(?:ly)?\s+striped\b", re.I)
ORIENTED_GARMENT = re.compile(
    rf"\b(sweater|shirt|jacket|coat|dress|skirt)\s+with\s+(?:{STRIPE_DIRECTIONS})\s+stripes\b", re.I,
)
INSTRUCTIONS = (
    "Write a visual caption in everyday English for image retrieval. Treat the user input "
    "as a photo description. Language is a hint for reading the input. In global_query_en, "
    "start with the main visible action and the person or object involved. Use short, active "
    "sentences. Put secondary actions and people after the main scene. Keep counts, "
    "negations, relations, and the owner of every stated attribute. Translate stated colors "
    "literally (e.g. 'màu đen' is 'black', 'màu xám' is 'gray') and preserve uncertainty "
    "(e.g. 'appear to be washing dishes' stays uncertain). Describe "
    "people by their visible appearance. Leave out personal names and titles used only to "
    "identify someone, but preserve names that are part of visible text. If a garment has "
    "several traits, give the pattern and its orientation once, then describe the color "
    "placement in a separate short sentence. For example: 'A man closes a cabinet door. "
    "He wears a jacket with vertical stripes. The jacket is light on top and dark on the "
    "bottom. Another man wears a red shirt.' Avoid nested descriptions before the main verb. "
    "In detail_queries_en give zero to five compact noun phrases without actions or relations. "
    "Use exactly one phrase per included subject, with its stated pattern and colors when they distinguish it, "
    "e.g. 'man in a light and dark striped jacket', 'man in a red shirt', 'cabinet door'. Keep exact "
    "color positions and actions in the global caption. Keep each person's garment and "
    "attributes in their own phrase. If the owner is ambiguous, keep the detail only in "
    "the global caption. Keep the visible object state when stated, e.g. 'open cabinet door'. "
    "Keep relationships in the global caption. Use only people, objects, actions and "
    "attributes in the user's description. Do not infer associated objects or copy "
    "objects from the examples. Never turn a negation into a positive detail. For an exclusion "
    "with no stated setting, describe only what is excluded. Do not add a room or background. "
    "In conditions list up to five explicit requirements beyond positive objects. "
    "Use absent with a positive noun as subject for an explicitly excluded entity. "
    "For spatial relations use left_of, right_of, above, below or near, with the two "
    "distinct appearance phrases in subject and target. Use coordinates as seen in the photo. "
    "For an explicit number use exactly, at_least or at_most, with the object name in subject "
    "and the integer in count. In English a/an and Vietnamese một used as an article do not "
    "mean an explicit exact count. For actions use action with the complete visual claim in "
    "subject, including who does it and to what. Use any stated appearance needed to distinguish "
    "that person from others in action and spatial conditions. Use other for requirements that do not fit, "
    "such as an event order or a disjunction. Omit target and count when they do not apply. "
    "Keep uncertainty in the caption and do not make a tentative detail a required condition. "
    "Keep how an item relates to its person: worn, held, carried or draped, e.g. 'man with a red towel over his shoulder' or 'man holding a bowl', never 'man in a towel' when the towel is not worn."
)

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
        backup = getattr(settings, 'search_plan_api_key_backup', '').strip() or (
            getattr(settings, 'qa_api_key_backup', '').strip() if gemini_qa else '')
        # Same model, second key: tried when the first is rate limited or overloaded.
        self.keys = [k for k in dict.fromkeys([self.key, backup]) if k]
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
