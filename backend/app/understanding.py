"""Shared understanding of a described scene, used by image search (KIS) and question answering (QA).

Both planners turn Vietnamese or English text into the same three parts:
  * a retrieval caption in English (KIS global_query_en, QA queries);
  * short region phrases, one per visible subject (KIS detail_queries_en, QA detail_queries);
  * structured scene conditions (absent, exactly/at_least/at_most, left_of..near, action, other)
    that backend/app/search_conditions.py checks against regions and detections.
One rule text keeps both planners consistent, so a QA question about a scene retrieves the same
images, with the same exclusions, as a KIS search for that scene.
"""
from pydantic import ValidationError

from .schemas import COUNT_CONDITIONS, SearchCondition

CAPTION_RULES = (
    'Write a visual caption in everyday English for image retrieval. Treat the user input as a photo '
    'description. Language is a hint for reading the input. In global_query_en, start with the main '
    'visible action and the person or object involved. Use short, active sentences. Put secondary '
    'actions and people after the main scene. Keep counts, negations, relations, and the owner of '
    "every stated attribute. Translate stated colors literally (e.g. 'màu đen' is 'black', 'màu xám' "
    "is 'gray') and preserve uncertainty (e.g. 'appear to be washing dishes' stays uncertain). "
    'Describe people by their visible appearance. Leave out personal names and titles used only to '
    'identify someone, but preserve names that are part of visible text. If a garment has several '
    'traits, give the pattern and its orientation once, then describe the color placement in a '
    "separate short sentence. For example: 'A man closes a cabinet door. He wears a jacket with "
    'vertical stripes. The jacket is light on top and dark on the bottom. Another man wears a red '
    "shirt.' Avoid nested descriptions before the main verb. In detail_queries_en give zero to five "
    'compact noun phrases without actions or relations. Use exactly one phrase per included subject, '
    "with its stated pattern and colors when they distinguish it, e.g. 'man in a light and dark "
    "striped jacket', 'man in a red shirt', 'cabinet door'. Keep exact color positions and actions "
    "in the global caption. Keep each person's garment and attributes in their own phrase. If the "
    'owner is ambiguous, keep the detail only in the global caption. Keep the visible object state '
    "when stated, e.g. 'open cabinet door'. Keep relationships in the global caption. Use only "
    "people, objects, actions and attributes in the user's description. Do not infer associated "
    'objects or copy objects from the examples. Never turn a negation into a positive detail. For an '
    'exclusion with no stated setting, describe only what is excluded. Do not add a room or '
    'background. '
)

CONDITION_RULES = (
    'In conditions list up to five explicit requirements beyond positive objects. Use absent with a '
    'positive noun as subject for an explicitly excluded entity. For spatial relations use left_of, '
    'right_of, above, below or near, with the two distinct appearance phrases in subject and target. '
    'Use coordinates as seen in the photo. For an explicit number use exactly, at_least or at_most, '
    'with the object name in subject and the integer in count. In English a/an and Vietnamese một '
    'used as an article do not mean an explicit exact count. For actions use action with the '
    'complete visual claim in subject, including who does it and to what. Use any stated appearance '
    'needed to distinguish that person from others in action and spatial conditions. Use other for '
    'requirements that do not fit, such as an event order or a disjunction. Omit target and count '
    'when they do not apply. Keep uncertainty in the caption and do not make a tentative detail a '
    'required condition. '
)

RELATION_RULE = (
    "Keep how an item relates to its person: worn, held, carried or draped, e.g. 'man with a red "
    "towel over his shoulder' or 'man holding a bowl', never 'man in a towel' when the towel is not "
    'worn.'
)

SEARCH_INSTRUCTIONS = CAPTION_RULES + CONDITION_RULES + RELATION_RULE

QA_RULES = (
    'Scene understanding shared with image search: write each entry of queries like global_query_en, '
    'each detail_queries entry like detail_queries_en, and conditions as described below. '
    + CAPTION_RULES + CONDITION_RULES + RELATION_RULE
    + ' conditions hold the same visual filters as known_conditions in structured form, only those the '
    'question states about the scene. Never put the asked quantity or answer in conditions: for "how many '
    'cups are on the table when nobody is in the kitchen" use absent person, not a count of cups.'
)


def valid_conditions(items, limit=5):
    """Keep the well-formed conditions and drop the rest, so one bad item does not fail a plan."""
    kept = []
    for item in items or ():
        try:
            condition = item if isinstance(item, SearchCondition) else SearchCondition.model_validate(item)
        except (ValidationError, TypeError, ValueError):
            continue
        if condition not in kept:
            kept.append(condition)
    return kept[:limit]


def retrieval_conditions(conditions, operation, subject):
    """Conditions that filter retrieval for a QA question.

    A count condition on the very thing being counted would encode the answer and drop the images
    that disprove it, so it is left to the reading step.
    """
    counted = (subject or '').casefold().strip()
    counting = operation in {'count_visible', 'count_max', 'count_distinct_people', 'count_distinct_objects'}
    return tuple(c for c in conditions
                 if not (counting and c.kind in COUNT_CONDITIONS
                         and (c.subject.casefold().strip() == counted
                              or (operation == 'count_distinct_people' and c.subject.casefold() in {'person', 'people'}))))
