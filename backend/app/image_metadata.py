"""Per-image visual metadata written offline by an open VLM (Kaggle notebook), read by search and QA.

Why: in Castle5000 almost every frame comes from the same fixed fisheye kitchen camera, so whole-image
vectors are nearly identical (cosine 0.96 to the corpus mean) and cannot express negation or counts.
What separates frames is who is there, where they stand, what they do, what lies on the counters and
the state of appliances. The schema records exactly that, in the plain English phrases search uses.

Counts and boxes stay with Grounding DINO (person_crops.py); the VLM describes, it does not count
precisely. One schema serves the notebook (as JSON schema for guided decoding) and the loader.
"""
import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

SCHEMA = 'image-metadata-v1'

Phrase = Field(default='', max_length=160)


class Person(BaseModel):
    model_config = ConfigDict(extra='forbid')
    description: str = Field(max_length=160, description=(
        "Short retrieval phrase for this person: apparent gender if clear, hair, and clothing with colors "
        "and patterns, e.g. 'man in a dark sweater with white and brown horizontal stripes'"))
    position: str = Phrase  # relative to landmarks: 'at the stove', 'beside the sink', 'in the doorway to the living room'
    posture: Literal['standing', 'sitting', 'bending', 'crouching', 'walking', 'unclear'] = 'unclear'
    action: str = Phrase  # 'cutting vegetables on a board', 'stirring a pot', 'opening the refrigerator'
    holding: list[str] = Field(default_factory=list, max_length=4)
    upper_clothing: str = Phrase
    lower_clothing: str = Phrase
    headwear: str = Phrase  # 'cap', 'head-mounted camera', 'none'
    partly_visible: bool = False


class ObjectEntry(BaseModel):
    model_config = ConfigDict(extra='forbid')
    name: str = Field(max_length=60)          # singular noun phrase: 'pineapple', 'red cutting board'
    count: int | None = Field(default=None, ge=0, le=50)  # None when not countable at a glance
    location: str = Phrase                     # 'on the kitchen island', 'on the stove'
    state: str = Field(default='', max_length=80)  # 'open', 'steaming', 'empty', 'hanging on the oven handle'


class StateEntry(BaseModel):
    model_config = ConfigDict(extra='forbid')
    thing: str = Field(max_length=60)    # 'refrigerator door', 'oven door', 'tap', 'stove burner', 'lights'
    state: str = Field(max_length=60)    # 'open', 'closed', 'running', 'on', 'not visible'


class ImageMetadata(BaseModel):
    model_config = ConfigDict(extra='forbid')
    frame_type: Literal['scene', 'test_card', 'blank_or_corrupt'] = 'scene'
    lighting: Literal['daylight', 'night_artificial', 'mixed', 'unclear'] = 'unclear'
    caption: str = Field(max_length=600, description=(
        "Two or three short English sentences for image search: the main activity and who does it first, "
        "then other people and notable objects."))
    activity: str = Field(default='', max_length=120)   # whole scene: 'several people preparing food', 'empty kitchen'
    people_count_estimate: int = Field(ge=0, le=40)
    people: list[Person] = Field(default_factory=list, max_length=15)
    objects: list[ObjectEntry] = Field(default_factory=list, max_length=25)
    states: list[StateEntry] = Field(default_factory=list, max_length=10)
    visible_text: list[str] = Field(default_factory=list, max_length=6)
    distinctive_details: list[str] = Field(default_factory=list, max_length=6)


PROMPT = (
    "You annotate one frame for an image retrieval and question answering system. Describe only what is "
    "visible. Use plain everyday English phrases that a person would type to search. Never guess names, "
    "identities or anything off-camera. Use 'unclear' or leave a field empty instead of guessing.\n"
    "- frame_type: 'test_card' for a test pattern or title card, 'blank_or_corrupt' for black, frozen or broken frames.\n"
    "- people: one entry per visible person, most prominent first, including people partly visible at the edges. "
    "Describe clothing with colors and patterns (striped, plaid, checkered), headwear including head-mounted cameras, "
    "where the person is relative to landmarks (stove, sink, island, refrigerator, doorway, sofa), what they are doing "
    "and what they hold.\n"
    "- objects: food, utensils, cookware, towels, bags and other notable items with location and state; give count "
    "only when clear at a glance.\n"
    "- states: doors (refrigerator, oven, cupboards), tap, stove, lights, when visible.\n"
    "- distinctive_details: anything that would let someone pick this frame out of many similar ones "
    "(e.g. 'red and white checkered towel over a man's shoulder').\n"
    "Return only JSON matching the schema."
)


def json_schema():
    return ImageMetadata.model_json_schema()


NORMALIZATION = 'norm-v1'

# Several Gemini models annotated the corpus (the free daily quota is per model). They agree on structure
# but word the same thing differently ('headlamp' vs 'head-mounted camera', 'trash cans' vs 'trash bin').
# These rewrites, applied when shards are merged, give the store one vocabulary as if one model wrote it.
_TEXT_RULES = [
    (r'\bhead[- ]?mounted (?:camera|device|light)s? (?:strap|mount)s?\b', 'head-mounted camera'),
    (r'\bhead[- ]?(?:band|strap) camera(?: strap)?s?\b', 'head-mounted camera'),
    (r'\bhead[- ]?lamp strap\b', 'head-mounted camera'),
    (r'\bhead[- ]?lamps\b', 'head-mounted cameras'),
    (r'\bhead[- ]?lamp\b', 'head-mounted camera'),
    (r'\bhead[- ]?mounted (?:device|light)s\b', 'head-mounted cameras'),
    (r'\bhead[- ]?mounted (?:device|light)\b', 'head-mounted camera'),
    (r'\b(?:trash|rubbish|garbage|waste) cans\b', 'trash bins'),
    (r'\b(?:trash|rubbish|garbage|waste) can\b', 'trash bin'),
    (r'\b(?:rubbish|garbage|waste) bin', 'trash bin'),
    (r'\bchopping board', 'cutting board'),
    (r'\baluminium\b', 'aluminum'),
    (r'\b(?:aluminum )?foil tray', 'aluminum tray'),
    (r'\bcouch(es)?\b', r'sofa'),
    (r'\bfridge\b', 'refrigerator'),
    (r'\b(?:pendant|hanging) lamp', 'pendant light'),
    (r'\bhanging light', 'pendant light'),
]
_HEAD_CAMERA = re.compile(r'head[- ]?(?:lamp|band|strap|set|mounted)', re.I)
_EMPTY = {'', 'none', 'unclear', 'no', 'n/a', 'na', 'nothing', 'not visible', 'unknown', 'no headwear'}
_KEEP_PLURAL = {'glasses', 'chips', 'noodles', 'clothes', 'scissors', 'tongs', 'trousers', 'jeans', 'shorts',
                'leggings', 'pants', 'crisps', 'oats', 'lentils', 'peas', 'beans', 'grapes', 'sweets', 'snacks'}
_OBJECT_NAMES = {'island': 'kitchen island', 'island counter': 'kitchen island', 'bin': 'trash bin',
                 'range cooker': 'stove', 'range stove': 'stove', 'stove and oven': 'stove', 'stove oven': 'stove',
                 'stool': 'bar stool', 'pendant': 'pendant light'}
_STATE_THINGS = {
    'pendant lights': 'pendant lights', 'pendant lamps': 'pendant lights', 'hanging lights': 'pendant lights',
    'hanging lamps': 'pendant lights', 'pendants': 'pendant lights', 'overhead pendant lights': 'pendant lights',
    'lights': 'lights', 'kitchen lights': 'lights', 'overhead lights': 'lights', 'ceiling lights': 'lights',
    'oven': 'oven door', 'oven doors': 'oven door', 'stove oven door': 'oven door', 'stove doors': 'oven door',
    'stove oven': 'stove', 'cooker': 'stove', 'hob': 'stove',
    'cupboard': 'cupboards', 'cupboard doors': 'cupboards', 'cabinets': 'cupboards', 'cabinet doors': 'cupboards',
    'upper cupboards': 'cupboards', 'pantry cupboard': 'cupboards', 'kitchen cupboards': 'cupboards',
    'refrigerator': 'refrigerator door', 'refrigerator doors': 'refrigerator door',
    'sink tap': 'tap', 'faucet': 'tap', 'kitchen tap': 'tap', 'trash cans': 'trash bins', 'trash bin': 'trash bins',
}
_ON = {'on', 'active', 'turned on', 'switched on', 'lit', 'illuminated', 'glowing', 'in use', 'burning'}
_OFF = {'off', 'idle', 'turned off', 'switched off', 'inactive', 'unlit'}


def _text(value):
    value = re.sub(r'\s+', ' ', str(value)).strip().lstrip(':;-').strip()
    for pattern, replacement in _TEXT_RULES:
        value = re.sub(pattern, replacement, value, flags=re.I)
    return value


def _phrase(value):
    text = _text(value).lower()
    return '' if text in _EMPTY else text


def _singular(name):
    words = name.split(' ')
    last = words[-1]
    if last in _KEEP_PLURAL or len(last) <= 3 or not last.endswith('s') or last.endswith(('ss', 'us', 'is')):
        return name
    if last.endswith('ies'):
        last = last[:-3] + 'y'
    elif last.endswith('ves'):
        last = last[:-3] + 'fe' if last == 'knives' else last[:-3] + 'f'
    elif last.endswith(('ches', 'shes', 'xes', 'sses', 'oes')):
        last = last[:-2]
    else:
        last = last[:-1]
    return ' '.join(words[:-1] + [last])


def _headwear(value):
    text = _phrase(value)
    if not text or not _HEAD_CAMERA.search(text):
        return text
    return 'cap and head-mounted camera' if 'cap' in text else 'head-mounted camera'


def _state_value(thing, value):
    text = _phrase(value)
    if text in _ON or (thing in ('stove', 'oven') and ('pot' in text or 'pan' in text or 'cooking' in text)):
        return 'on'
    if text in _OFF:
        return 'off'
    if 'running' in text:
        return 'running'
    if text in ('ajar', 'partly open', 'partially open', 'slightly open', 'half open'):
        return 'partly open'
    if text in ('open', 'opened', 'wide open'):
        return 'open'
    if text in ('closed', 'shut', 'mostly closed'):
        return 'closed'
    return ''  # free text such as 'red tiles' or 'cluttered with food' is not an appliance state


def normalize_metadata(metadata):
    """One vocabulary for records written by different models; input and output follow ImageMetadata."""
    m = ImageMetadata.model_validate(metadata).model_dump()
    m['caption'] = _text(m['caption'])
    m['activity'] = _phrase(m['activity'])
    m['visible_text'] = [t for t in (_text(v) for v in m['visible_text']) if t]
    m['distinctive_details'] = list(dict.fromkeys(t for t in (_text(v) for v in m['distinctive_details']) if t))
    for person in m['people']:
        person['description'] = _text(person['description'])
        for key in ('position', 'action', 'upper_clothing', 'lower_clothing'):
            person[key] = _phrase(person[key])
        person['headwear'] = _headwear(person['headwear'])
        person['holding'] = list(dict.fromkeys(h for h in (_phrase(v) for v in person['holding']) if h))
    objects = {}
    for entry in m['objects']:
        name = _singular(_phrase(entry['name']))
        name = _OBJECT_NAMES.get(name, name)
        if not name or name in objects:
            continue
        state = _phrase(entry['state'])
        objects[name] = {**entry, 'name': name, 'location': _phrase(entry['location']),
                         'state': 'on' if state in _ON else 'off' if state in _OFF else state}
    m['objects'] = list(objects.values())
    states = {}
    for entry in m['states']:
        thing = _phrase(entry['thing'])
        thing = _STATE_THINGS.get(thing, thing)
        value = _state_value(thing, entry['state'])
        if thing and value and thing not in states:
            states[thing] = {'thing': thing, 'state': value}
    m['states'] = list(states.values())
    return ImageMetadata.model_validate(m).model_dump()


def frame_time(image_id):
    """Capture time from the file name 'HH_second.jpg' (hour of day, second within that hour).

    Taken from the name rather than asked of the VLM, which can only guess it from the picture.
    None when the name does not follow the pattern."""
    parts = image_id.rsplit('.', 1)[0].split('_')
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        return None
    hour, second = int(parts[0]), int(parts[1])
    return {'hour': hour, 'second': second, 'clock': f'{hour:02d}:{second // 60:02d}:{second % 60:02d}'}


class MetadataStore:
    """metadata.jsonl lines {"image_id": ..., "time": {...}, "metadata": {...}} plus manifest.json."""

    def __init__(self, records, manifest):
        self.records = records
        self.manifest = manifest

    @staticmethod
    def time(image_id):
        return frame_time(image_id)

    @classmethod
    def load(cls, directory, ids):
        directory = Path(directory)
        manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
        if manifest.get('schema') != SCHEMA:
            raise ValueError('Not an image metadata store')
        known = set(ids)
        records = {}
        with (directory / 'metadata.jsonl').open(encoding='utf-8') as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row['image_id'] not in known:
                    raise ValueError('Metadata for an unknown image: ' + row['image_id'])
                try:
                    records[row['image_id']] = ImageMetadata.model_validate(row['metadata'])
                except ValidationError:
                    continue  # One bad record must not disable the rest.
        return cls(records, manifest)

    def get(self, image_id):
        return self.records.get(image_id)
