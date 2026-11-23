"""Who appears in the corpus, from the image metadata: groups of person descriptions that look like one person.

Used to plan which frames a "how many different people" question reads. Reading the frames most similar to
the question repeats the same few people; one frame per description group covers everyone who appears often.
The groups are a plan, not the answer: text descriptions vary ('blue shirt' / 'blue button-down shirt'), so the
final count still comes from comparing the people seen in those frames.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

_COLORS = {"black", "white", "grey", "gray", "blue", "red", "green", "yellow", "orange", "pink", "purple", "brown",
           "beige", "cream", "navy", "dark", "light", "maroon", "khaki", "teal", "burgundy", "olive", "tan"}
_PATTERNS = {"plaid", "checkered", "striped", "stripes", "stripe", "patterned", "floral", "graphic", "logo"}
_GARMENTS = {"shirt", "t-shirt", "top", "sweater", "hoodie", "jacket", "polo", "jersey", "blouse", "cardigan",
             "fleece", "vest", "dress", "apron", "coat", "jumper", "sweatshirt"}
_SAME = {"gray": "grey", "navy": "blue", "stripes": "striped", "stripe": "striped", "checkered": "plaid",
         "jumper": "sweater", "sweatshirt": "sweater", "t-shirt": "tshirt", "blouse": "top"}


def _words(text):
    return [_SAME.get(w, w) for w in re.findall(r"[a-z]+(?:-[a-z]+)?", (text or "").lower())]


def signature(person) -> tuple | None:
    """(gender, colours, patterns, garment, head camera) of one person record, or None when too vague."""
    desc = _words(person.description)
    upper = _words(person.upper_clothing) or desc
    gender = "woman" if {"woman", "girl", "female", "lady"} & set(desc) else \
        "man" if {"man", "boy", "male", "guy"} & set(desc) else "?"
    colours = frozenset(w for w in upper if w in _COLORS and w not in {"dark", "light"})
    patterns = frozenset(w for w in upper if w in _PATTERNS)
    garment = next((w for w in upper if w in _GARMENTS or w == "tshirt"), None)
    camera = "camera" in " ".join(_words(person.headwear)) or "head-mounted" in person.description.lower()
    if not colours or garment is None:
        return None
    return gender, colours, patterns, garment, camera


@dataclass
class Group:
    key: tuple
    frames: list = field(default_factory=list)   # (image_id, people in the frame)
    variants: Counter = field(default_factory=Counter)

    @property
    def label(self):
        gender, colours, patterns, garment, camera = self.key
        parts = [gender if gender != "?" else "person", "/".join(sorted(colours)), " ".join(sorted(patterns)), garment]
        return " ".join(p for p in parts if p) + (" + head camera" if camera else "")

    def representative(self):
        """A frame where this person is easy to see: fewest people, then earliest."""
        return min(self.frames, key=lambda f: (f[1], f[0]))[0]


def people_groups(store, ids, min_frames=15):
    """Description groups seen in at least `min_frames` frames, most frequent first."""
    groups: dict[tuple, Group] = {}
    for image_id in ids:
        meta = store.get(image_id)
        if meta is None:
            continue
        for person in meta.people:
            key = signature(person)
            if key is None:
                continue
            g = groups.setdefault(key, Group(key))
            g.frames.append((image_id, len(meta.people)))
            g.variants[person.upper_clothing.lower()] += 1
    # A head-camera flag missed on some frames should not split a person in two.
    merged: dict[tuple, Group] = {}
    for key, g in sorted(groups.items(), key=lambda kv: -len(kv[1].frames)):
        base = key[1:4]
        # Same clothing with the head-camera flag or the gender missing on some frames is the same group.
        target = next((m for k, m in merged.items() if k[1:4] == base and (k[0] == key[0] or "?" in (k[0], key[0]))),
                      None)
        if target is None:
            merged[key] = g
        else:
            target.frames += g.frames
            target.variants.update(g.variants)
    out = [g for g in merged.values() if len(g.frames) >= min_frames]
    return sorted(out, key=lambda g: -len(g.frames))


def coverage_frames(store, ids, limit, min_frames=None):
    """The most crowded frame, then one representative frame per frequent description group (most frequent
    groups first): together they show everyone who appears often."""
    if min_frames is None:
        # 15 frames (~2 minutes) over the whole day; fewer for a smaller scope such as one hour.
        min_frames = max(1, min(15, len(ids) // 300))
    crowded = max_simultaneous(store, ids)[1]
    picks = [crowded] if crowded else []
    for g in people_groups(store, ids, min_frames):
        frame = g.representative()
        if frame not in picks:
            picks.append(frame)
        if len(picks) >= limit:
            break
    return picks


def max_simultaneous(store, ids):
    """Largest head count in any one frame (a lower bound on how many people took part) and its frame."""
    best = (0, None)
    for image_id in ids:
        meta = store.get(image_id)
        if meta is not None and meta.people_count_estimate > best[0]:
            best = (meta.people_count_estimate, image_id)
    return best


def brief_metadata(meta, limit=700) -> str:
    """Compact frame description for the QA reader: head count, each person, objects and states."""
    parts = [f"about {meta.people_count_estimate} people"]
    for i, p in enumerate(meta.people, 1):
        bits = [p.description, p.position, p.action]
        if p.holding:
            bits.append("holding " + ", ".join(p.holding))
        parts.append(f"P{i}: " + "; ".join(b for b in bits if b))
    objects = [o.name + (f" ({o.state})" if o.state else "") for o in meta.objects[:10]]
    if objects:
        parts.append("objects: " + ", ".join(objects))
    states = [f"{s.thing} {s.state}" for s in meta.states]
    if states:
        parts.append("states: " + ", ".join(states))
    text = " | ".join(parts)
    return text if len(text) <= limit else text[:limit - 1] + "…"
