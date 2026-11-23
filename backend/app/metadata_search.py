"""Text match between a query and each image's metadata (data/image-metadata), added to the SigLIP score.

Why: every Castle5000 frame shows the same kitchen from the same camera, so whole-image vectors are nearly
identical and miss small, specific things ("striped sweater", "pineapple", "head-mounted camera"). The
metadata names exactly those things in words, so a plain BM25 match over it separates frames that SigLIP
ranks as ties. The match is added as a bonus in SigLIP-score units, so the scale every downstream step
was calibrated on (region weights, thresholds) stays the same, and an image without metadata only loses
the bonus.
"""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

from .image_metadata import ImageMetadata

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = set("""
a an the and or of in on at to for with by from into onto over under near next beside behind front back
is are was were be being been it its this that these those there their they them he she his her him one two
some any all each other another while who whom which what where when how than then also very just only
image frame photo picture scene camera view visible shows showing shown appears seen kitchen
""".split())
# Rough singular form so 'sweaters'/'sweater' and 'knives'/'knife' meet. Words that only look plural stay.
_KEEP = {"glasses", "jeans", "trousers", "shorts", "pants", "leggings", "scissors", "tongs", "clothes", "dress",
         "glass", "bus", "gas", "class", "grass", "press", "cross", "process", "this", "is", "was", "has", "its"}


def _stem(word: str) -> str:
    if word in _KEEP or len(word) <= 3:
        return word
    if word.endswith("ives") and len(word) > 5:
        return word[:-3] + "fe"        # knives -> knife
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith(("ches", "shes", "xes", "sses")):
        return word[:-2]
    if word.endswith("s") and not word.endswith(("ss", "us")):
        return word[:-1]
    return word


def tokens(text: str) -> list[str]:
    return [_stem(w) for w in _TOKEN.findall(text.lower()) if w not in _STOP]


def document(meta: ImageMetadata) -> str:
    """Every descriptive field, in plain words. Counts, lighting and frame type are left to other checks."""
    parts = [meta.caption, meta.activity]
    for p in meta.people:
        parts += [p.description, p.position, p.action, p.upper_clothing, p.lower_clothing, p.headwear,
                  p.posture if p.posture != "unclear" else "", *p.holding]
    for o in meta.objects:
        parts += [o.name, o.location, o.state]
    for s in meta.states:
        parts.append(f"{s.thing} {s.state}")
    parts += meta.visible_text + meta.distinctive_details
    return " . ".join(x for x in parts if x)


def entities(meta: ImageMetadata) -> list[str]:
    """One short document per person and per object, so a described person is matched against one person
    ('cream sweater with brown stripes' must sit on the same person, not be spread over two)."""
    out = []
    for p in meta.people:
        out.append(" . ".join(x for x in [p.description, p.upper_clothing, p.lower_clothing, p.headwear,
                                         p.action, p.position, *p.holding] if x))
    for o in meta.objects:
        out.append(" . ".join(x for x in [o.name, o.state, o.location] if x))
    for s in meta.states:
        out.append(f"{s.thing} {s.state}")
    return [x for x in out if x]


class _BM25:
    """BM25 over a list of token bags; rows are documents."""

    def __init__(self, docs: list[Counter], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.size = len(docs)
        self.lengths = np.array([sum(d.values()) for d in docs], dtype="float32")
        filled = self.lengths > 0
        self.avg = float(self.lengths[filled].mean()) if filled.any() else 1.0
        self.postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        raw: dict[str, list[tuple[int, int]]] = {}
        for i, doc in enumerate(docs):
            for term, tf in doc.items():
                raw.setdefault(term, []).append((i, tf))
        for term, rows in raw.items():
            self.postings[term] = (np.array([i for i, _ in rows], dtype=np.int64),
                                   np.array([f for _, f in rows], dtype="float32"))
        n = int(filled.sum())
        self.idf = {t: math.log(1 + (n - len(r) + 0.5) / (len(r) + 0.5)) for t, (r, _) in self.postings.items()}
        self.norm = self.k1 * (1 - self.b + self.b * self.lengths / self.avg)

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        out = np.zeros(self.size, dtype="float32")
        for term, qtf in Counter(query_tokens).items():
            if term not in self.postings:
                continue
            rows, tf = self.postings[term]
            out[rows] += self.idf[term] * tf * (self.k1 + 1) / (tf + self.norm[rows]) * (1 + 0.5 * (qtf - 1))
        return out

    def ceiling(self, query_tokens: list[str]) -> float:
        """Score of a document holding every query word once at average length (scale for 0..1)."""
        return sum(self.idf.get(t, 0.0) for t in set(query_tokens)) or 1.0


_NUMBERS = {"one": 1, "single": 1, "lone": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
            "seven": 7, "eight": 8, "nine": 9, "ten": 10}
_PEOPLE = r"(?:people|persons|men|women|adults|guests|individuals|man|woman|person)"


def people_count(query: str) -> tuple[int, bool] | None:
    """(n, exact) when the query states how many people are in the frame: 'six people in the kitchen',
    'a single man ... nobody else' (exact), 'five men around the island' (at least). None otherwise."""
    q = query.lower()
    if re.search(r"\b(?:empty kitchen|no one|nobody (?:is )?(?:in|there|present))\b", q) and \
            not re.search(r"\bnobody else\b", q):
        return 0, True
    if re.search(r"\b(?:alone|nobody else|no one else|only one (?:man|woman|person))\b", q):
        return 1, True
    m = re.search(rf"\b(\d{{1,2}}|{'|'.join(_NUMBERS)})\s+(?:\w+\s+){{0,2}}?{_PEOPLE}\b", q)
    if not m:
        return None
    n = int(m.group(1)) if m.group(1).isdigit() else _NUMBERS[m.group(1)]
    # 'two men stand together' names a group, not the whole frame: only a total wording is exact.
    exact = bool(re.search(r"\b(?:people|persons)\b\s+(?:in|around|standing|sitting|at)\b|\bin the kitchen\b",
                           q[m.start():m.end() + 30])) or n == 1 and m.group(1) == "single"
    return (n, exact) if n >= 1 else None


class MetadataMatcher:
    """Text match between a query and each image's metadata, aligned with the vector index ids.

    Two views of the same metadata: one document per image (the whole scene) and one per person or object.
    The scene view rewards shared words anywhere; the entity view asks that each described person or object
    is found as one entity, which separates 'a man in a cream striped sweater and a woman in black' from a
    frame where those words belong to different people."""

    def __init__(self, store, ids: list[str]):
        self.ids = ids
        self.has = np.zeros(len(ids), dtype=bool)
        docs: list[Counter] = []
        entity_docs: list[Counter] = []
        owner: list[int] = []
        for i, image_id in enumerate(ids):
            meta = store.get(image_id)
            self.has[i] = meta is not None
            docs.append(Counter(tokens(document(meta))) if meta else Counter())
            for text in entities(meta) if meta else []:
                entity_docs.append(Counter(tokens(text)))
                owner.append(i)
        self.counts = np.array([float(store.get(i).people_count_estimate) if store.get(i) else np.nan for i in ids],
                               dtype="float32")
        self.scene = _BM25(docs)
        self.entity = _BM25(entity_docs)
        self.owner = np.array(owner, dtype=np.int64)

    @property
    def coverage(self) -> int:
        return int(self.has.sum())

    def scores(self, query: str) -> np.ndarray:
        """Scene BM25 score of every image (0 without metadata or without a shared word)."""
        return self.scene.scores(tokens(query))

    def entity_scores(self, phrases: list[str]) -> np.ndarray:
        """Mean over phrases of the best single-entity match in each image, each phrase scaled to 0..1."""
        out = np.zeros(len(self.ids), dtype="float32")
        used = 0
        for phrase in phrases:
            words = tokens(phrase)
            if not words:
                continue
            per_entity = self.entity.scores(words) / self.entity.ceiling(words)
            best = np.zeros(len(self.ids), dtype="float32")
            np.maximum.at(best, self.owner, per_entity)
            out += np.minimum(best, 1.0)
            used += 1
        return out / used if used else out

    def match(self, query: str, details=()) -> np.ndarray:
        """Standardised metadata match (scene + entities), only the part above the corpus mean."""
        def z(values):
            if not values.any():
                return np.zeros_like(values)
            return np.clip((values - values.mean()) / (values.std() + 1e-6), 0, None)
        parts = [z(self.scores(query))]
        phrases = [d for d in details if d and d.strip()]
        if phrases:
            parts.append(z(self.entity_scores(phrases)))
        stated = people_count(query)
        if stated is not None:
            parts.append(self.count_scores(*stated))
        return sum(parts) / len(parts)

    def count_scores(self, n: int, exact: bool) -> np.ndarray:
        """2 when the metadata head count fits the stated number, 1 when it is one off, else 0.
        'At least' wordings fit any larger count. Images without metadata get 0."""
        est = self.counts
        off = np.abs(est - n) if exact else np.clip(n - est, 0, None)
        out = np.where(off == 0, 2.0, np.where(off <= 1, 1.0, 0.0))
        return np.nan_to_num(out, nan=0.0).astype("float32")

    def bonus(self, query: str, clip_scores: np.ndarray, weight: float, details=()) -> np.ndarray:
        """Metadata match in SigLIP units: weight x (standard deviation of SigLIP scores) per standard
        deviation of the match above its mean. Images that share no word with the query get no bonus."""
        if weight <= 0 or not query.strip():
            return np.zeros_like(clip_scores)
        return (weight * float(clip_scores.std()) * self.match(query, details)).astype(clip_scores.dtype)
