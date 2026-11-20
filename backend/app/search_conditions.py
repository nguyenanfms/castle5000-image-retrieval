"""Check explicit search conditions with the locally indexed image evidence."""

import json
import math
from pathlib import Path

from .regions import iou
from .schemas import COUNT_CONDITIONS, SPATIAL_CONDITIONS


DETECTION_SCORE = 0.60
DETECTOR = "IDEA-Research/grounding-dino-tiny"
DETECTOR_LABELS = frozenset({
    "person", "refrigerator", "sink", "faucet", "table", "stove", "cup", "bowl", "plate", "bottle", "pot", "pan",
})
ALIASES = {"people": "person", "persons": "person", "fridge": "refrigerator"}


def canonical_label(text):
    label = text.casefold().strip()
    if label in ALIASES:
        return ALIASES[label]
    if label in DETECTOR_LABELS:
        return label
    return label.removesuffix("s") if label.removesuffix("s") in DETECTOR_LABELS else None


def valid_detection(row):
    box, score = row["box"], row["score"]
    return (row["label"] in DETECTOR_LABELS and 0 <= score <= 1 and len(box) == 4
            and 0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1)


def load_detections(path: Path, ids):
    data = json.loads(path.read_bytes())
    if (data["schema"] != "search-detections-v1" or data["detector"] != DETECTOR
            or not isinstance(data["images"], dict) or list(data["images"]) != list(ids)
            or not all(valid_detection(row) for items in data["images"].values() for row in items)):
        raise ValueError("Invalid search detection index")
    return data["images"], data["version"]


def distinct_detections(rows):
    selected = []
    for row in sorted(rows, key=lambda item: -item["score"]):
        if any(same_detection(row["box"], existing["box"]) for existing in selected):
            continue
        selected.append(row)
    return selected


def same_detection(a, b):
    overlap = iou(a, b)
    areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in (a, b)]
    shared_area = overlap * sum(areas) / (1 + overlap)
    return overlap >= .35 or shared_area >= .8 * min(areas)


def positive_checks(matches):
    return [{"kind": "present", "subject": match["query"], "status": "met" if match["supported"] else "unknown",
             "reason": "localized" if match["supported"] else "weak_region",
             "evidence": [match] if match.get("box") else []} for match in matches]


def supports_position(kind, a, b):
    if iou(a, b) >= .35:
        return False
    dx = (b[0] + b[2] - a[0] - a[2]) / 2
    dy = (b[1] + b[3] - a[1] - a[3]) / 2
    gap_x = max(0, a[0] - b[2], b[0] - a[2])
    gap_y = max(0, a[1] - b[3], b[1] - a[3])
    return {"left_of": dx >= .06, "right_of": dx <= -.06, "above": dy >= .06, "below": dy <= -.06,
            "near": math.hypot(gap_x, gap_y) <= .10}[kind]


def check_constraint(condition, regions, detections):
    def detected(phrase):
        label = canonical_label(phrase)
        return [{"query": phrase, "score": row["score"], "box": row["box"], "source": "detection",
                 "supported": True, "model": "Grounding DINO"}
                for row in detections or () if row["label"] == label and row["score"] >= DETECTION_SCORE]

    def localized(phrase):
        match = regions.get(phrase)
        if match and match["supported"]:
            return match
        return max(detected(phrase), key=lambda item: item["score"], default=None)

    result = {**condition.model_dump(), "status": "unknown", "reason": "undetermined", "evidence": []}
    if condition.kind == "absent" or condition.kind in COUNT_CONDITIONS:
        if detections is None or not canonical_label(condition.subject):
            result["reason"] = "detector_unavailable"
            return result
        items = distinct_detections(detected(condition.subject))
        result["evidence"] = items[:5]
        if condition.kind == "absent":
            result.update(status="violated" if items else "unknown",
                          reason="excluded_object_detected" if items else "absence_unconfirmed")
        else:
            number = len(items)
            result["observed_count"] = number
            if condition.kind == "at_least" and number >= condition.count:
                result.update(status="met", reason="minimum_observed")
            elif condition.kind != "at_least" and number > condition.count:
                result.update(status="violated", reason="count_exceeded")
            else:
                result["reason"] = "count_unconfirmed"
    elif condition.kind in SPATIAL_CONDITIONS:
        left, right = localized(condition.subject), localized(condition.target)
        result["evidence"] = [item for item in (left, right) if item]
        if left and right and supports_position(condition.kind, left["box"], right["box"]):
            result.update(status="met", reason="spatial_match")
        else:
            result["reason"] = "spatial_unconfirmed"
    elif condition.kind == "action":
        result["reason"] = "action_unconfirmed"
    return result


def rank_checked_shortlist(index, baseline, details, conditions, candidate_k, threshold, weight, detections):
    prefix = baseline[:candidate_k]
    terms = list(dict.fromkeys([*details, *(term for c in conditions if c.kind in SPATIAL_CONDITIONS
                                           for term in (c.subject, c.target))]))
    matched = index.shortlist_matches(prefix, terms) if index and terms else {}
    checkable = len(details) + sum(c.kind not in {"action", "other"} for c in conditions)
    ranked = []
    for rank, row in enumerate(prefix, 1):
        owner = row["id"]
        regions = {match["query"]: {**match, "source": "region", "model": "RegionCLIP-RN50",
                                    "supported": match["score"] is not None and match["score"] >= threshold}
                   for match in matched.get(owner, [])}
        matches = [regions.get(term, {"query": term, "score": None, "source": "region", "supported": False})
                   for term in details]
        evidence = [*positive_checks(matches), *(check_constraint(c, regions, detections.get(owner)) for c in conditions)]
        passed = sum(check["status"] == "met" for check in evidence)
        violated = sum(check["status"] == "violated" for check in evidence)
        score = 1 / (60 + rank) + weight * passed / (60 * max(1, checkable))
        ranked.append((violated, {**row, "score": score, "clip_rank": rank, "matches": matches,
                                   "conditions": evidence, "region_evaluated": True}))
    ranked.sort(key=lambda item: (item[0], -item[1]["score"]))
    return [row for _, row in ranked] + [
        {**row, "clip_rank": rank} for rank, row in enumerate(baseline[candidate_k:], candidate_k + 1)
    ]
