"""Possible duplicates in a people roster: the same person listed twice from two different frames.

Two signals, no extra model call:
- re-ID: OSNet-AIN embeddings of the person crops (reid_osnet.py), trained to match one person across
  cameras and poses by clothing and build;
- attributes: gender, colours, garment, pattern, head camera or cap and glasses read from each person's
  appearance text (Vietnamese or English), which states the distinguishing details directly.
People in the same frame are different people and are never compared.
"""
from __future__ import annotations

import re
import unicodedata

# phrase -> attribute; longest phrases first so 'xanh dương' wins over 'xanh'
_COLOURS = {
    "xanh dương": "blue", "xanh lam": "blue", "xanh nước biển": "blue", "xanh đậm": "navy", "xanh navy": "navy",
    "navy": "navy", "blue": "blue", "xanh lá": "green", "green": "green", "xám": "grey", "ghi": "grey",
    "grey": "grey", "gray": "grey", "trắng": "white", "white": "white", "đen": "black", "black": "black",
    "đỏ": "red", "red": "red", "kem": "cream", "be": "cream", "cream": "cream", "beige": "cream", "nâu": "brown",
    "brown": "brown", "hồng": "pink", "pink": "pink", "vàng": "yellow", "yellow": "yellow", "cam": "orange",
    "orange": "orange", "tím": "purple", "purple": "purple",
}
_GARMENTS = {
    "áo sơ mi": "shirt", "sơ mi": "shirt", "shirt": "shirt", "áo len": "sweater", "sweater": "sweater",
    "jumper": "sweater", "hoodie": "hoodie", "có mũ trùm": "hoodie", "áo khoác có mũ": "hoodie", "hooded": "hoodie",
    "áo khoác": "jacket", "jacket": "jacket", "polo": "polo", "áo thun": "tshirt", "áo phông": "tshirt",
    "t-shirt": "tshirt", "tshirt": "tshirt", "áo đấu": "jersey", "jersey": "jersey", "áo thể thao": "jersey",
    "tạp dề": "apron", "apron": "apron", "váy": "skirt", "dress": "dress", "skirt": "skirt",
}
_PATTERNS = {"kẻ sọc": "plaid", "ca rô": "plaid", "caro": "plaid", "kẻ ô": "plaid", "plaid": "plaid",
             "checkered": "plaid", "sọc ngang": "striped", "kẻ ngang": "striped", "striped": "striped",
             "stripes": "striped", "họa tiết": "pattern", "logo": "logo", "in chữ": "logo"}
_HEADCAM = ("camera", "đèn gắn đầu", "đèn đầu", "đèn đeo", "head-mounted", "headlamp", "head camera")
_CAP = ("đội mũ", "mũ lưỡi trai", " cap", "baseball cap", "beanie")
_GLASSES = ("kính", "glasses", "spectacles")


def _norm(text):
    return " " + unicodedata.normalize("NFC", (text or "").lower()) + " "


def _found(text, table):
    """Values of the phrases present in text, matching whole words, longest phrases first."""
    out, rest = set(), text
    for phrase in sorted(table, key=len, reverse=True):
        pattern = r"(?<![\w])" + re.escape(phrase) + r"(?![\w])"
        if re.search(pattern, rest):
            out.add(table[phrase])
            rest = re.sub(pattern, " ", rest)
    return out


def attributes(text):
    t = _norm(text)
    gender = ("F" if re.search(r"(?<![\w])(nữ|phụ nữ|cô gái|woman|female|girl)(?![\w])", t)
              else "M" if re.search(r"(?<![\w])(nam|đàn ông|man|male|boy|chàng trai)(?![\w])", t) else None)
    return {
        "gender": gender,
        "colours": _found(t, _COLOURS),
        "garments": _found(t, _GARMENTS),
        "patterns": _found(t, _PATTERNS),
        "headcam": any(k in t for k in _HEADCAM),
        "cap": any(k in t for k in _CAP),
        "glasses": any(k in t for k in _GLASSES),
    }


def _jaccard(a, b):
    return len(a & b) / len(a | b) if a and b else (1.0 if not a and not b else 0.0)


def attribute_similarity(a, b):
    """0..1. Different genders are different people (0)."""
    if a["gender"] and b["gender"] and a["gender"] != b["gender"]:
        return 0.0
    colour = _jaccard(a["colours"], b["colours"])
    garment = _jaccard(a["garments"], b["garments"])
    pattern = 1.0 if a["patterns"] == b["patterns"] else 0.5 if (a["patterns"] & b["patterns"]) else 0.0
    head = 1.0 if (a["headcam"] == b["headcam"] and a["cap"] == b["cap"]) else 0.5
    return round(0.35 * colour + 0.3 * garment + 0.2 * pattern + 0.15 * head, 3)


def suspected_duplicates(entries, vectors, min_score=0.6, top=8):
    """Pairs of people from different frames that may be one person, most likely first.

    entries: roster people (label, appearance, image_id); vectors: their re-ID embeddings (None when the person
    has no crop). score = 0.6 x re-ID cosine + 0.4 x attribute similarity."""
    attrs = [attributes(e.get("appearance", "")) for e in entries]
    pairs = []
    for i in range(len(entries)):
        for j in range(i + 1, len(entries)):
            if entries[i]["image_id"] == entries[j]["image_id"]:
                continue  # one frame never shows the same person twice
            attr = attribute_similarity(attrs[i], attrs[j])
            if attr == 0.0:
                continue
            reid = float(vectors[i] @ vectors[j]) if vectors[i] is not None and vectors[j] is not None else None
            score = 0.6 * reid + 0.4 * attr if reid is not None else attr * 0.8
            if score >= min_score:
                pairs.append(dict(a=entries[i]["label"], b=entries[j]["label"], reid=None if reid is None else round(reid, 3),
                                  attributes=attr, score=round(score, 3)))
    return sorted(pairs, key=lambda p: -p["score"])[:top]
