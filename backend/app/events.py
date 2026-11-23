"""Frames of one event: Castle5000 frames are named HH_second.jpg and taken ~7 s apart, so frames of the same
hour a few seconds apart show the same moment. DRES accepts any frame of the event, so search ranks events:
a frame borrows part of its neighbours' score (one blurred or crowded frame does not hide the event) and only
the best frame of each event is listed (the others are one click away in the neighbour strip)."""
from __future__ import annotations

import numpy as np


def frame_key(image_id: str):
    try:
        hour, second = image_id.rsplit(".", 1)[0].split("_")
        return int(hour), int(second)
    except ValueError:
        return None


def neighbour_lists(ids: list[str], window_s: int) -> list[np.ndarray]:
    """For each frame, the indices of other frames of the same hour within window_s seconds."""
    by_hour: dict[int, list[tuple[int, int]]] = {}
    for i, image_id in enumerate(ids):
        key = frame_key(image_id)
        if key is not None:
            by_hour.setdefault(key[0], []).append((key[1], i))
    out = [np.empty(0, dtype=np.int64) for _ in ids]
    for frames in by_hour.values():
        frames.sort()
        seconds = np.array([s for s, _ in frames])
        index = np.array([i for _, i in frames])
        for pos, (second, i) in enumerate(frames):
            lo = np.searchsorted(seconds, second - window_s, side="left")
            hi = np.searchsorted(seconds, second + window_s, side="right")
            near = index[lo:hi]
            out[i] = near[near != i]
    return out


def smooth(scores: np.ndarray, neighbours: list[np.ndarray], share: float) -> np.ndarray:
    """score + share x (best neighbour score - score) when a neighbour scores higher."""
    if share <= 0:
        return scores
    best = scores.copy()
    for i, near in enumerate(neighbours):
        if near.size:
            best[i] = max(scores[i], float(scores[near].max()))
    return scores + share * (best - scores)


def one_per_event(order, ids: list[str], neighbours: list[np.ndarray], k: int) -> list[int]:
    """The first k indices of `order` (best first), skipping frames next to an already listed frame."""
    taken: set[int] = set()
    picked: list[int] = []
    for i in order:
        i = int(i)
        if i in taken:
            continue
        picked.append(i)
        taken.add(i)
        taken.update(int(j) for j in neighbours[i])
        if len(picked) >= k:
            break
    return picked
