"""Bounded, owner-isolated QA traces and validated answer cache.

Image bytes and credentials are never stored. Entries expire on a monotonic
clock; changing an examined image invalidates a cached answer.
"""
from __future__ import annotations

import json
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any


class EvidenceStore:
    def __init__(self, capacity: int = 128, ttl: float = 600):
        self.capacity = max(0, capacity)
        self.ttl = max(0.0, ttl)
        self.entries: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.traces: OrderedDict[tuple[str, str], dict[str, Any]] = OrderedDict()

    def save_trace(self, owner: str, request_id: str, trace: dict) -> None:
        if not self.capacity or not self.ttl:
            return
        now = time.monotonic()
        for key, entry in list(self.traces.items()):
            if entry['expires'] <= now:
                del self.traces[key]
        self.traces[owner, request_id] = {'expires': now + self.ttl, 'trace': trace}
        while len(self.traces) > self.capacity:
            self.traces.popitem(last=False)

    @staticmethod
    def key(owner: str, request: dict, versions: dict) -> str:
        return json.dumps([owner, request, versions], sort_keys=True, ensure_ascii=False,
                          separators=(',', ':'), default=str)

    @staticmethod
    def image_signature(image_dir: Path, ids: list[str]) -> dict[str, tuple[int, int]] | None:
        signatures = {}
        base = image_dir.resolve()
        try:
            for image_id in ids:
                path = (base / image_id).resolve()
                if path.parent != base or not path.is_file():
                    return None
                stat = path.stat()
                signatures[image_id] = (stat.st_size, stat.st_mtime_ns)
        except OSError:
            return None
        return signatures

    def put(self, key: str, trace: dict, response: dict, image_dir: Path, ids: list[str]) -> None:
        if not self.capacity or not self.ttl or not ids:
            return
        signature = self.image_signature(image_dir, ids)
        if signature is None:
            return
        self.entries[key] = {'expires': time.monotonic() + self.ttl, 'trace': trace,
                             'response': response, 'signature': signature}
        self.entries.move_to_end(key)
        while len(self.entries) > self.capacity:
            self.entries.popitem(last=False)

    def get(self, key: str, image_dir: Path) -> dict | None:
        entry = self.entries.get(key)
        if entry is None:
            return None
        if (entry['expires'] <= time.monotonic() or
                self.image_signature(image_dir, list(entry['signature'])) != entry['signature']):
            self.entries.pop(key, None)
            return None
        self.entries.move_to_end(key)
        return entry
