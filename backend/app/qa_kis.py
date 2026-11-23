"""QA finds its scenes with the image-search (KIS) pipeline instead of a retrieval of its own.

QA works out what moment to look for; image search is the part built and checked for finding a described
moment: the search planner turns the description into an English caption, phrases and conditions, the
region search with condition filtering and metadata ranks the frames (one per event), and the verifier
checks the top frames requirement by requirement with close-ups. QA then reads the frames the verifier
confirmed first, counts or answers on them, and writes the answer.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from .query_parser import split_query

log = logging.getLogger("retrieval.qa.kis")


@dataclass
class KisResult:
    ordered: list[str]                          # confirmed, then partial, then unchecked; rejected dropped
    confirmed: set[str] = field(default_factory=set)
    rejected: set[str] = field(default_factory=set)
    plan: dict = field(default_factory=dict)
    took_ms: int = 0


class KisFinder:
    def __init__(self, planner, engine, verifier, k=50, verify_images=4, verify=True):
        self.planner, self.engine, self.verifier = planner, engine, verifier
        self.k, self.verify_images = k, verify_images
        # verify=False: QA reads the frames itself, so only the verifier's text-only metadata ranking is used
        # (one quick call) instead of its image checks and close-ups.
        self.verify = verify

    async def find(self, description: str, language: str = "vi") -> KisResult:
        started = time.perf_counter()
        query, details, conditions = description, split_query(description), ()
        try:
            plan = await self.planner.plan(description, "auto")
            query, details, conditions = plan.global_query_en, list(plan.detail_queries_en), tuple(plan.conditions)
        except Exception as exc:  # The planner is an aid: search the description as written.
            log.warning("qa_kis_plan_failed error=%s", getattr(exc, "args", [exc])[0])
        rows, *_ = await self.engine.search_demo(query, self.k, details, "clip_region", conditions)
        ids = [row["id"] for row in rows]
        confirmed, rejected, partial = set(), set(), set()
        if self.verifier is not None and ids and not self.verify and hasattr(self.verifier, "metadata_order"):
            try:
                ids, _ = await self.verifier.metadata_order(query, details, ids, max(self.verify_images * 2, 8))
            except Exception as exc:
                log.warning("qa_kis_rank_failed error=%s", getattr(exc, "message", None) or type(exc).__name__)
        elif self.verifier is not None and ids and self.verifier.ready():
            try:
                checked = await self.verifier.verify(query, details, list(conditions), ids, language,
                                                     limit=self.verify_images)
                for image_id, verdict in checked.get("verdicts", {}).items():
                    {"yes": confirmed, "partial": partial, "no": rejected}.get(verdict.get("match"), set()).add(image_id)
                # Frames the metadata ranking put first come next even when they were not checked.
                ranked = checked.get("metadata_ranked") or []
                ids = list(dict.fromkeys(ranked + ids))
            except Exception as exc:
                log.warning("qa_kis_verify_failed error=%s", getattr(exc, "message", None) or type(exc).__name__)
        order = {x: i for i, x in enumerate(ids)}
        tier = lambda x: 0 if x in confirmed else 1 if x in partial else 2
        ordered = sorted((x for x in ids if x not in rejected), key=lambda x: (tier(x), order[x]))
        result = KisResult(ordered, confirmed, rejected,
                           {"query": query, "details": details, "conditions": [c.model_dump(exclude_none=True)
                                                                                for c in conditions]},
                           int((time.perf_counter() - started) * 1000))
        log.info("qa_kis_find candidates=%d confirmed=%d rejected=%d took_ms=%d",
                 len(ordered), len(confirmed), len(rejected), result.took_ms)
        return result
