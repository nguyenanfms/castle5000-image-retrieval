"""QAService: orchestrates selected_image and retrieval_qa modes, admission queue, caching, and policy."""
from __future__ import annotations

import asyncio
import hashlib
import logging
import threading
import time
import unicodedata
import uuid
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Any

from .config import Settings
from .metadata import MetadataStore
from .qa_client import (
    BaseQAClient,
    ModelOutput,
    QAClientError,
    QAInvalidOutputError,
    QATimeoutError,
    QAUnavailableError,
    create_qa_client,
)
from .qa_evidence import EvidenceError, EvidenceLoader, LoadedImageBundle
from .qa_policy import AnswerPolicy, PolicyDecision, _reason_message
from .qa_query import QAQueryPlanner
from .schemas import QAMode, QARequest, QAResponse, QAStatusResponse, QAVersions
from .search import SearchEngine

logger = logging.getLogger("retrieval.qa")


class QAError(Exception):
    def __init__(self, status: int, code: str, message: str, headers: dict[str, str] | None = None):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.headers = headers or {}


class QABusyError(QAError):
    def __init__(self, message: str = "Hệ thống hỏi đáp đang bận, vui lòng thử lại sau"):
        super().__init__(503, "qa_busy", message)


class QAModeDisabledError(QAError):
    def __init__(self, message: str = "Chế độ hỏi đáp này hiện chưa được bật"):
        super().__init__(409, "qa_mode_disabled", message)


class QAUnknownImageError(QAError):
    def __init__(self, message: str = "Ảnh không tồn tại hoặc không khả dụng"):
        super().__init__(404, "unknown_image", message)


class _AdmissionController:
    """Bounded concurrency + bounded FIFO waiting queue with atomic slot transfer."""

    def __init__(self, concurrency: int, queue_limit: int):
        self._concurrency = max(1, int(concurrency))
        self._queue_limit = max(0, int(queue_limit))
        self._active = 0
        self._waiters: deque[asyncio.Future[None]] = deque()

    @property
    def active_count(self) -> int:
        return self._active

    @property
    def waiting_count(self) -> int:
        return len(self._waiters)

    async def acquire(self, timeout_s: float) -> None:
        if self._active < self._concurrency and not self._waiters:
            self._active += 1
            return

        if len(self._waiters) >= self._queue_limit:
            raise QABusyError("Hàng đợi suy luận đã đầy, vui lòng thử lại sau")

        if timeout_s <= 0:
            raise QABusyError("Quá thời gian chờ trong hàng đợi suy luận")

        loop = asyncio.get_running_loop()
        fut: asyncio.Future[None] = loop.create_future()
        self._waiters.append(fut)
        try:
            await asyncio.wait_for(fut, timeout=timeout_s)
        except asyncio.TimeoutError as exc:
            if fut in self._waiters:
                self._waiters.remove(fut)
            elif fut.done() and not fut.cancelled():
                self.release()
            raise QABusyError("Quá thời gian chờ trong hàng đợi suy luận, vui lòng thử lại") from exc
        except asyncio.CancelledError:
            if fut in self._waiters:
                self._waiters.remove(fut)
            elif fut.done() and not fut.cancelled():
                self.release()
            raise

    def release(self) -> None:
        while self._waiters:
            next_fut = self._waiters.popleft()
            if not next_fut.done():
                next_fut.set_result(None)
                return
        self._active = max(0, self._active - 1)


@dataclass(frozen=True)
class _CachedQAResult:
    created_at: float
    decision: PolicyDecision
    searched_count: int
    examined_count: int
    versions: QAVersions


class _QACache:
    """Thread-safe in-memory LRU + TTL cache for validated QA business responses (no image bytes)."""

    def __init__(self, max_size: int, ttl_s: float):
        self._max_size = max(0, int(max_size))
        self._ttl_s = max(0.0, float(ttl_s))
        self._lock = threading.Lock()
        self._store: OrderedDict[tuple[Any, ...], _CachedQAResult] = OrderedDict()

    def get(self, key: tuple[Any, ...]) -> _CachedQAResult | None:
        if self._max_size <= 0 or self._ttl_s <= 0:
            return None
        now = time.monotonic()
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            if now - entry.created_at > self._ttl_s:
                self._store.pop(key, None)
                return None
            self._store.move_to_end(key)
            return entry

    def put(self, key: tuple[Any, ...], value: _CachedQAResult) -> None:
        if self._max_size <= 0 or self._ttl_s <= 0:
            return
        with self._lock:
            self._store[key] = value
            self._store.move_to_end(key)
            while len(self._store) > self._max_size:
                self._store.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


def compute_corpus_fingerprint(settings: Settings, search_engine: SearchEngine) -> str:
    """Compute a content-aware fingerprint over corpus IDs, FAISS index stats, and metadata file."""
    hasher = hashlib.sha256()
    hasher.update(json.dumps(search_engine.vector_index.ids, separators=(",", ":")).encode("utf-8"))
    for path in (settings.index_path, settings.metadata_path):
        try:
            st = path.stat()
            hasher.update(f"{path.name}:{st.st_size}:{st.st_mtime_ns}".encode("utf-8"))
        except OSError:
            hasher.update(f"{path.name}:missing".encode("utf-8"))
    return hasher.hexdigest()[:16]


import json  # noqa: E402


class QAService:
    """Coordinates single-image and retrieval-based Question Answering."""

    def __init__(
        self,
        settings: Settings,
        search_engine: SearchEngine,
        metadata_store: MetadataStore | None = None,
        client: BaseQAClient | None = None,
        evidence_loader: EvidenceLoader | None = None,
        policy: AnswerPolicy | None = None,
        query_planner: QAQueryPlanner | None = None,
    ):
        self.settings = settings
        self.search_engine = search_engine
        from .qa_reasoning import ReasoningQA
        self.reasoning = ReasoningQA(self)
        self.metadata_store = metadata_store
        self.client = client if client is not None else create_qa_client(settings)
        self.evidence_loader = (
            evidence_loader
            if evidence_loader is not None
            else EvidenceLoader(
                images_dir=settings.images_dir,
                has_id_fn=search_engine.has_id,
                metadata_store=metadata_store,
                max_side=settings.qa_image_max_side,
                max_crops=settings.qa_max_crops,
            )
        )
        self.policy = policy if policy is not None else AnswerPolicy(ground_crops=False)
        self.query_planner = query_planner if query_planner is not None else QAQueryPlanner()
        self.admission = _AdmissionController(
            concurrency=settings.qa_concurrency,
            queue_limit=settings.qa_queue_limit,
        )
        self.cache = _QACache(
            max_size=settings.qa_cache_size,
            ttl_s=settings.qa_cache_ttl_s,
        )
        self.corpus_version = compute_corpus_fingerprint(settings, search_engine)

    async def aclose(self) -> None:
        await self.client.aclose()

    @property
    def is_ready(self) -> bool:
        return bool(self.settings.qa_enabled and self.client.is_ready())

    def supported_modes(self) -> list[QAMode]:
        if not self.settings.qa_enabled:
            return []
        modes: list[QAMode] = ["selected_image"]
        if self.settings.qa_retrieval_enabled:
            modes.append("retrieval_qa")
            if hasattr(self.client, "structured"):
                modes.append("auto")
        return modes

    def status_info(self) -> QAStatusResponse:
        return QAStatusResponse(
            enabled=bool(self.settings.qa_enabled),
            ready=self.is_ready,
            supported_modes=self.supported_modes(),
            max_question_length=self.settings.qa_question_max_len,
        )

    def _current_versions(self) -> QAVersions:
        region_ver = (
            self.search_engine.region_index.version
            if self.search_engine.region_index is not None
            else None
        )
        return QAVersions(
            model=self.client.model_version,
            prompt=self.settings.qa_prompt_version,
            corpus=self.corpus_version,
            region=region_ver,
        )

    def _make_cache_key(
        self,
        req: QARequest,
        candidate_ids: tuple[str, ...],
        versions: QAVersions,
        access_scope: str,
    ) -> tuple[Any, ...]:
        norm_q = unicodedata.normalize("NFC", req.question).strip()
        norm_rq = unicodedata.normalize("NFC", req.retrieval_query).strip() if req.retrieval_query else ""
        return (
            req.mode,
            norm_q,
            req.answer_language,
            norm_rq,
            candidate_ids,
            versions.corpus,
            versions.region,
            versions.model,
            versions.prompt,
            self.settings.qa_policy_version,
            self.settings.qa_retrieval_strategy,
            access_scope,
        )

    def _build_response(
        self,
        request_id: str,
        decision: PolicyDecision,
        searched_count: int,
        examined_count: int,
        took_ms: int,
        cache_hit: bool,
        versions: QAVersions,
        partial: bool = False,
    ) -> QAResponse:
        return QAResponse(
            request_id=request_id,
            status=decision.status,
            answer=decision.answer,
            reason_code=decision.reason_code,
            message=decision.message,
            evidence=decision.evidence,
            alternatives=decision.alternatives,
            note=decision.note,
            note_evidence=decision.note_evidence,
            partial=partial,
            searched_count=searched_count,
            examined_count=examined_count,
            took_ms=took_ms,
            cache_hit=cache_hit,
            versions=versions,
        )

    def _regions_for_image(self, image_id: str) -> list[dict] | None:
        reg_idx = self.search_engine.region_index
        if reg_idx is None:
            return None
        return [r for r in reg_idx.records if r.get("id") == image_id]

    async def answer(self, req: QARequest, access_scope: str = "participant") -> QAResponse:
        t0 = time.perf_counter()
        deadline = time.monotonic() + min(float(self.settings.qa_request_timeout_s), 60.0) if req.mode == "auto" else time.monotonic() + float(self.settings.qa_request_timeout_s)
        request_id = str(uuid.uuid4())

        if not self.settings.qa_enabled or not self.client.is_ready():
            raise QAError(503, "qa_unavailable", "Chức năng hỏi đáp hiện chưa sẵn sàng")

        if req.mode in {"auto", "retrieval_qa"} and not self.settings.qa_retrieval_enabled:
            raise QAModeDisabledError("Chế độ tìm ảnh rồi hỏi đáp (retrieval_qa) hiện đang tắt")

        versions = self._current_versions()

        try:
            if req.mode == "auto":
                if not hasattr(self.client, "structured"):
                    raise QAError(409, "qa_provider_unsupported", "Chế độ tự động yêu cầu Gemini hỗ trợ đầu ra có cấu trúc")
                resp = await self.reasoning.answer(req, access_scope, request_id, deadline, versions)
            elif req.mode == "selected_image":
                resp = await self._answer_selected_image(
                    req=req,
                    request_id=request_id,
                    t0=t0,
                    deadline=deadline,
                    versions=versions,
                    access_scope=access_scope,
                )
            else:
                resp = await self._answer_retrieval_qa(
                    req=req,
                    request_id=request_id,
                    t0=t0,
                    deadline=deadline,
                    versions=versions,
                    access_scope=access_scope,
                )
        except QAError:
            raise
        except EvidenceError as exc:
            raise QAError(exc.status, exc.code, exc.message) from exc
        except QAClientError as exc:
            logger.warning('qa_api_failed request_id=%s mode=%s code=%s http=%d elapsed_ms=%d model=%s', request_id, req.mode, exc.code, exc.status, int((time.perf_counter()-t0)*1000), self.client.model_version)
            raise QAError(exc.status, exc.code, exc.message, headers={'X-QA-Request-ID': request_id}) from exc

        logger.info(
            "qa_request request_id=%s mode=%s status=%s reason=%s searched=%d examined=%d partial=%s cache_hit=%s took_ms=%d model=%s prompt=%s",
            resp.request_id,
            req.mode,
            resp.status,
            resp.reason_code,
            resp.searched_count,
            resp.examined_count,
            resp.partial,
            resp.cache_hit,
            resp.took_ms,
            versions.model,
            versions.prompt,
        )
        return resp

    async def _answer_selected_image(
        self,
        req: QARequest,
        request_id: str,
        t0: float,
        deadline: float,
        versions: QAVersions,
        access_scope: str,
    ) -> QAResponse:
        assert req.image_id is not None
        image_id = req.image_id

        # Verify allowlist and file existence before checking cache or acquiring queue slot
        self.evidence_loader.resolve_safe_path(image_id)

        cache_key = self._make_cache_key(req, (image_id,), versions, access_scope)
        cached = self.cache.get(cache_key)
        if cached is not None:
            took_ms = max(0, int((time.perf_counter() - t0) * 1000))
            return self._build_response(
                request_id=request_id,
                decision=cached.decision,
                searched_count=cached.searched_count,
                examined_count=cached.examined_count,
                took_ms=took_ms,
                cache_hit=True,
                versions=cached.versions,
                partial=False,
            )

        scope_decision = self.policy.check_question_scope(
            req.question, req.answer_language, default_image_id=image_id
        )
        if scope_decision is not None:
            took_ms = max(0, int((time.perf_counter() - t0) * 1000))
            self.cache.put(
                cache_key,
                _CachedQAResult(
                    created_at=time.monotonic(),
                    decision=scope_decision,
                    searched_count=1,
                    examined_count=1,
                    versions=versions,
                ),
            )
            return self._build_response(
                request_id=request_id,
                decision=scope_decision,
                searched_count=1,
                examined_count=1,
                took_ms=took_ms,
                cache_hit=False,
                versions=versions,
                partial=False,
            )

        queue_timeout = min(float(self.settings.qa_queue_timeout_s), max(0.0, deadline - time.monotonic()))
        await self.admission.acquire(queue_timeout)
        try:
            if deadline - time.monotonic() <= 0:
                raise QATimeoutError()

            regions = self._regions_for_image(image_id)
            bundle = await asyncio.to_thread(
                self.evidence_loader.load_for_image,
                image_id,
                req.question,
                "e1",
                regions,
                True,
            )
            model_out = await self.client.answer(
                question=req.question,
                language=req.answer_language,
                evidence=bundle.items,
                deadline=deadline,
            )
            decision = self.policy.evaluate_single(model_out, bundle, req.answer_language)
        finally:
            self.admission.release()

        took_ms = max(0, int((time.perf_counter() - t0) * 1000))
        self.cache.put(
            cache_key,
            _CachedQAResult(
                created_at=time.monotonic(),
                decision=decision,
                searched_count=1,
                examined_count=1,
                versions=versions,
            ),
        )
        return self._build_response(
            request_id=request_id,
            decision=decision,
            searched_count=1,
            examined_count=1,
            took_ms=took_ms,
            cache_hit=False,
            versions=versions,
            partial=False,
        )

    async def _answer_retrieval_qa(
        self,
        req: QARequest,
        request_id: str,
        t0: float,
        deadline: float,
        versions: QAVersions,
        access_scope: str,
    ) -> QAResponse:
        scope_decision = self.policy.check_question_scope(
            req.question, req.answer_language, default_image_id=None
        )
        if scope_decision is not None:
            took_ms = max(0, int((time.perf_counter() - t0) * 1000))
            return self._build_response(
                request_id=request_id,
                decision=scope_decision,
                searched_count=0,
                examined_count=0,
                took_ms=took_ms,
                cache_hit=False,
                versions=versions,
                partial=False,
            )

        plan = self.query_planner.plan(req.question, req.retrieval_query)
        if not plan.sufficient_context:
            decision = PolicyDecision(
                status="ambiguous",
                answer=None,
                reason_code="missing_context",
                message=_reason_message("missing_context", req.answer_language),
                evidence=[],
                alternatives=[],
            )
            took_ms = max(0, int((time.perf_counter() - t0) * 1000))
            return self._build_response(
                request_id=request_id,
                decision=decision,
                searched_count=0,
                examined_count=0,
                took_ms=took_ms,
                cache_hit=False,
                versions=versions,
                partial=False,
            )

        search_results, _ = await self.search_engine.search(
            plan.search_query,
            self.settings.qa_retrieval_k,
            plan.detail_queries,
        )
        # Deduplicate IDs while preserving rank order
        unique_ids: list[str] = []
        seen_ids: set[str] = set()
        for item in search_results:
            cid = item["id"]
            if cid not in seen_ids and self.search_engine.has_id(cid):
                seen_ids.add(cid)
                unique_ids.append(cid)

        searched_count = len(unique_ids)
        candidate_ids = tuple(unique_ids[: max(1, int(self.settings.qa_max_images))])

        if not candidate_ids:
            decision = PolicyDecision(
                status="insufficient_evidence",
                answer=None,
                reason_code="no_relevant_image",
                message=_reason_message("no_relevant_image", req.answer_language),
                evidence=[],
                alternatives=[],
            )
            took_ms = max(0, int((time.perf_counter() - t0) * 1000))
            return self._build_response(
                request_id=request_id,
                decision=decision,
                searched_count=0,
                examined_count=0,
                took_ms=took_ms,
                cache_hit=False,
                versions=versions,
                partial=False,
            )

        cache_key = self._make_cache_key(req, candidate_ids, versions, access_scope)
        cached = self.cache.get(cache_key)
        if cached is not None:
            took_ms = max(0, int((time.perf_counter() - t0) * 1000))
            return self._build_response(
                request_id=request_id,
                decision=cached.decision,
                searched_count=cached.searched_count,
                examined_count=cached.examined_count,
                took_ms=took_ms,
                cache_hit=True,
                versions=cached.versions,
                partial=False,
            )

        queue_timeout = min(float(self.settings.qa_queue_timeout_s), max(0.0, deadline - time.monotonic()))
        await self.admission.acquire(queue_timeout)
        evaluated: list[tuple[LoadedImageBundle, ModelOutput]] = []
        primary: tuple[LoadedImageBundle, ModelOutput] | None = None
        confirmations: list[tuple[LoadedImageBundle, ModelOutput]] = []
        use_early_stop = self.settings.qa_retrieval_strategy == "verified_early_stop"
        confirm_target = max(0, int(self.settings.qa_confirm_candidates))
        partial = False
        try:
            for idx, cid in enumerate(candidate_ids, start=1):
                if deadline - time.monotonic() <= 0:
                    if not evaluated:
                        raise QATimeoutError()
                    partial = True
                    break

                try:
                    bundle = await asyncio.to_thread(
                        self.evidence_loader.load_for_image,
                        cid,
                        req.question,
                        f"e{idx}",
                        None,
                        False,
                    )
                except EvidenceError:
                    continue

                try:
                    out = await self.client.answer(
                        question=req.question,
                        language=req.answer_language,
                        evidence=bundle.items,
                        deadline=deadline,
                    )
                except QATimeoutError:
                    if not evaluated:
                        raise
                    partial = True
                    break

                evaluated.append((bundle, out))
                if use_early_stop:
                    if primary is None:
                        if self.policy.is_confident_answer(out):
                            primary = (bundle, out)
                            if confirm_target == 0:
                                break
                    else:
                        confirmations.append((bundle, out))
                        if len(confirmations) >= confirm_target:
                            break
        finally:
            self.admission.release()

        examined_count = len(evaluated)
        if use_early_stop:
            decision = self.policy.evaluate_early_stop(
                primary=primary,
                confirmations=confirmations,
                evaluated=evaluated,
                language=req.answer_language,
                partial=partial,
            )
        else:
            decision = self.policy.evaluate_candidates(
                evaluated=evaluated,
                language=req.answer_language,
                partial=partial,
            )
        took_ms = max(0, int((time.perf_counter() - t0) * 1000))

        # Never cache partial results
        if not partial:
            self.cache.put(
                cache_key,
                _CachedQAResult(
                    created_at=time.monotonic(),
                    decision=decision,
                    searched_count=searched_count,
                    examined_count=examined_count,
                    versions=versions,
                ),
            )

        return self._build_response(
            request_id=request_id,
            decision=decision,
            searched_count=searched_count,
            examined_count=examined_count,
            took_ms=took_ms,
            cache_hit=False,
            versions=versions,
            partial=partial,
        )
