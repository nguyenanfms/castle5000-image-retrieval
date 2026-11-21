from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path

# torch and faiss each bundle their own OpenMP runtime; on macOS loading both
# in one process aborts at import time ("OMP: Error #15"), and merely
# silencing that abort (KMP_DUPLICATE_LIB_OK alone) still segfaults once real
# work runs on both pools concurrently. Forcing everything to 1 OpenMP thread
# avoids that -- cheap here since we only encode one short query string per
# request; MPS/GPU work (used offline in tools/build_index.py) is unaffected.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import faiss
import numpy as np
import open_clip
import torch

faiss.omp_set_num_threads(1)
torch.set_num_threads(1)

from .config import Settings
from .regions import RegionIndex
from .regionclip import RegionClipIndex
from .search_conditions import canonical_label, positive_checks, rank_checked_shortlist
from .ondemand_detector import OnDemandDetector


class IndexNotBuiltError(RuntimeError):
    pass


class VectorIndex:
    """Pure top-k search over a FAISS flat index + parallel id table.

    Kept separate from the CLIP model so the top-k logic (order, k-limit,
    unknown-id lookups) can be unit tested without downloading any model
    weights -- see backend/tests/test_search.py.
    """

    def __init__(self, index: faiss.Index, ids: list[str]):
        if index.ntotal != len(ids):
            raise IndexNotBuiltError(
                f"Index has {index.ntotal} vectors but {len(ids)} ids were given"
            )
        self.index = index
        self.ids = ids
        self.id_set = set(ids)

    @classmethod
    def load(cls, index_path: Path, ids_path: Path) -> "VectorIndex":
        if not index_path.exists() or not ids_path.exists():
            raise IndexNotBuiltError(
                f"Missing index at {index_path} or ids at {ids_path}. "
                "Run `python tools/build_index.py` first."
            )
        ids = json.loads(ids_path.read_text())
        index = faiss.read_index(str(index_path))
        return cls(index, ids)

    @property
    def num_images(self) -> int:
        return len(self.ids)

    def has_id(self, image_id: str) -> bool:
        return image_id in self.id_set

    def search_vector(self, vec: np.ndarray, k: int) -> list[dict]:
        k = min(k, self.num_images)
        if k <= 0:
            return []
        scores, idxs = self.index.search(vec, k)
        results: list[dict] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0:
                continue
            results.append({"id": self.ids[idx], "score": float(score)})
        return results


class SearchEngine:
    """Loads the FAISS index + CLIP text encoder once and serves text->image search.

    Encoding a query is the expensive part, so concurrent encodes are bounded
    by a semaphore (per technical note section 4/8) instead of letting 40
    simultaneous requests fight over CPU with DRES.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._cache_lock = threading.Lock()
        self._cache: OrderedDict[tuple, tuple[list[dict], int | None]] = OrderedDict()
        self._semaphore = asyncio.Semaphore(settings.encode_concurrency)
        self.vector_index = VectorIndex.load(settings.index_path, settings.ids_path)
        self.condition_detections, self.condition_detector_version = {}, None
        self.detector = OnDemandDetector(settings.images_dir, settings.detection_model_dir,
                                         threads=settings.detection_threads)
        self.region_index = None
        self.region_notice = 'Chỉ mục vùng chưa sẵn sàng; điều kiện chi tiết dùng toàn ảnh làm dự phòng.'
        try:
            self.region_index = self._load_regions(settings.region_index_dir)
            self.region_notice = None
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            logging.getLogger('retrieval').warning('Region index unavailable: %s', exc)
        self._global_vectors = self.vector_index.index.reconstruct_n(0, self.num_images)
        self._load_model()

    def _load_regions(self, path):
        manifest=json.loads((path/'manifest.json').read_text())
        if manifest.get('schema')=='regionclip-v1':
            return RegionClipIndex.load(path,self.vector_index.ids)
        return RegionIndex.load(path,self.vector_index.ids,self.settings.embed_model_name,self.settings.embed_pretrained,self.vector_index.index.d)

    def person_scores(self):
        """Per image, the highest RegionCLIP similarity to "person"; computed once per region index."""
        index = self.region_index
        if not isinstance(index, RegionClipIndex):
            return None
        cached = getattr(self, '_person_scores', None)
        if cached is not None and cached[0] == index.version:
            return cached[1]
        best = np.full(len(self.vector_index.ids), -1.0, dtype=np.float32)
        np.maximum.at(best, index.owners, index.vectors @ index.encode(['person'])[0])
        scores = dict(zip(self.vector_index.ids, best.tolist()))
        self._person_scores = (index.version, scores)
        return scores

    def refresh_regions(self):
        """Activate a completed build without disrupting whole-image search."""
        if self.region_index is not None:
            return
        path = self.settings.region_index_dir
        if not (path / 'manifest.json').exists():
            return
        try:
            loaded = self._load_regions(path)
        except (OSError, ValueError, KeyError, RuntimeError):
            return
        self.region_index = loaded
        self.region_notice = None
        with self._cache_lock:
            self._cache.clear()

    def _load_model(self) -> None:
        s = self.settings
        self.device = ("cuda" if torch.cuda.is_available()
                       else "mps" if torch.backends.mps.is_available() else "cpu")
        self.model, _, self._image_preprocess = open_clip.create_model_and_transforms(
            s.embed_model_name, pretrained=s.embed_pretrained, device=self.device
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(s.embed_model_name)

    @property
    def num_images(self) -> int:
        return self.vector_index.num_images

    def has_id(self, image_id: str) -> bool:
        return self.vector_index.has_id(image_id)

    def _encode_text_sync(self, query: str) -> np.ndarray:
        tokens = self.tokenizer([query] if isinstance(query, str) else query).to(self.device)
        with torch.no_grad():
            feats = self.model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return np.ascontiguousarray(feats.cpu().numpy().astype("float32"))

    async def encode_text(self, query: str) -> np.ndarray:
        async with self._semaphore:
            return await asyncio.to_thread(self._encode_text_sync, query)

    def _encode_image_crops_sync(self, crops):
        vectors = []
        for start in range(0, len(crops), 16):
            batch = torch.stack([self._image_preprocess(image) for image in crops[start:start + 16]]).to(self.device)
            with torch.no_grad():
                features = self.model.encode_image(batch)
                features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            vectors.append(features.cpu().numpy().astype('float32'))
        return np.concatenate(vectors) if vectors else np.empty((0, 0), dtype='float32')

    async def encode_image_crops(self, crops):
        async with self._semaphore:
            return await asyncio.to_thread(self._encode_image_crops_sync, crops)

    async def detect_candidates(self, image_ids, labels, deadline):
        return await asyncio.to_thread(self.detector.detect_many, image_ids, labels, deadline)

    def _cache_get(self, key: tuple) -> tuple[list[dict], int | None] | None:
        with self._cache_lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        return None

    def _cache_put(self, key: tuple, value: tuple[list[dict], int | None]) -> None:
        with self._cache_lock:
            self._cache[key] = value
            self._cache.move_to_end(key)
            while len(self._cache) > self.settings.search_cache_size:
                self._cache.popitem(last=False)

    async def retrieve_shortlist(self, query, k, details, candidate_k, threshold, weight, context_weight=0.0):
        index = self.region_index
        return await self._retrieve_shortlist(index, query, k, details, candidate_k, threshold, weight, context_weight)

    async def _retrieve_shortlist(self, index, query, k, details, candidate_k, threshold, weight, context_weight=0.0):
        vec = await self.encode_text(query)
        if not details or not isinstance(index, RegionClipIndex):
            return self.vector_index.search_vector(vec, k)
        baseline = self.vector_index.search_vector(vec, max(k, candidate_k))
        ranked = await asyncio.to_thread(index.rank_shortlist, baseline, details,
                                         candidate_k, threshold, weight, context_weight)
        return ranked[:k]

    async def search_demo(self, query, k, details, mode, conditions=()):
        conditions = tuple(dict.fromkeys(conditions))
        index = self.region_index if isinstance(self.region_index, RegionClipIndex) else None
        candidate_k = min(self.settings.search_candidate_k, self.num_images)
        threshold = self.settings.search_region_threshold
        weight = self.settings.search_region_weight
        version = index.version if index else None
        # Grounding DINO in search is opt-in: without it, conditions use RegionCLIP regions only.
        use_detector = bool(conditions) and getattr(self.settings, 'search_detection_enabled', False)
        detector_version = 'grounding-dino-on-demand-v1' if use_detector else ('regions-only' if conditions else None)
        actual_mode = 'clip_region' if mode == 'clip_region' and (conditions or (details and index)) else 'clip'
        key = ('demo', query, k, tuple(details), conditions, mode, version, detector_version,
               candidate_k, threshold, weight)
        cached = self._cache_get(key)
        if cached is not None:
            results, checked = cached
            return results, 0, actual_mode, version, checked
        started = time.perf_counter()
        checked = None
        if actual_mode == 'clip_region' and conditions:
            baseline = self.vector_index.search_vector(await self.encode_text(query), self.num_images)
            labels = {canonical_label(value) for c in conditions
                      if c.kind not in {'action', 'other'} for value in (c.subject, c.target) if value} if use_detector else set()
            labels.discard(None)
            limit = min(len(baseline), self.settings.detection_max_images)
            detections = await self.detect_candidates([row['id'] for row in baseline[:limit]], labels,
                time.monotonic() + self.settings.detection_timeout_s) if labels else {}
            ranked, examined = await asyncio.to_thread(
                rank_checked_shortlist, index, baseline, details, conditions, candidate_k, threshold, weight,
                detections, target_k=k, person_scores=None if use_detector else self.person_scores(),
            )
            checked = len(detections) if use_detector else examined
        elif actual_mode == 'clip_region':
            ranked = await self._retrieve_shortlist(index, query, k, details, candidate_k, threshold, weight)
            checked = candidate_k
        else:
            ranked = self.vector_index.search_vector(await self.encode_text(query), k)
        results = []
        for rank, row in enumerate(ranked, 1):
            matches = [{**match, 'source': 'region', 'model': 'RegionCLIP-RN50'} for match in row.get('matches', [])]
            checks = row.get('conditions') if 'conditions' in row else positive_checks(matches)
            results.append({**row, 'final_rank': rank, 'clip_rank': row.get('clip_rank', rank),
                            'region_evaluated': row.get('region_evaluated', False),
                            'matches': matches, 'conditions': checks})
        # An interrupted scan must be retried, not cached as complete.
        if not (actual_mode == 'clip_region' and conditions and labels and checked < limit):
            self._cache_put(key, (results, checked))
        return results, int((time.perf_counter() - started) * 1000), actual_mode, version, checked

    async def search(self, query: str, k: int, detail_queries: list[str] | None = None) -> tuple[list[dict], int]:
        details = tuple(detail_queries or [])
        version = self.region_index.version if self.region_index else 'global-fallback'
        key = (query, k, details, version)
        cached = self._cache_get(key)
        if cached is not None:
            return cached[0], 0

        t0 = time.perf_counter()
        if not details:
            vec = await self.encode_text(query)
            results = self.vector_index.search_vector(vec, k)
        elif isinstance(self.region_index,RegionClipIndex):
            baseline=query or '. '.join(details)
            vec=await self.encode_text(baseline)
            scores=self._global_vectors@vec[0]
            results=await asyncio.to_thread(self.region_index.rank_fused,list(details),scores,self.vector_index.ids,k,bool(query))
        else:
            vectors = await self.encode_text(([query] if query else []) + list(details))
            whole = vectors[0] if query else None
            detail_vectors = vectors[1:] if query else vectors
            regions = self.region_index or RegionIndex(np.empty((0, self.vector_index.index.d), dtype='float32'), [], self.vector_index.ids, version)
            results = await asyncio.to_thread(regions.rank, detail_vectors, self._global_vectors, self.vector_index.ids, k, whole, list(details))
        took_ms = int((time.perf_counter() - t0) * 1000)

        self._cache_put(key, (results, None))
        return results, took_ms
