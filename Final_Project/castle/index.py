import hashlib
import json
import os
import time
import uuid
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .catalog import scan, chronological
from .model import ClipEncoder
from .partner_metadata import load_metadata_matcher


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def build(cfg, encoder_factory=ClipEncoder):
    started = time.perf_counter()
    records = scan(cfg)
    usable = [r for r in records if not r["excluded"]]
    if not usable:
        raise ValueError("Tat ca anh deu bi exclude.")
    model_tag = hashlib.sha256(json.dumps(cfg.model_id, sort_keys=True).encode()).hexdigest()[:16]
    cache = cfg.artifacts_dir / "features" / model_tag
    thumbs = cfg.artifacts_dir / "thumbs"
    cache.mkdir(parents=True, exist_ok=True)
    thumbs.mkdir(parents=True, exist_ok=True)
    errors, vectors, indexed, pending = [], [], [], []
    print(f"Scan: {len(records)} anh; exclude: {len(records)-len(usable)}. Tao thumbnail...", flush=True)
    for position, r in enumerate(usable, 1):
        thumb = thumbs / (r["fingerprint"] + ".jpg")
        try:
            # Verify the source even when its feature is cached.
            with Image.open(cfg.data_dir / r["path"]) as original:
                im = ImageOps.exif_transpose(original).convert("RGB")
                if not thumb.exists():
                    im.thumbnail((400, 300))
                    im.save(thumb, "JPEG", quality=85)
            feature = cache / (r["fingerprint"] + ".npy")
            if feature.exists():
                vector = np.load(feature, allow_pickle=False)
                if vector.ndim != 1 or not np.isfinite(vector).all() or np.linalg.norm(vector) < 1e-8:
                    raise ValueError("Cache vector khong hop le; xoa file feature nay va build lai.")
                vectors.append(vector)
                indexed.append(r)
            else:
                pending.append(r)
        except Exception as exc:
            errors.append({"path": r["path"], "error": str(exc)})
        if position % 1000 == 0 or position == len(usable):
            print(f"Thumbnail/source check: {position}/{len(usable)}; cached: {len(indexed)}; pending: {len(pending)}; errors: {len(errors)}", flush=True)
    encoder = encoder_factory(cfg) if pending else None
    batch_size = max(1, int(cfg.batch_size))
    i = 0
    while i < len(pending):
        chunk = pending[i:i+batch_size]
        images, valid = [], []
        for r in chunk:
            try:
                with Image.open(cfg.data_dir / r["path"]) as im:
                    images.append(ImageOps.exif_transpose(im).convert("RGB"))
                valid.append(r)
            except Exception as exc:
                errors.append({"path": r["path"], "error": str(exc)})
        try:
            feats = encoder.images(images) if images else []
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and batch_size > 1:
                batch_size = max(1, batch_size // 2)
                encoder.clear_oom()
                print(f"GPU thieu VRAM; giam batch_size xuong {batch_size} va thu lai.", flush=True)
                continue
            raise
        finally:
            for im in images:
                im.close()
        if len(feats) != len(valid):
            raise ValueError("So vector khong khop so anh.")
        for r, feature in zip(valid, feats):
            target = cache / (r["fingerprint"] + ".npy")
            temp = target.with_suffix(".tmp")
            with temp.open("wb") as f:
                np.save(f, feature, allow_pickle=False)
            os.replace(temp, target)
            vectors.append(feature)
            indexed.append(r)
        i += len(chunk)
        print(f"Embedding: {i}/{len(pending)}; cached: {len(usable)-len(pending)-len(errors)}", flush=True)
    if not indexed:
        raise ValueError("Khong co anh doc duoc. Kiem tra build_errors.json va du lieu.")
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("Embedding matrix khong hop le.")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms < 1e-8):
        raise ValueError("Embedding vector bang 0.")
    matrix /= norms
    # Publish one complete generation. CURRENT changes only after all files exist.
    generation = uuid.uuid4().hex
    folder = cfg.artifacts_dir / "indexes" / generation
    folder.mkdir(parents=True)
    np.save(folder / "embeddings.npy", matrix, allow_pickle=False)
    stats = {"scanned": len(records), "indexed": len(indexed), "excluded": len(records)-len(usable),
             "errors": errors, "dimension": matrix.shape[1], "model": cfg.model_id,
             "elapsed_s": round(time.perf_counter()-started, 2),
             "device": encoder.device if encoder else "cache", "generation": generation}
    atomic_json(folder / "manifest.json", {"model": cfg.model_id, "item_prefix": cfg.item_prefix,
                                          "records": indexed, "stats": stats})
    atomic_json(cfg.artifacts_dir / "build_report.json", stats)
    atomic_json(cfg.artifacts_dir / "CURRENT.json", {"generation": generation})
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return stats


class SearchIndex:
    def __init__(self, cfg, encoder_factory=ClipEncoder):
        self.cfg = cfg
        self.encoder_factory = encoder_factory
        self.encoder = None
        self.model_lock = threading.Lock()
        current = json.loads((cfg.artifacts_dir / "CURRENT.json").read_text(encoding="utf-8"))
        generation = current["generation"]
        if not generation.isalnum():
            raise ValueError("Generation khong hop le.")
        folder = cfg.artifacts_dir / "indexes" / generation
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        if manifest["model"] != cfg.model_id:
            raise ValueError("Model trong config da doi. Chay build lai truoc khi search.")
        if manifest.get("item_prefix", cfg.item_prefix) != cfg.item_prefix:
            raise ValueError("item_prefix da doi. Chay build lai de cap nhat ma DRES.")
        self.records = manifest["records"]
        self.stats = manifest["stats"]
        self.matrix = np.load(folder / "embeddings.npy", allow_pickle=False)
        if self.matrix.ndim != 2 or len(self.matrix) != len(self.records) or not np.isfinite(self.matrix).all():
            raise ValueError("Index hong hoac manifest khong khop. Chay build lai.")
        self.lookup = {r["key"]: i for i, r in enumerate(self.records)}
        self.ordered = sorted(self.records, key=chronological)
        self.metadata_matcher = None
        self.metadata_error = None
        try:
            self.metadata_matcher = load_metadata_matcher(cfg, self.records)
        except (OSError, ImportError, SyntaxError, ValueError, KeyError, AttributeError, TypeError) as exc:
            self.metadata_error = str(exc)[:230]

    @lru_cache(maxsize=128)
    def query_vector(self, query):
        if self.encoder is None:
            with self.model_lock:
                if self.encoder is None:
                    self.encoder = self.encoder_factory(self.cfg)
        return self.encoder.text(query)

    def record(self, key):
        if key not in self.lookup:
            raise KeyError("Khong tim thay anh trong index.")
        return self.records[self.lookup[key]]

    @staticmethod
    def public(record, score=None):
        result = {k: record[k] for k in ("key", "filename", "batch", "hour", "frame", "item_name", "ambiguous_item", "description")}
        result["score"] = float(score) if score is not None else None
        result["thumb"] = f"/api/image/{record['key']}?thumb=true"
        return result

    def effective_mode(self, mode, query, image_key=None):
        if mode == "hybrid" and query.strip() and not image_key and self.metadata_matcher is not None:
            return "hybrid"
        return "clip"

    def search(self, query="", k=60, batch="", hour=None, image_key=None, mode="clip"):
        indices = [i for i, r in enumerate(self.records)
                   if (not batch or r["batch"] == batch) and (hour is None or r["hour"] == hour)]
        if not indices:
            return []
        if image_key:
            vector = self.matrix[self.lookup[self.record(image_key)["key"]]]
        elif query.strip():
            vector = np.asarray(self.query_vector(query.strip()), dtype=np.float32)
        else:
            filtered = {self.records[i]["key"] for i in indices}
            return [self.public(r) for r in self.ordered if r["key"] in filtered][:k]
        if vector.shape != (self.matrix.shape[1],) or not np.isfinite(vector).all():
            raise ValueError("Query embedding khong khop index.")
        vector = vector / max(float(np.linalg.norm(vector)), 1e-12)
        if self.effective_mode(mode, query, image_key) == "hybrid":
            clip_scores = self.matrix @ vector
            bonus = self.metadata_matcher.bonus(query, clip_scores, self.cfg.metadata_weight)
            scores = (clip_scores + bonus)[indices]
        else:
            scores = self.matrix[indices] @ vector
        order = np.argsort(-scores, kind="stable")[:k]
        return [self.public(self.records[indices[j]], scores[j]) for j in order]

    def neighbors(self, key, radius=6):
        current = self.record(key)
        if current["frame"] is None:
            return [self.public(current)]
        # Cross Batch boundaries only for the same source and hour with unique frames.
        group = [r for r in self.ordered if r["source"] == current["source"] and r["hour"] == current["hour"]]
        if len({r["frame"] for r in group}) != len(group):
            group = [r for r in group if r["batch"] == current["batch"]]
        pos = next(i for i, r in enumerate(group) if r["key"] == key)
        return [self.public(r) for r in group[max(0, pos-radius):pos+radius+1]]
