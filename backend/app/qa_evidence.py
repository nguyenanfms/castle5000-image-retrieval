"""Evidence loader for QA: safe allowlist image loading, EXIF transpose, crops, and verified metadata."""
from __future__ import annotations

import io
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PIL import Image, ImageOps

from .metadata import ImageMetadata, MetadataStore
from .regions import iou
from .schemas import QAEVidenceSource

_COUNTING_RE = re.compile(
    r"\b(?:bao\s+nhiêu|mấy\s+\w+|đếm|số\s+lượng|how\s+many|count|number\s+of)\b",
    re.IGNORECASE,
)

_METADATA_RE = re.compile(
    r"\b(?:ngày\s+(?:mấy|nào|bao\s+nhiêu)|thứ\s+mấy|mấy\s+giờ|lúc\s+mấy\s+giờ|"
    r"thời\s+gian\s+nào|khung\s+giờ\s+nào|người\s+đeo\s+camera|chủ\s+nhân\s+camera|"
    r"what\s+day|which\s+day|what\s+time|which\s+hour|what\s+hour|whose\s+camera|camera\s+owner)\b",
    re.IGNORECASE,
)


def is_global_counting_question(question: str) -> bool:
    """Return True if the question asks to count objects across the scene."""
    return bool(_COUNTING_RE.search(question))


def is_metadata_question(question: str) -> bool:
    """Return True if the question specifically asks for provenance metadata (day/hour/owner)."""
    return bool(_METADATA_RE.search(question))


def is_valid_normalized_box(box: list[float] | tuple[float, ...] | None) -> bool:
    if box is None or len(box) != 4:
        return False
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in box):
        return False
    x1, y1, x2, y2 = (float(v) for v in box)
    return 0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0


@dataclass(frozen=True)
class EvidenceInput:
    """Single piece of visual or provenance evidence passed to QAClient with an opaque ID."""

    evidence_id: str
    image_id: str
    image_bytes: bytes
    mime_type: str = "image/jpeg"
    kind: str = "full_image"  # "full_image" or "crop"
    source: QAEVidenceSource = "image"
    box: list[float] | None = None
    width: int = 0
    height: int = 0
    metadata_text: str | None = None


@dataclass(frozen=True)
class LoadedImageBundle:
    """All evidence items prepared for a single corpus image."""

    image_id: str
    natural_width: int
    natural_height: int
    items: list[EvidenceInput]
    verified_metadata: dict[str, str] | None = None


class EvidenceError(Exception):
    def __init__(self, code: str, message: str, status: int = 404):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


class EvidenceLoader:
    """Resolves corpus image IDs from an allowlist and loads oriented, bounded JPEG bytes."""

    def __init__(
        self,
        images_dir: Path,
        has_id_fn: Callable[[str], bool],
        metadata_store: MetadataStore | None = None,
        max_side: int = 1024,
        max_crops: int = 2,
    ):
        self._images_dir = images_dir.resolve()
        self._has_id = has_id_fn
        self._metadata_store = metadata_store
        self._max_side = max(128, int(max_side))
        self._max_crops = max(0, int(max_crops))

    def resolve_safe_path(self, image_id: str) -> Path:
        if not image_id or any(sep in image_id for sep in ("/", "\\", "..", ":")):
            raise EvidenceError("unknown_image", "Ảnh không tồn tại hoặc không hợp lệ", 404)
        if not self._has_id(image_id):
            raise EvidenceError("unknown_image", "Ảnh không tồn tại trong bộ dữ liệu", 404)
        candidate = (self._images_dir / image_id).resolve()
        if candidate.parent != self._images_dir or not candidate.is_file():
            raise EvidenceError("unknown_image", "Không tìm thấy tệp ảnh trên máy chủ", 404)
        return candidate

    def _encode_jpeg(self, img: Image.Image) -> bytes:
        w, h = img.size
        if max(w, h) > self._max_side:
            scale = self._max_side / float(max(w, h))
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()

    def _extract_verified_metadata(self, image_id: str, question: str) -> tuple[str | None, dict[str, str] | None]:
        """Return strictly provenance metadata (day, hour, owner, frame) ONLY when the question asks for it.

        Detector-generated fields (description, classes, class_count) are never exposed to the VLM prompt
        to avoid benchmark leakage or treating detector guesses as ground truth.
        """
        if not self._metadata_store or not is_metadata_question(question):
            return None, None
        meta: ImageMetadata | None = self._metadata_store.get(image_id)
        if meta is None:
            return None, None
        verified = {
            "owner": str(meta.owner),
            "day": str(meta.day),
            "hour": str(meta.hour),
            "frame": str(meta.frame),
        }
        text = f"provenance_metadata(owner={meta.owner}, day={meta.day}, hour={meta.hour}, frame={meta.frame})"
        return text, verified

    def load_for_image(
        self,
        image_id: str,
        question: str,
        evidence_prefix: str = "e1",
        region_records: list[dict] | None = None,
        include_crops: bool = True,
    ) -> LoadedImageBundle:
        path = self.resolve_safe_path(image_id)
        try:
            with Image.open(path) as raw_img:
                oriented = ImageOps.exif_transpose(raw_img).convert("RGB")
        except OSError as exc:
            raise EvidenceError("unknown_image", f"Không đọc được tệp ảnh: {exc}", 404) from exc

        nat_w, nat_h = oriented.size
        full_bytes = self._encode_jpeg(oriented)
        meta_text, verified_meta = self._extract_verified_metadata(image_id, question)
        primary_source: QAEVidenceSource = "metadata" if meta_text is not None else "image"

        items: list[EvidenceInput] = [
            EvidenceInput(
                evidence_id=evidence_prefix,
                image_id=image_id,
                image_bytes=full_bytes,
                mime_type="image/jpeg",
                kind="full_image",
                source=primary_source,
                box=None,
                width=nat_w,
                height=nat_h,
                metadata_text=meta_text,
            )
        ]

        # Add at most max_crops supplementary crops for non-counting questions in single-image mode.
        if (
            include_crops
            and self._max_crops > 0
            and region_records
            and not is_global_counting_question(question)
        ):
            selected_boxes: list[list[float]] = []
            sorted_regions = sorted(
                region_records,
                key=lambda r: float(r.get("confidence", 0.0)),
                reverse=True,
            )
            for reg in sorted_regions:
                raw_box = reg.get("box")
                if not is_valid_normalized_box(raw_box):
                    continue
                norm_box = [round(float(v), 6) for v in raw_box]
                # Skip near-full-image boxes or boxes that heavily overlap already chosen crops.
                if (norm_box[2] - norm_box[0]) * (norm_box[3] - norm_box[1]) >= 0.85:
                    continue
                if any(iou(norm_box, prev) > 0.5 for prev in selected_boxes):
                    continue
                x1 = max(0, min(nat_w - 1, int(math.floor(norm_box[0] * nat_w))))
                y1 = max(0, min(nat_h - 1, int(math.floor(norm_box[1] * nat_h))))
                x2 = max(x1 + 1, min(nat_w, int(math.ceil(norm_box[2] * nat_w))))
                y2 = max(y1 + 1, min(nat_h, int(math.ceil(norm_box[3] * nat_h))))
                crop_img = oriented.crop((x1, y1, x2, y2))
                crop_bytes = self._encode_jpeg(crop_img)
                crop_idx = len(selected_boxes) + 1
                items.append(
                    EvidenceInput(
                        evidence_id=f"{evidence_prefix}_c{crop_idx}",
                        image_id=image_id,
                        image_bytes=crop_bytes,
                        mime_type="image/jpeg",
                        kind="crop",
                        source="region",
                        box=norm_box,
                        width=x2 - x1,
                        height=y2 - y1,
                        metadata_text=None,
                    )
                )
                selected_boxes.append(norm_box)
                if len(selected_boxes) >= self._max_crops:
                    break

        return LoadedImageBundle(
            image_id=image_id,
            natural_width=nat_w,
            natural_height=nat_h,
            items=items,
            verified_metadata=verified_meta,
        )
