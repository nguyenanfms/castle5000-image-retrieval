"""Grounding DINO on retrieved images only; cached in memory by image and label."""
from collections import OrderedDict
import logging
import threading
import time
from pathlib import Path

from .search_conditions import canonical_label

log = logging.getLogger(__name__)


class OnDemandDetector:
    def __init__(self, images_dir, model_dir, *, cache_size=2048):
        self.images_dir = Path(images_dir).resolve()
        self.model_dir = Path(model_dir)
        self.cache_size = cache_size
        self.cache = OrderedDict()
        self.lock = threading.Lock()
        self.processor = self.model = self.device = None

    def _load(self):
        if self.model is not None:
            return
        import torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Never download model weights as a side effect of a user query.
        processor = AutoProcessor.from_pretrained(str(self.model_dir), local_files_only=True)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(
            str(self.model_dir), local_files_only=True, disable_custom_kernels=True).to(self.device).eval()
        self.processor, self.model = processor, model

    def _detect(self, path, label):
        import torch
        from PIL import Image, ImageOps
        self._load()
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert('RGB')
        inputs = self.processor(images=image, text=label + '.', return_tensors='pt').to(self.device)
        with torch.inference_mode():
            output = self.model(**inputs)
        result = self.processor.post_process_grounded_object_detection(
            output, inputs.input_ids, threshold=.3, text_threshold=.25,
            target_sizes=[(image.height, image.width)])[0]
        rows = []
        for box, score in zip(result['boxes'], result['scores']):
            x1, y1, x2, y2 = (float(x) for x in box)
            coords = [max(0., min(1., v)) for v in
                      (x1 / image.width, y1 / image.height, x2 / image.width, y2 / image.height)]
            if coords[0] < coords[2] and coords[1] < coords[3]:
                rows.append({'box': coords, 'label': label, 'score': float(score)})
        return rows

    def detect_many(self, image_ids, labels, deadline):
        """Missing image keys mean unchecked, never an empty detection result."""
        labels = tuple(sorted({canonical_label(label) for label in labels} - {None}))
        if not labels:
            return {}
        results = {}
        if not self.lock.acquire(timeout=max(0., deadline - time.monotonic())):
            return results
        try:
            for image_id in image_ids:
                if time.monotonic() >= deadline:
                    break
                path = (self.images_dir / image_id).resolve()
                if path.parent != self.images_dir or not path.is_file():
                    continue
                try:
                    stat = path.stat()
                    entries = []
                    complete = True
                    for label in labels:
                        key = (image_id, stat.st_mtime_ns, stat.st_size, label)
                        if key not in self.cache:
                            if time.monotonic() >= deadline:
                                complete = False
                                break
                            self.cache[key] = self._detect(path, label)
                            while len(self.cache) > self.cache_size:
                                self.cache.popitem(last=False)
                        self.cache.move_to_end(key)
                        entries.extend(self.cache[key])
                    if complete:
                        results[image_id] = entries
                except (OSError, RuntimeError, ImportError, ValueError) as exc:
                    log.warning('grounding_dino_unavailable image_id=%s error=%s', image_id, type(exc).__name__)
                    break
        finally:
            self.lock.release()
        return results
