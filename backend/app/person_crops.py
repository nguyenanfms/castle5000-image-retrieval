"""Precomputed Grounding DINO boxes and crop vectors per label (built by tools/build_person_crops.py).

Images in the store skip on-demand detection and crop encoding in QA; other images keep the
on-demand path, so a partial store is still useful.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from .regions import iou
from .search_conditions import DETECTION_SCORE, distinct_detections

SCHEMA = 'detection-crops-v2'


def qa_label_boxes(rows, label='person'):
    """The boxes QA turns into regions for one image and label: same filter, merge and cap as qa_reasoning."""
    kept = distinct_detections([row for row in rows if row['label'] == label and row['score'] >= DETECTION_SCORE])
    return [row['box'] for row in kept[:40]]


class PersonCrops:
    def __init__(self, detections, crops, vectors, manifest):
        self.detections = detections  # label -> {image_id: rows}
        self.manifest = manifest
        self.version = f"{SCHEMA}:{manifest['num_images']}:{manifest['num_crops']}"
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.rows_by_image = {}
        for row, crop in enumerate(crops):
            self.rows_by_image.setdefault(crop['id'], []).append((crop['box'], row))

    @classmethod
    def load(cls, directory, ids, embed_model, embed_pretrained):
        directory = Path(directory)
        manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
        if manifest.get('schema') != SCHEMA or not manifest.get('labels'):
            raise ValueError('Not a detection crop store')
        if manifest.get('ids_sha256') != hashlib.sha256(json.dumps(list(ids)).encode()).hexdigest():
            raise ValueError('Person crops were built for another image list')
        if manifest.get('embed_model') != embed_model or manifest.get('embed_pretrained') != embed_pretrained:
            raise ValueError('Person crop vectors come from another embedding model')
        detections = json.loads((directory / 'detections.json').read_text(encoding='utf-8'))
        crops = json.loads((directory / 'crops.json').read_text(encoding='utf-8'))
        vectors = np.load(directory / 'vectors.npy', allow_pickle=False)
        known = set(ids)
        if (len(crops) != len(vectors) or set(detections) != set(manifest['labels'])
                or any(not set(images) <= known for images in detections.values())
                or any(c['id'] not in known for c in crops)):
            raise ValueError('Invalid person crop store')
        if len(vectors) and not np.allclose(np.linalg.norm(vectors.astype(np.float32), axis=1), 1, atol=2e-2):
            raise ValueError('Invalid person crop vectors')
        return cls(detections, crops, vectors, manifest)

    def vector_for(self, image_id, box):
        """Stored vector when box is one of the stored QA boxes of that image (detector boxes match exactly).

        A crop vector depends only on the box, so a box stored under any label serves every label.
        """
        for stored, row in self.rows_by_image.get(image_id, ()):
            if iou(box, stored) >= .98:
                return self.vectors[row]
        return None


qa_person_boxes = qa_label_boxes
