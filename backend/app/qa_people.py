"""Corpus coverage and reviewed cross-image correspondences for counting people."""

import json
import logging

from pydantic import BaseModel, ConfigDict, Field

from .schemas import QAClaim, QARegion
from .search_conditions import DETECTION_SCORE, distinct_detections


class ReviewedPeople(BaseModel):
    model_config = ConfigDict(extra='forbid')
    images: dict[str, list[QARegion]]

    @classmethod
    def load(cls, path):
        if not path.is_file():
            return None
        try:
            catalog = cls.model_validate_json(path.read_bytes())
            for regions in catalog.images.values():
                if len({region.label for region in regions}) != len(regions):
                    raise ValueError('Duplicate person label in one image')
                if len({tuple(region.box) for region in regions}) != len(regions):
                    raise ValueError('Duplicate person box in one image')
            return catalog
        except (OSError, ValueError):
            logging.getLogger(__name__).warning('Ignoring invalid reviewed QA people catalog')
            return None

    def tally(self, scope_ids, *, vi):
        if not scope_ids or not all(image_id in self.images for image_id in scope_ids):
            return None
        # Prefer a representative frame that can show several distinct people at once.
        sources = sorted(scope_ids, key=lambda image_id: -len(self.images[image_id]))
        identities = set()
        claims = []
        boxes = {}
        for image_id in sources:
            new = [region for region in self.images[image_id] if region.label not in identities]
            if not new:
                continue
            boxes[image_id] = new
            identities.update(region.label for region in new)
            labels = ', '.join(region.label for region in new)
            claims.append(QAClaim(text=f'Người đã đối chiếu: {labels}.' if vi else f'Reviewed people: {labels}.',
                                  evidence_ids=[image_id]))
        if not claims:
            claims.append(QAClaim(text='Ảnh đã kiểm duyệt không có người.' if vi else 'Reviewed images contain no people.',
                                  evidence_ids=[sources[0]]))
        return len(identities), claims, boxes


def person_shortlist(scope_ids, detections):
    """Rank promising frames for a lower bound, without dropping undetected frames from coverage."""
    indexed = [image_id for image_id in scope_ids if image_id in detections]
    counts = {}
    for image_id in indexed:
        people = [row for row in detections[image_id]
                  if row['label'] == 'person' and row['score'] >= DETECTION_SCORE]
        counts[image_id] = len(distinct_detections(people))
    candidates = sorted((image_id for image_id in indexed if counts[image_id]),
                        key=lambda image_id: -counts[image_id])
    return candidates, len(indexed)
