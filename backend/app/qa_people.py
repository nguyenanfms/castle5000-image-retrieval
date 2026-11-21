"""Corpus coverage and reviewed cross-image correspondences for counting people."""

import json
import logging
import io

import numpy as np
from PIL import Image

from pydantic import BaseModel, ConfigDict, Field

from .schemas import QAClaim, QARegion
from .regions import iou
from .search_conditions import DETECTION_SCORE, distinct_detections
from .search_conditions import canonical_label


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


def _prioritize_observations(observations):
    """Seed matching with the richest observed frame, with stable tie-breaking."""
    return sorted((o for o in observations if o.context_match == 'yes'),
                  key=lambda o: (-len(o.regions), str(o.image_id)))


def regionclip_object_regions(index, image_id, category):
    """Return unverified, deduplicated index proposals for the requested class."""
    if index is None or not hasattr(index, 'rows_by_image') or not hasattr(index, 'records'):
        return []
    category = canonical_label(category)
    if not category:
        return []
    candidates = []
    for row in index.rows_by_image.get(image_id, ()):
        record = index.records[int(row)]
        labels = record.get('objects', [])
        label = canonical_label(labels[0].get('label', '')) if labels else record.get('kind', '')
        if label != category:
            continue
        try:
            region = QARegion(label='candidate', box=record['box'])
        except (KeyError, ValueError):
            continue
        candidates.append((float(labels[0].get('score', 0)) if labels else 0.0,
                           int(row), region))
    regions = []
    for _, row, region in sorted(candidates, key=lambda x: (-x[0], x[1])):
        if any(iou(region.box, previous.box) >= .7 for previous in regions):
            continue
        regions.append(region.model_copy(update={'label': f'R{len(regions) + 1}'}))
    return regions


def prepare_object_crops(observations, evidence, max_people=120):
    """Crop normalized boxes from the same oriented image shown to the VLM."""
    keys, crops = [], []
    for observation in _prioritize_observations(observations):
        if observation.image_id not in evidence:
            continue
        try:
            with Image.open(io.BytesIO(evidence[observation.image_id].image_bytes)) as raw:
                image = raw.convert('RGB')
                w, h = image.size
                for index, region in enumerate(observation.regions):
                    if len(crops) >= max_people:
                        return keys, crops
                    x1, y1, x2, y2 = region.box
                    left, top = int(x1*w), int(y1*h)
                    right, bottom = max(left+1, int(np.ceil(x2*w))), max(top+1, int(np.ceil(y2*h)))
                    crop = image.crop((left, top, min(right, w), min(bottom, h)))
                    if crop.width and crop.height:
                        keys.append((observation.image_id, index))
                        crops.append(crop)
        except (OSError, ValueError):
            continue
    return keys, crops


def estimate_distinct_people(observations, region_index, threshold, *, max_people=120, crop_vectors=None, category='person'):
    """Cluster observed people for an explicitly provisional visual estimate.

    CLIP scores are not calibrated identity probabilities. A match against any
    member joins that group; two boxes from the same image never share a group.
    Regions without a matched proposal remain unassigned rather than guessed.
    """
    if crop_vectors is None and (region_index is None or not hasattr(region_index, 'rows_by_image')):
        return None
    groups = []
    unassigned = 0
    total = 0
    matched = 0
    for observation in _prioritize_observations(observations):
        available = ([int(i) for i in region_index.rows_by_image.get(observation.image_id, ())
                      if region_index.records[int(i)].get('kind') == 'person'] if crop_vectors is None else [])
        for region_number, region in enumerate(observation.regions):
            total += 1
            if matched >= max_people:
                unassigned += 1
                continue
            index = None
            if crop_vectors is not None:
                raw_vector = crop_vectors.get((observation.image_id, region_number))
                if raw_vector is None:
                    unassigned += 1
                    continue
                vector = np.asarray(raw_vector, dtype=np.float32)
            else:
                overlaps = [(iou(region.box, region_index.records[i]['box']), i) for i in available]
                if not overlaps:
                    unassigned += 1
                    continue
                overlap, index = max(overlaps)
                if overlap < .3:
                    unassigned += 1
                    continue
                vector = np.asarray(region_index.vectors[index], dtype=np.float32)
            norm = float(np.linalg.norm(vector))
            if not np.isfinite(norm) or norm < 1e-8:
                unassigned += 1
                continue
            if index is not None:
                available.remove(index)
            vector = vector / norm
            member = {'image_id': observation.image_id, 'label': region.label, 'box': list(region.box),
                      'region_id': f'{region_index.version}:{index}' if index is not None else None,
                      'sequence': matched + 1, '_vector': vector}
            options = []
            comparisons = []
            for group_number, group in enumerate(groups, 1):
                blocked = any(item['image_id'] == observation.image_id for item in group)
                scores = [float(vector @ item['_vector']) for item in group]
                for item, score in zip(group, scores):
                    if np.isfinite(score):
                        comparisons.append({'image_id': item['image_id'], 'label': item['label'],
                                            'box': item['box'], 'group_id': f'G{group_number}',
                                            'similarity': round(score, 6), 'eligible': not blocked,
                                            'above_threshold': score >= threshold,
                                            'sequence': item['sequence']})
                if blocked:
                    continue
                if any(np.isfinite(score) and score >= threshold for score in scores):
                    options.append((max(scores), group_number, group))
            if options:
                _, selected_group, group = max(options, key=lambda item: item[0])
                group.append(member)
                decision = 'matched'
            else:
                groups.append([member])
                selected_group = len(groups)
                decision = 'first' if matched == 0 else 'new_group'
            member['comparison'] = {'decision': decision, 'group_id': f'G{selected_group}',
                                    'comparisons': comparisons}
            matched += 1
    if not matched:
        return None
    return {'status': 'experimental', 'category': category, 'embedding_model': 'CLIP crop' if crop_vectors is not None else 'RegionCLIP proposal',
            'threshold': threshold, 'estimated_count': len(groups),
            'observed_regions': total, 'assigned_regions': matched, 'unassigned_regions': unassigned,
            'examined_images': len({o.image_id for o in observations}),
            'groups': [{'group_id': f'G{i + 1}',
                        'members': [{k: v for k, v in member.items() if k != '_vector'} for member in group]}
                       for i, group in enumerate(groups)]}


def candidate_person_pairs(observations, region_index, *, limit=5, max_people=120):
    """Rank possible cross-image matches for review, never as verified identities.

    RegionCLIP was not calibrated for re-identification. IoU only associates a
    Gemini box with an existing person proposal; cosine is a ranking signal.
    """
    if limit <= 0 or max_people <= 0 or region_index is None or not hasattr(region_index, 'rows_by_image'):
        return []
    candidates = []
    for observation in observations:
        if observation.context_match != 'yes':
            continue
        rows = region_index.rows_by_image.get(observation.image_id, ())
        rows = [int(i) for i in rows if region_index.records[int(i)].get('kind') == 'person']
        used_rows = set()
        for region in observation.regions:
            overlaps = [(iou(region.box, region_index.records[i]['box']), i) for i in rows if i not in used_rows]
            if not overlaps:
                continue
            overlap, index = max(overlaps)
            if overlap < .3:
                continue
            vec = np.asarray(region_index.vectors[index], dtype=np.float32)
            norm = float(np.linalg.norm(vec))
            if not np.isfinite(norm) or norm < 1e-8:
                continue
            used_rows.add(index)
            candidates.append({'image_id': observation.image_id, 'label': region.label,
                               'box': list(region.box), 'region_id': f'{region_index.version}:{index}',
                               'vector': vec / norm})
            if len(candidates) >= max_people:
                break
        if len(candidates) >= max_people:
            break
    pairs = []
    for i, first in enumerate(candidates):
        for second in candidates[i + 1:]:
            if first['image_id'] == second['image_id']:
                continue
            score = float(first['vector'] @ second['vector'])
            if not np.isfinite(score):
                continue
            pairs.append((score, first, second))
    pairs.sort(key=lambda item: -item[0])
    output = []
    used = set()
    for score, first, second in pairs:
        keys = ((first['image_id'], first['label']), (second['image_id'], second['label']))
        if any(key in used for key in keys):
            continue
        used.update(keys)
        output.append({'first': {k: v for k, v in first.items() if k != 'vector'},
                       'second': {k: v for k, v in second.items() if k != 'vector'},
                       'similarity': round(score, 4), 'status': 'unverified'})
        if len(output) >= max(0, min(int(limit), 20)):
            break
    return output
