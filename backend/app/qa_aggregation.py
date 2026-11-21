"""Small, bounded operations over observations; no generated code is executed."""
from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class AggregationStep(BaseModel):
    model_config = ConfigDict(extra='forbid')
    operation: Literal['filter', 'exclude', 'distinct', 'count', 'max', 'min',
                       'group_by', 'aggregate', 'exists', 'compare']
    field: Literal['image_id', 'context_match', 'visible_count'] | None = None
    value: str | int | None = None
    other_image_id: str | None = Field(default=None, max_length=500)


def aggregate_observations(observations, steps: list[AggregationStep]) -> list[dict]:
    """Apply validated operations to image-local rows and return source IDs.

    Counts refer to visible instances in one image; their sum is appearances,
    never distinct people or independent events. Partial counts are excluded.
    """
    rows = [{'image_id': o.image_id, 'context_match': o.context_match,
             'visible_count': len(o.regions) if o.count_visibility == 'complete' else None}
            for o in observations]
    results = []
    for step in steps:
        op, field = step.operation, step.field
        if op in {'filter', 'exclude'}:
            if field is None or step.value is None:
                raise ValueError('filter requires a field and value')
            rows = [r for r in rows if (r[field] == step.value) == (op == 'filter')]
            continue
        if op == 'distinct':
            if field != 'image_id':
                raise ValueError('distinct is supported only for image IDs')
            rows = list({row['image_id']: row for row in rows}.values())
            continue
        sources = [row['image_id'] for row in rows]
        if op == 'count':
            value = len(rows)
        elif op in {'max', 'min', 'aggregate'}:
            if field != 'visible_count' or not rows or any(r[field] is None for r in rows):
                raise ValueError('numeric aggregation needs complete per-image counts')
            values = [r[field] for r in rows]
            value = {'max': max, 'min': min, 'aggregate': sum}[op](values)
        elif op == 'exists':
            value = bool(rows)
        elif op == 'group_by':
            if field not in {'context_match', 'visible_count'}:
                raise ValueError('unsupported grouping field')
            groups = {}
            for row in rows:
                key = str(row[field])
                groups.setdefault(key, []).append(row['image_id'])
            value = [{'value': key, 'count': len(ids), 'image_ids': ids} for key, ids in groups.items()]
        elif op == 'compare':
            if field != 'visible_count' or not isinstance(step.value, str) or not step.other_image_id:
                raise ValueError('compare requires two image IDs and visible_count')
            selected = {r['image_id']: r['visible_count'] for r in rows
                        if r['image_id'] in {step.value, step.other_image_id}}
            if len(selected) != 2 or any(x is None for x in selected.values()):
                raise ValueError('comparison requires two complete counts')
            a, b = selected[step.value], selected[step.other_image_id]
            value = {'left': a, 'right': b, 'difference': a - b}
            sources = [step.value, step.other_image_id]
        else:
            raise ValueError('unsupported operation')
        results.append({'operation': op, 'field': field, 'value': value, 'image_ids': sources,
                        'semantics': 'visible appearances, not distinct identities' if op == 'aggregate' else None})
    return results
