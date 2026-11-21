from types import SimpleNamespace

import pytest

from app.qa_aggregation import AggregationStep, aggregate_observations
from app.qa_store import EvidenceStore


def observation(image_id, count, visibility='complete'):
    return SimpleNamespace(image_id=image_id, context_match='yes',
                           count_visibility=visibility, regions=[object()] * count)


def test_per_image_counts_are_computed_and_keep_provenance():
    rows = [observation('a.jpg', 2), observation('b.jpg', 3)]
    steps = [AggregationStep(operation='max', field='visible_count'),
             AggregationStep(operation='compare', field='visible_count', value='b.jpg', other_image_id='a.jpg'),
             AggregationStep(operation='aggregate', field='visible_count')]
    maximum, comparison, total = aggregate_observations(rows, steps)
    assert maximum['value'] == 3 and maximum['image_ids'] == ['a.jpg', 'b.jpg']
    assert comparison['value']['difference'] == 1
    assert total['value'] == 5 and 'not distinct' in total['semantics']


def test_partial_counts_and_identity_dedup_are_rejected():
    with pytest.raises(ValueError):
        aggregate_observations([observation('a', 2, 'partial')],
                               [AggregationStep(operation='max', field='visible_count')])
    with pytest.raises(ValueError):
        aggregate_observations([observation('a', 2)],
                               [AggregationStep(operation='distinct', field='visible_count')])


def test_filter_group_and_record_count_keep_image_ids():
    rows = [observation('a', 2), observation('b', 2), observation('c', 1)]
    result = aggregate_observations(rows, [AggregationStep(operation='filter', field='visible_count', value=2),
                          AggregationStep(operation='group_by', field='visible_count'),
                          AggregationStep(operation='count')])
    assert result[0]['value'][0]['image_ids'] == ['a', 'b']
    assert result[1]['value'] == 2


def test_cache_is_owner_scoped_and_invalidates_changed_image(tmp_path):
    image = tmp_path / 'a.jpg'
    image.write_bytes(b'first')
    store = EvidenceStore(2, 60)
    key = store.key('owner-a', {'question': 'q'}, {'model': 'm'})
    other = store.key('owner-b', {'question': 'q'}, {'model': 'm'})
    store.put(key, {'observations': []}, {'answer': 'a'}, tmp_path, ['a.jpg'])
    assert store.get(key, tmp_path)['response']['answer'] == 'a'
    assert store.get(other, tmp_path) is None
    image.write_bytes(b'different length')
    assert store.get(key, tmp_path) is None


def test_cache_refuses_paths_outside_image_directory(tmp_path):
    assert EvidenceStore.image_signature(tmp_path, ['../other.jpg']) is None
