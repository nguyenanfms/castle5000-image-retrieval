import time

from app.ondemand_detector import OnDemandDetector
from app.search_conditions import canonical_label


def test_only_requested_images_and_labels_are_scanned_and_cached(tmp_path, monkeypatch):
    for name in ('a.jpg', 'b.jpg', 'unused.jpg'):
        (tmp_path / name).write_bytes(b'image')
    detector = OnDemandDetector(tmp_path, tmp_path)
    calls = []
    def detect(path, label):
        calls.append((path.name, label))
        return [{'label': label, 'score': .9, 'box': [.1, .1, .5, .9]}]
    monkeypatch.setattr(detector, '_detect', detect)
    first = detector.detect_many(['a.jpg', 'b.jpg'], ['person'], time.monotonic() + 5)
    second = detector.detect_many(['a.jpg'], ['person'], time.monotonic() + 5)
    assert list(first) == ['a.jpg', 'b.jpg'] and list(second) == ['a.jpg']
    assert calls == [('a.jpg', 'person'), ('b.jpg', 'person')]
    assert detector.detect_many(['unused.jpg'], ['person'], time.monotonic() - 1) == {}
    assert canonical_label('backpack') == 'backpack'


def test_detector_failure_is_unchecked_not_an_empty_detection(tmp_path, monkeypatch):
    (tmp_path / 'a.jpg').write_bytes(b'image')
    detector = OnDemandDetector(tmp_path, tmp_path)
    def fail(*args):
        raise RuntimeError('model unavailable')
    monkeypatch.setattr(detector, '_detect', fail)
    assert detector.detect_many(['a.jpg'], ['person'], time.monotonic() + 5) == {}


def test_kis_does_not_treat_unscanned_candidates_as_empty():
    import asyncio
    from types import SimpleNamespace
    from app.search import SearchEngine
    from app.schemas import SearchCondition
    engine = SearchEngine.__new__(SearchEngine)
    engine.settings = SimpleNamespace(search_candidate_k=3, search_region_threshold=.2,
        search_region_weight=.25, detection_max_images=2, detection_timeout_s=5, search_detection_enabled=True)
    engine.region_index = None
    engine.vector_index = SimpleNamespace(num_images=3, search_vector=lambda vec, k: [
        {'id': 'a.jpg', 'score': .9}, {'id': 'b.jpg', 'score': .8}, {'id': 'c.jpg', 'score': .7}])
    engine._cache_get = lambda key: None
    engine._cache_put = lambda *args: None
    async def encode(query):
        return None
    calls = []
    async def detect(ids, labels, deadline):
        calls.append((ids, labels))
        return {'a.jpg': [{'label': 'person', 'score': .9, 'box': [.1, .1, .5, .9]}], 'b.jpg': []}
    engine.encode_text = encode
    engine.detect_candidates = detect
    condition = SearchCondition(kind='at_least', subject='person', count=1)
    results, _, _, _, checked = asyncio.run(engine.search_demo('kitchen', 3, [], 'clip_region', [condition]))
    assert checked == 2 and calls == [(['a.jpg', 'b.jpg'], {'person'})]
    states = {r['id']: r['conditions'][0]['reason'] for r in results}
    assert states['a.jpg'] == 'minimum_observed'
    assert states['b.jpg'] == 'count_unconfirmed'
    assert states['c.jpg'] == 'detector_unavailable'


def test_search_conditions_skip_detector_unless_enabled():
    import asyncio
    from types import SimpleNamespace
    from app.search import SearchEngine
    from app.schemas import SearchCondition
    engine = SearchEngine.__new__(SearchEngine)
    engine.settings = SimpleNamespace(search_candidate_k=3, search_region_threshold=.2,
        search_region_weight=.25, detection_max_images=2, detection_timeout_s=5, search_detection_enabled=False)
    engine.region_index = None
    engine.vector_index = SimpleNamespace(num_images=3, search_vector=lambda vec, k: [
        {'id': 'a.jpg', 'score': .9}, {'id': 'b.jpg', 'score': .8}, {'id': 'c.jpg', 'score': .7}])
    engine._cache_get = lambda key: None
    engine._cache_put = lambda *args: None
    async def encode(query):
        return None
    async def detect(ids, labels, deadline):
        raise AssertionError('Grounding DINO must not run in search when disabled')
    engine.encode_text = encode
    engine.detect_candidates = detect
    condition = SearchCondition(kind='at_least', subject='person', count=1)
    results, _, mode, _, checked = asyncio.run(engine.search_demo('kitchen', 3, [], 'clip_region', [condition]))
    assert mode == 'clip_region' and [r['id'] for r in results] == ['a.jpg', 'b.jpg', 'c.jpg']
    assert {r['conditions'][0]['reason'] for r in results} == {'detector_unavailable'}


def test_precomputed_person_boxes_skip_the_model(tmp_path, monkeypatch):
    for name in ('a.jpg', 'b.jpg'):
        (tmp_path / name).write_bytes(b'image')
    detector = OnDemandDetector(tmp_path, tmp_path)
    stored = [{'label': 'person', 'score': .8, 'box': [.1, .1, .4, .9]}]
    detector.precomputed['person'] = {'a.jpg': stored, 'b.jpg': []}
    monkeypatch.setattr(detector, '_detect', lambda *args: (_ for _ in ()).throw(AssertionError('model ran')))
    # Even with the lock held elsewhere (deadline already past), stored images answer at once.
    assert detector.detect_many(['a.jpg', 'b.jpg'], ['person'], time.monotonic() - 1) == {'a.jpg': stored, 'b.jpg': []}


def test_person_crop_store_round_trip(tmp_path):
    import hashlib, json
    import numpy as np
    from app.person_crops import SCHEMA, PersonCrops, qa_label_boxes as qa_person_boxes
    ids = ['a.jpg', 'b.jpg']
    rows = [{'label': 'person', 'score': .9, 'box': [.1, .1, .4, .9]},
            {'label': 'person', 'score': .85, 'box': [.11, .1, .41, .9]},   # same person: merged
            {'label': 'person', 'score': .4, 'box': [.6, .1, .9, .9]}]      # below DETECTION_SCORE
    boxes = qa_person_boxes(rows)
    assert boxes == [[.1, .1, .4, .9]]
    (tmp_path / 'detections.json').write_text(json.dumps({'person': {'a.jpg': rows}, 'cup': {'a.jpg': []}}))
    (tmp_path / 'crops.json').write_text(json.dumps([{'id': 'a.jpg', 'label': 'person', 'box': boxes[0]}]))
    np.save(tmp_path / 'vectors.npy', np.asarray([[0.6, 0.8]], dtype=np.float16))
    manifest = dict(schema=SCHEMA, labels=['person', 'cup'], embed_model='m', embed_pretrained='p', num_images=1, num_crops=1,
                    ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(), complete=False)
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    store = PersonCrops.load(tmp_path, ids, 'm', 'p')
    assert store.vector_for('a.jpg', [.1, .1, .4, .9]) is not None
    assert store.vector_for('a.jpg', [.5, .1, .9, .9]) is None
    assert store.detections['cup'] == {'a.jpg': []}
    import pytest
    with pytest.raises(ValueError):
        PersonCrops.load(tmp_path, ids, 'other-model', 'p')
