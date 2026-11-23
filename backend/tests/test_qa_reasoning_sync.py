"""Exercise orchestration without a network-backed Windows event loop."""
import json
import time
from types import SimpleNamespace

from app.qa_reasoning import ReasoningQA, normalize_observations, Plan, apply_distinct_scope
from app.qa_people import candidate_person_pairs, estimate_distinct_people
from app.qa_client import QATimeoutError
from app.schemas import QARegion, QARequest, QAVersions


class Admission:
    async def acquire(self, timeout):
        pass

    def release(self):
        pass


class Client:
    def __init__(self):
        self.calls = []
        self.corpus_aggregate = False
        self.verified = True
        self.distinct_people = False

    async def structured(self, prompt, images, deadline):
        task = json.loads(prompt[prompt.index('{'):])
        stage = task['stage']
        self.calls.append((stage, [item.image_id for item in images]))
        if stage == 'planning':
            if self.distinct_people:
                return {'question': task['question'], 'question_type': 'multi_image',
                        'operation': 'count_distinct_people', 'answer_scope': 'corpus',
                        'subject': 'person', 'information_needed': 'distinct people', 'queries': ['people']}
            if self.corpus_aggregate:
                return {'question': task['question'], 'question_type': 'multi_image',
                        'operation': 'describe', 'answer_scope': 'corpus', 'subject': 'person',
                        'information_needed': 'image count', 'queries': ['people'],
                        'processing_steps': [{'operation': 'count'}]}
            return {'question': task['question'], 'question_type': 'multi_image',
                    'operation': 'count_max', 'answer_scope': 'corpus', 'subject': 'person',
                    'information_needed': 'maximum', 'queries': ['people']}
        if stage == 'observing':
            return {'observations': [{'image_id': x.image_id, 'context_match': 'yes',
                      'observation': 'one person', 'count_visibility': 'complete',
                      'regions': [{'label': 'P1', 'box': [.1, .1, .4, .9]}]} for x in images]}
        if stage == 'verifying':
            return {'supported': self.verified, 'conflicts': False}
        if stage == 'synthesizing':
            count = task['aggregation_results'][0]['value']
            return {'status': 'answered', 'answer': f'{count} images',
                    'claims': [{'text': f'{count} images',
                                'evidence_ids': [x['image_id'] for x in task['observations']]}]}
        raise AssertionError(stage)


class Timeout:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


async def inline_thread(fn, *args, **kwargs):
    return fn(*args, **kwargs)


def run_inline(coro):
    try:
        result = coro.send(None)
    except StopIteration as done:
        return done.value
    raise AssertionError(f'unexpected asynchronous suspension: {result!r}')


def make_reasoner(tmp_path, monkeypatch, count, max_images):
    monkeypatch.setattr('app.qa_reasoning.asyncio.to_thread', inline_thread)
    monkeypatch.setattr('app.qa_reasoning.asyncio.timeout', lambda seconds: Timeout())
    ids = [f'{i:02}.jpg' for i in range(count)]
    for image_id in ids:
        (tmp_path / image_id).write_bytes(b'image')
    settings = SimpleNamespace(qa_sequence_path=tmp_path/'missing-sequences.json',
        qa_people_catalog_path=tmp_path/'missing-people.json', qa_cache_size=10,
        qa_cache_ttl_s=60, qa_corpus_max_images=max_images, images_dir=tmp_path)
    engine = SimpleNamespace(vector_index=SimpleNamespace(ids=ids), region_index=None,
                             has_id=lambda x: x in ids, condition_detections={})
    async def detect_candidates(image_ids, labels, deadline):
        return {image_id: [{'label': labels[0], 'score': .9, 'box': [.1, .1, .4, .9]}]
                for image_id in image_ids}
    engine.detect_candidates = detect_candidates
    def load(image_id, *args, **kwargs):
        return SimpleNamespace(items=[SimpleNamespace(image_id=image_id, evidence_id=image_id,
                                                       kind='full_image', box=None)])
    service = SimpleNamespace(settings=settings, search_engine=engine, client=Client(),
                              evidence_loader=SimpleNamespace(load_for_image=load), admission=Admission())
    return ReasoningQA(service), service, ids


def test_scan_more_than_24_and_cache_verified_answer(tmp_path, monkeypatch):
    reasoner, service, ids = make_reasoner(tmp_path, monkeypatch, 26, 30)
    req = QARequest(question='Which image has the most people?', answer_language='en')
    versions = QAVersions(model='test', prompt='test', corpus='test', region=None)
    # Gemini checked the detector proposals and reports complete counts: all 26 images tie at one.
    response = run_inline(reasoner.answer(req, 'user', 'r1', time.monotonic()+30, versions))
    assert response.reason_code == 'too_many_maximum_images'
    assert response.examined_count == len(ids) and response.status == 'insufficient_evidence'


def test_scan_limit_never_claims_full_coverage(tmp_path, monkeypatch):
    reasoner, service, ids = make_reasoner(tmp_path, monkeypatch, 26, 16)
    req = QARequest(question='Which image has the most people?', answer_language='en')
    versions = QAVersions(model='test', prompt='test', corpus='test', region=None)
    response = run_inline(reasoner.answer(req, 'user', 'r1', time.monotonic()+30, versions))
    assert response.reason_code == 'incomplete_corpus_coverage'
    assert response.examined_count == 16 and not response.examined_scope['exhaustive']


def test_verified_result_is_reused_only_for_same_owner(tmp_path, monkeypatch):
    reasoner, service, ids = make_reasoner(tmp_path, monkeypatch, 3, 16)
    service.client.corpus_aggregate = True
    req = QARequest(question='How many images show people?', answer_language='en')
    versions = QAVersions(model='test', prompt='test', corpus='test', region=None)
    first = run_inline(reasoner.answer(req, 'user', 'r1', time.monotonic()+30, versions))
    calls = len(service.client.calls)
    second = run_inline(reasoner.answer(req, 'user', 'r2', time.monotonic()+30, versions))
    assert first.status == 'answered' and second.cache_hit and len(service.client.calls) == calls
    third = run_inline(reasoner.answer(req, 'other', 'r3', time.monotonic()+30, versions))
    assert not third.cache_hit and len(service.client.calls) > calls


def test_corpus_aggregate_is_computed_and_all_images_verified(tmp_path, monkeypatch):
    reasoner, service, ids = make_reasoner(tmp_path, monkeypatch, 3, 16)
    service.client.corpus_aggregate = True
    req = QARequest(question='How many images show people?', answer_language='en')
    versions = QAVersions(model='test', prompt='test', corpus='test', region=None)
    response = run_inline(reasoner.answer(req, 'user', 'r1', time.monotonic()+30, versions))
    assert response.status == 'answered' and response.answer == '3 images'
    assert response.examined_scope['exhaustive']
    checked = [x for stage, batch in service.client.calls if stage == 'verifying' for x in batch]
    assert set(checked) == set(ids)


def test_unverified_corpus_result_is_never_returned_as_answer(tmp_path, monkeypatch):
    reasoner, service, _ = make_reasoner(tmp_path, monkeypatch, 3, 16)
    service.client.verified = False
    req = QARequest(question='Which image has the most people?', answer_language='en')
    versions = QAVersions(model='test', prompt='test', corpus='test', region=None)
    response = run_inline(reasoner.answer(req, 'user', 'r1', time.monotonic()+30, versions))
    assert response.status == 'insufficient_evidence' and response.answer is None
    assert not response.examined_scope['exhaustive']
    assert response.examined_scope['coverage']['reviewed_images'] == 0


def test_reviewed_catalog_supports_max_beyond_request_scan_limit(tmp_path, monkeypatch):
    reasoner, service, ids = make_reasoner(tmp_path, monkeypatch, 26, 8)
    reasoner.people = SimpleNamespace(images={image_id: ([QARegion(label='person', box=[.1, .1, .5, .9])]
                                              if index == 0 else [])
                                              for index, image_id in enumerate(ids)})
    req = QARequest(question='Which image has the most people?', answer_language='en')
    versions = QAVersions(model='test', prompt='test', corpus='test', region=None)
    response = run_inline(reasoner.answer(req, 'user', 'r1', time.monotonic()+30, versions))
    assert response.status == 'answered' and ids[0] in response.answer
    assert response.examined_scope['exhaustive']
    assert response.examined_scope['coverage']['reviewed_images'] == len(ids)
    assert [stage for stage, _ in service.client.calls] == ['planning']


def test_invalid_optional_fields_never_become_a_complete_count():
    parsed = normalize_observations({'observations': [{
        'image_id': 'a.jpg', 'context_match': 'yes', 'observation': 'people',
        'state': None, 'checks': None, 'count_visibility': 'complete',
        'regions': [{'label': 'P1', 'box': [.1, .1, .3, .9], 'confidence': .9},
                    {'label': 'P2', 'box': [[1, 2], [3, 4]]}],
        'unused_extra': 'ignore'}]})
    obs = parsed.observations[0]
    assert len(obs.regions) == 1 and obs.count_visibility == 'partial'
    assert obs.image_id == 'a.jpg' and obs.state == 'unknown'


def test_null_regions_cannot_become_exact_zero():
    parsed = normalize_observations({'observations': [{
        'image_id': 'a.jpg', 'context_match': 'yes', 'observation': 'unclear',
        'count_visibility': 'complete', 'regions': None}]})
    assert parsed.observations[0].count_visibility == 'unknown'


def test_total_count_routes_to_distinct_without_forcing_single_image_or_events():
    def plan(subject='person'):
        return Plan(question='q', question_type='scene_search', subject=subject,
                    operation='count_visible', information_needed='count', queries=['kitchen'])
    assert apply_distinct_scope(plan(), QARequest(question='Có tổng cộng bao nhiêu người?')).operation == 'count_distinct_people'
    assert apply_distinct_scope(plan('bottle'), QARequest(question='Tổng cộng có bao nhiêu chai?')).operation == 'count_distinct_objects'
    assert apply_distinct_scope(plan(), QARequest(question='Tổng cộng có bao nhiêu người trong ảnh này?')).operation == 'count_visible'
    assert apply_distinct_scope(plan(), QARequest(question='Tổng cộng có bao nhiêu lượt xuất hiện?')).operation == 'count_visible'
    assert apply_distinct_scope(plan(), QARequest(question='Có tổng cộng bao nhiêu ảnh có người?')).operation == 'count_visible'
    assert apply_distinct_scope(plan(), QARequest(mode='selected_image', question='Có tổng cộng bao nhiêu người?', image_id='a.jpg')).operation == 'count_visible'
    prior = plan().model_copy(update={'context_source': 'previous_evidence'})
    assert apply_distinct_scope(prior, QARequest(question='Có tổng cộng bao nhiêu người?')).context_source == 'new_search'
    prior = plan().model_copy(update={'context_source': 'previous_evidence'})
    assert apply_distinct_scope(prior, QARequest(question='Có tổng cộng bao nhiêu người trong các ảnh đó?')).context_source == 'previous_evidence'


def test_invalid_observation_is_isolated_and_cannot_make_corpus_answer(tmp_path, monkeypatch):
    reasoner, service, ids = make_reasoner(tmp_path, monkeypatch, 3, 16)
    original = service.client.structured

    async def malformed_first_image(prompt, images, deadline):
        task = json.loads(prompt[prompt.index('{'):])
        if task['stage'] == 'observing' and any(image.image_id == ids[0] for image in images):
            return {'observations': [{'image_id': image.image_id, 'context_match': 'invalid',
                                      'observation': 'unclear'} for image in images]}
        return await original(prompt, images, deadline)

    service.client.structured = malformed_first_image
    req = QARequest(question='Which image has the most people?', answer_language='en')
    versions = QAVersions(model='test', prompt='test', corpus='test', region=None)
    response = run_inline(reasoner.answer(req, 'user', 'r1', time.monotonic()+30, versions))
    assert response.status == 'insufficient_evidence' and response.answer is None
    assert response.reason_code == 'incomplete_corpus_coverage'
    assert response.examined_count == 2 and not response.examined_scope['exhaustive']


def test_candidate_pairs_are_cross_image_and_unverified():
    import numpy as np
    boxes = [[.1, .1, .4, .9], [.6, .1, .9, .9], [.1, .1, .4, .9]]
    regions = [QARegion(label='P1', box=boxes[0]), QARegion(label='P2', box=boxes[1])]
    observations = [SimpleNamespace(image_id='a.jpg', context_match='yes', regions=regions),
                    SimpleNamespace(image_id='b.jpg', context_match='yes',
                                    regions=[QARegion(label='P1', box=boxes[2])])]
    index = SimpleNamespace(version='v1', rows_by_image={'a.jpg': [0, 1], 'b.jpg': [2]},
                            records=[{'kind': 'person', 'box': box} for box in boxes],
                            vectors=np.array([[1, 0], [0, 1], [.99, .01]], dtype=np.float32))
    pairs = candidate_person_pairs(observations, index)
    assert len(pairs) == 1 and pairs[0]['status'] == 'unverified'
    assert pairs[0]['first']['image_id'] == 'a.jpg' and pairs[0]['second']['image_id'] == 'b.jpg'
    assert pairs[0]['first']['label'] == 'P1' and pairs[0]['first']['region_id'] == 'v1:0'
    assert candidate_person_pairs(observations, index, limit=0) == []


def test_distinct_threshold_and_same_image_constraint():
    import numpy as np
    box1, box2 = [.1, .1, .4, .9], [.6, .1, .9, .9]
    observations = [
        SimpleNamespace(image_id='a', context_match='yes', regions=[QARegion(label='P1', box=box1), QARegion(label='P2', box=box2)]),
        SimpleNamespace(image_id='b', context_match='yes', regions=[QARegion(label='P1', box=box1)]),
        SimpleNamespace(image_id='c', context_match='yes', regions=[QARegion(label='P1', box=box1)]),
        SimpleNamespace(image_id='missing', context_match='yes', regions=[QARegion(label='P1', box=box1)]),
    ]
    idx = SimpleNamespace(version='v', rows_by_image={'a': [0, 1], 'b': [2], 'c': [3]},
        records=[{'kind': 'person', 'box': x} for x in [box1, box2, box1, box1]],
        vectors=np.array([[1, 0], [1, 0], [.99, .01], [0, 1]], dtype=np.float32))
    estimate = estimate_distinct_people(observations, idx, .85)
    assert estimate['status'] == 'experimental' and estimate['estimated_count'] == 3
    assert estimate['assigned_regions'] == 4 and estimate['unassigned_regions'] == 1
    assert len(estimate['groups'][0]['members']) == 2
    assert {x['image_id'] for x in estimate['groups'][0]['members']} == {'a', 'b'}
    assert len(estimate['groups'][1]['members']) == 1  # same image, even with cosine 1
    first, second = estimate['groups'][0]['members']
    blocked = estimate['groups'][1]['members'][0]
    new = estimate['groups'][2]['members'][0]
    assert first['sequence'] == 1 and first['comparison']['comparisons'] == []
    assert blocked['comparison']['decision'] == 'new_group'
    assert blocked['comparison']['comparisons'][0]['eligible'] is False
    assert blocked['comparison']['comparisons'][0]['similarity'] == 1.0
    assert second['sequence'] == 3 and second['comparison']['decision'] == 'matched'
    assert {item['group_id'] for item in second['comparison']['comparisons']} == {'G1', 'G2'}
    assert all(item['above_threshold'] for item in second['comparison']['comparisons'])
    assert new['comparison']['decision'] == 'new_group'
    assert all(not item['above_threshold'] for item in new['comparison']['comparisons'])
    import json
    json.dumps(estimate)
    assert estimate_distinct_people(observations, idx, .99999)['estimated_count'] == 4


def test_distinct_seeds_richest_frame_independent_of_retrieval_order():
    import numpy as np
    from app.qa_people import prepare_object_crops
    box1, box2 = [.1, .1, .4, .9], [.6, .1, .9, .9]
    sparse = SimpleNamespace(image_id='a', context_match='yes', regions=[QARegion(label='P1', box=box1)])
    rich = SimpleNamespace(image_id='b', context_match='yes', regions=[QARegion(label='P1', box=box1), QARegion(label='P2', box=box2)])
    vectors = {('a', 0): np.array([1., 0.]), ('b', 0): np.array([1., 0.]),
               ('b', 1): np.array([1., 0.])}
    first = estimate_distinct_people([sparse, rich], None, .85, crop_vectors=vectors)
    second = estimate_distinct_people([rich, sparse], None, .85, crop_vectors=vectors)
    assert first == second
    assert first['estimated_count'] == 2
    assert [member['image_id'] for group in first['groups'] for member in group['members']].count('b') == 2
    assert sorted(member['image_id'] for group in first['groups'] for member in group['members'] if member['sequence'] <= 2) == ['b', 'b']
    from PIL import Image
    import io
    buf = io.BytesIO()
    Image.new('RGB', (10, 10)).save(buf, format='PNG')
    evidence = {key: SimpleNamespace(image_bytes=buf.getvalue()) for key in ('a', 'b')}
    keys, _ = prepare_object_crops([sparse, rich], evidence, max_people=2)
    assert keys == [('b', 0), ('b', 1)]


def test_distinct_candidates_do_not_become_an_exact_total(tmp_path, monkeypatch):
    import io
    import numpy as np
    from PIL import Image
    reasoner, service, ids = make_reasoner(tmp_path, monkeypatch, 3, 16)
    service.client.distinct_people = True
    box = [.1, .1, .4, .9]
    service.search_engine.region_index = SimpleNamespace(
        version='v1', rows_by_image={image_id: [i] for i, image_id in enumerate(ids)},
        records=[{'kind': 'person', 'box': box} for _ in ids],
        vectors=np.array([[1., 0.], [.99, .01], [0., 1.]], dtype=np.float32))

    async def search(query, k, detail=None):
        return ([{'id': image_id, 'matches': []} for image_id in ids], 0)

    service.search_engine.search = search
    async def encode(crops):
        return np.array([[1., 0.], [.99, .01], [0., 1.]], dtype=np.float32)
    async def wait_inline(awaitable, timeout):
        return await awaitable
    monkeypatch.setattr('app.qa_reasoning.asyncio.wait_for', wait_inline)
    service.search_engine.encode_image_crops = encode
    image = io.BytesIO()
    Image.new('RGB', (20, 20)).save(image, format='JPEG')
    service.evidence_loader.load_for_image = lambda image_id, *a, **kw: SimpleNamespace(items=[
        SimpleNamespace(image_id=image_id, evidence_id=image_id, kind='full_image', box=None, image_bytes=image.getvalue())])
    req = QARequest(question='How many different people?', answer_language='en', conversation_id='trace-test')
    versions = QAVersions(model='test', prompt='test', corpus='test', region='v1')
    response = run_inline(reasoner.answer(req, 'user', 'r1', time.monotonic()+30, versions))
    assert response.status == 'answered' and response.partial
    assert 'Estimated 2' in response.answer and 'unverified' in response.answer
    assert not response.identity_match_candidates
    assert response.distinct_estimate['status'] == 'experimental'
    assert response.distinct_estimate['estimated_count'] == 2
    assert not response.examined_scope['exhaustive']
    progress = reasoner.progress['user', 'trace-test']
    assert progress['plan']['operation'] == 'count_distinct_people'
    assert progress['plan']['subject'] == 'person'
    assert [item['image_id'] for item in progress['observations']] == ids
    assert all('observation' in item and 'regions' in item for item in progress['observations'])
    assert 'planning' in progress['stages'] and 'observing' in progress['stages']


def make_proposal_index(ids, label):
    import numpy as np
    box = [.1, .1, .4, .9]
    return SimpleNamespace(version='v1', rows_by_image={image_id: [i] for i, image_id in enumerate(ids)},
        records=[{'kind': label, 'box': box, 'objects': [{'label': label, 'score': .8}]} for _ in ids],
        vectors=np.array([[1., 0.] for _ in ids], dtype=np.float32))


def test_regionclip_proposals_filter_class_deduplicate_and_have_no_gemini_fallback():
    from app.qa_people import regionclip_object_regions
    index = SimpleNamespace(rows_by_image={'a': [0, 1, 2]}, records=[
        {'box': [.1, .1, .4, .9], 'objects': [{'label': 'person', 'score': .8}]},
        {'box': [.1, .1, .4, .9], 'objects': [{'label': 'person', 'score': .7}]},
        {'box': [.5, .1, .9, .9], 'objects': [{'label': 'chair', 'score': .9}]},
    ])
    regions = regionclip_object_regions(index, 'a', 'person')
    assert len(regions) == 1 and regions[0].label == 'R1'
    assert regions[0].box == [.1, .1, .4, .9]
    assert regionclip_object_regions(index, 'missing', 'person') == []
    assert regionclip_object_regions(None, 'a', 'person') == []


def test_distinct_processes_top_images_in_order_and_encodes_every_crop(tmp_path, monkeypatch):
    import io
    import numpy as np
    from PIL import Image
    reasoner, service, ids = make_reasoner(tmp_path, monkeypatch, 10, 16)
    service.settings.qa_early_stop = False  # Checks that every top image is processed in order.
    service.search_engine.region_index = make_proposal_index(ids, 'person')
    service.client.distinct_people = True
    original_client = service.client.structured
    async def scene_misclassification(prompt, images, deadline):
        task = json.loads(prompt[prompt.index('{'):])
        if task['stage'] == 'planning':
            return {'question': task['question'], 'question_type': 'scene_search', 'operation': 'count_visible',
                    'answer_scope': 'scene', 'subject': 'person', 'information_needed': 'count', 'queries': ['people']}
        return await original_client(prompt, images, deadline)
    service.client.structured = scene_misclassification
    ranked = list(reversed(ids))
    async def search(query, k, detail=None):
        return ([{'id': image_id, 'matches': []} for image_id in ranked[:k]], 0)
    async def encode(crops):
        return np.tile(np.array([[1., 0.]], dtype=np.float32), (len(crops), 1))
    async def wait_inline(awaitable, timeout):
        return await awaitable
    monkeypatch.setattr('app.qa_reasoning.asyncio.wait_for', wait_inline)
    service.search_engine.search = search
    service.search_engine.encode_image_crops = encode
    img_bytes = io.BytesIO()
    Image.new('RGB', (20, 20)).save(img_bytes, format='JPEG')
    def load(image_id, *args, **kwargs):
        return SimpleNamespace(items=[SimpleNamespace(image_id=image_id, evidence_id=image_id,
            kind='full_image', box=None, image_bytes=img_bytes.getvalue())])
    service.evidence_loader.load_for_image = load
    req = QARequest(question='Có tổng cộng bao nhiêu người?', answer_language='en')
    versions = QAVersions(model='test', prompt='test', corpus='test', region=None)
    response = run_inline(reasoner.answer(req, 'user', 'r1', time.monotonic()+30, versions))
    seen = [image for stage, batch in service.client.calls if stage == 'observing' for image in batch]
    assert seen == ranked
    assert response.distinct_estimate['assigned_regions'] == 10
    assert response.distinct_estimate['unassigned_regions'] == 0
    assert response.distinct_estimate['estimated_count'] == 1
    assert response.distinct_estimate['embedding_model'] == 'CLIP crop'
    assert response.partial and not response.examined_scope['exhaustive']
    assert response.operation == 'count_distinct_people'


def test_distinct_works_for_a_non_person_category(tmp_path, monkeypatch):
    import io
    import numpy as np
    from PIL import Image
    reasoner, service, ids = make_reasoner(tmp_path, monkeypatch, 2, 16)
    service.search_engine.region_index = make_proposal_index(ids, 'bottle')
    original = service.client.structured
    async def client(prompt, images, deadline):
        task = json.loads(prompt[prompt.index('{'):])
        if task['stage'] == 'planning':
            return {'question': task['question'], 'question_type': 'multi_image',
                    'operation': 'count_distinct_objects', 'answer_scope': 'corpus',
                    'subject': 'bottle', 'information_needed': 'different bottles', 'queries': ['bottle']}
        return await original(prompt, images, deadline)
    async def search(query, k, detail=None):
        return ([{'id': image_id, 'matches': []} for image_id in ids], 0)
    async def encode(crops):
        return np.array([[1., 0.], [0., 1.]], dtype=np.float32)
    async def wait_inline(awaitable, timeout):
        return await awaitable
    monkeypatch.setattr('app.qa_reasoning.asyncio.wait_for', wait_inline)
    service.client.structured = client
    service.search_engine.search, service.search_engine.encode_image_crops = search, encode
    img_bytes = io.BytesIO()
    Image.new('RGB', (20, 20)).save(img_bytes, format='JPEG')
    service.evidence_loader.load_for_image = lambda image_id, *a, **kw: SimpleNamespace(items=[
        SimpleNamespace(image_id=image_id, evidence_id=image_id, kind='full_image', box=None,
                        image_bytes=img_bytes.getvalue())])
    req = QARequest(question='How many distinct bottles?', answer_language='en')
    versions = QAVersions(model='test', prompt='test', corpus='test', region=None)
    result = run_inline(reasoner.answer(req, 'user', 'r1', time.monotonic()+30, versions))
    assert result.distinct_estimate['category'] == 'bottle'
    assert result.distinct_estimate['estimated_count'] == 2
    assert 'Estimated 2' in result.answer and result.partial


def test_distinct_timeout_keeps_examined_regions_and_partial_estimate(tmp_path, monkeypatch):
    import io
    import numpy as np
    from PIL import Image
    reasoner, service, ids = make_reasoner(tmp_path, monkeypatch, 10, 16)
    service.search_engine.region_index = make_proposal_index(ids, 'person')
    service.client.distinct_people = True
    service.search_engine.condition_detections = {ids[0]: [
        {'label': 'person', 'score': .9, 'box': [.6, .1, .9, .9]}]}
    original = service.client.structured
    calls = 0
    async def client(prompt, images, deadline):
        nonlocal calls
        task = json.loads(prompt[prompt.index('{'):])
        if task['stage'] == 'observing':
            calls += 1
            if calls == 2:
                raise QATimeoutError()
        return await original(prompt, images, deadline)
    async def search(query, k, detail=None):
        return ([{'id': image_id, 'matches': []} for image_id in ids], 0)
    async def encode(crops):
        return np.tile(np.array([[1., 0.]], dtype=np.float32), (len(crops), 1))
    async def wait_inline(awaitable, timeout):
        return await awaitable
    monkeypatch.setattr('app.qa_reasoning.asyncio.wait_for', wait_inline)
    service.client.structured = client
    service.search_engine.search, service.search_engine.encode_image_crops = search, encode
    img_bytes = io.BytesIO()
    Image.new('RGB', (20, 20)).save(img_bytes, format='JPEG')
    service.evidence_loader.load_for_image = lambda image_id, *a, **kw: SimpleNamespace(items=[
        SimpleNamespace(image_id=image_id, evidence_id=image_id, kind='full_image', box=None,
                        image_bytes=img_bytes.getvalue())])
    req = QARequest(question='How many distinct people?', answer_language='en')
    versions = QAVersions(model='test', prompt='test', corpus='test', region=None)
    result = run_inline(reasoner.answer(req, 'user', 'r1', time.monotonic()+30, versions))
    assert result.status == 'answered' and result.partial
    assert result.distinct_estimate['examined_images'] == 2
    assert result.distinct_estimate['requested_images'] == 10 and result.distinct_estimate['incomplete']
    assert len(result.evidence) == 2 and all(item.regions for item in result.evidence)
    first = next(item for item in result.evidence if item.image_id == ids[0])
    assert not first.unverified_proposals  # Old precomputed detections are not reused.
    assert first.regions[0].label == 'D1'
    assert result.distinct_estimate['assigned_regions'] == 2  # proposals are not silently counted


def test_distinct_verification_timeout_keeps_experimental_groups(tmp_path, monkeypatch):
    import io
    import numpy as np
    from PIL import Image
    reasoner, service, ids = make_reasoner(tmp_path, monkeypatch, 2, 16)
    service.client.distinct_people = True
    original = service.client.structured
    async def client(prompt, images, deadline):
        if json.loads(prompt[prompt.index('{'):])['stage'] == 'verifying':
            raise QATimeoutError()
        return await original(prompt, images, deadline)
    async def search(query, k, detail=None):
        return ([{'id': image_id, 'matches': []} for image_id in ids], 0)
    async def encode(crops):
        return np.tile(np.array([[1., 0.]], dtype=np.float32), (len(crops), 1))
    async def wait_inline(awaitable, timeout):
        return await awaitable
    monkeypatch.setattr('app.qa_reasoning.asyncio.wait_for', wait_inline)
    service.client.structured = client
    service.search_engine.search = search
    service.search_engine.encode_image_crops = encode
    image = io.BytesIO()
    Image.new('RGB', (20, 20)).save(image, format='JPEG')
    service.evidence_loader.load_for_image = lambda image_id, *a, **kw: SimpleNamespace(items=[
        SimpleNamespace(image_id=image_id, evidence_id=image_id, kind='full_image', box=None,
                        image_bytes=image.getvalue())])
    req = QARequest(question='How many distinct people?', answer_language='en')
    versions = QAVersions(model='test', prompt='test', corpus='test', region=None)
    result = run_inline(reasoner.answer(req, 'user', 'r1', time.monotonic()+30, versions))
    assert result.status == 'insufficient_evidence' and result.reason_code == 'budget_exceeded'
    assert result.distinct_estimate['estimated_count'] == 1
    assert result.distinct_estimate['status'] == 'experimental' and 'unverified' in result.message


def test_distinct_keeps_completed_crop_batches_on_timeout(tmp_path, monkeypatch):
    import io
    import numpy as np
    from PIL import Image
    reasoner, service, ids = make_reasoner(tmp_path, monkeypatch, 6, 16)
    service.client.distinct_people = True
    service.settings.qa_early_stop = False
    async def search(query, k, detail=None):
        return ([{'id': image_id, 'matches': []} for image_id in ids], 0)
    batches = []
    async def encode(crops):
        batches.append(len(crops))
        if len(batches) > 1:
            raise TimeoutError('budget')
        return np.tile(np.array([[1., 0.]], dtype=np.float32), (len(crops), 1))
    async def wait_inline(awaitable, timeout):
        return await awaitable
    monkeypatch.setattr('app.qa_reasoning.asyncio.wait_for', wait_inline)
    service.search_engine.search = search
    service.search_engine.encode_image_crops = encode
    image = io.BytesIO()
    Image.new('RGB', (20, 20)).save(image, format='JPEG')
    service.evidence_loader.load_for_image = lambda image_id, *a, **kw: SimpleNamespace(items=[
        SimpleNamespace(image_id=image_id, evidence_id=image_id, kind='full_image', box=None,
                        image_bytes=image.getvalue())])
    req = QARequest(question='How many distinct people?', answer_language='en')
    versions = QAVersions(model='test', prompt='test', corpus='test', region=None)
    result = run_inline(reasoner.answer(req, 'user', 'r1', time.monotonic()+30, versions))
    assert batches == [4, 2]
    assert result.distinct_estimate['assigned_regions'] == 4
    assert result.distinct_estimate['unassigned_regions'] == 2
    assert result.distinct_estimate['status'] == 'experimental'


def test_repair_box_fixes_only_lossless_format_slips():
    from app.qa_reasoning import repair_box, repair_observation_regions
    assert repair_box([[.1, .2], [.4, .9]]) == [.1, .2, .4, .9]
    assert repair_box(['0.4', '0.9', '0.1', '0.2']) == [.1, .2, .4, .9]
    assert repair_box({'xmin': .1, 'ymin': .2, 'xmax': .4, 'ymax': 1.01}) == [.1, .2, .4, 1.0]
    assert repair_box([100, 200, 300, 400]) is None and repair_box([.1, .2, .3]) is None
    assert repair_box([.1, .2, .1, .5]) is None
    raw = {'observations': [{'image_id': 'a', 'count_visibility': 'complete',
                             'regions': [{'box': [[.1, .1], [.3, .9]]}, {'label': 'P2', 'box': [5, 5, 50, 50]}]}]}
    row = repair_observation_regions(raw)['observations'][0]
    assert row['regions'] == [{'label': 'R1', 'box': [.1, .1, .3, .9]}]
    assert row['count_visibility'] == 'partial'


def test_metadata_ordered_maximum_reads_crowded_frames_and_answers(tmp_path, monkeypatch):
    """With image metadata, a maximum reads the most crowded frames first and is answered from them
    (verifying only frames that were read: unread frames have no evidence to re-check)."""
    from app.image_metadata import ImageMetadata
    reasoner, service, ids = make_reasoner(tmp_path, monkeypatch, 26, 128)
    crowd = {'17.jpg': 3, '05.jpg': 2}
    records = {x: ImageMetadata.model_validate({'caption': 'kitchen', 'people_count_estimate': crowd.get(x, 1),
                                                'people': [{'description': 'person'}] * crowd.get(x, 1)})
               for x in ids}
    service.search_engine.image_metadata = SimpleNamespace(get=records.get)
    original = service.client.structured

    async def structured(prompt, images, deadline):
        result = await original(prompt, images, deadline)
        if 'observations' in result:
            for item in result['observations']:
                n = crowd.get(item['image_id'], 1)
                item['regions'] = [{'label': f'P{i}', 'box': [.1 + .2 * i, .1, .25 + .2 * i, .9]} for i in range(n)]
        return result
    service.client.structured = structured
    req = QARequest(question='Which image has the most people?', answer_language='en')
    versions = QAVersions(model='test', prompt='test', corpus='test', region=None)
    response = run_inline(reasoner.answer(req, 'user', 'r1', time.monotonic()+30, versions))
    observed = [image_id for stage, batch in service.client.calls if stage == 'observing' for image_id in batch]
    assert observed[:2] == ['17.jpg', '05.jpg'] and len(set(observed)) <= 16
    assert response.status == 'answered' and response.short_answer == '3'
    assert '17.jpg' in response.answer


def test_scene_count_compares_frames_when_the_true_one_is_left_unclear(tmp_path, monkeypatch):
    """The 13_521 case: the true frame was read with the right count but left 'unclear' (a condition was
    skipped) while a loose frame passed. Frames that disagree are compared side by side and the chosen
    frame's count is the answer."""
    reasoner, service, ids = make_reasoner(tmp_path, monkeypatch, 4, 128)
    service.search_engine.search = None
    calls = []

    async def search(query, k, detail=None):
        return [{'id': x, 'score': 1.0 - i / 10} for i, x in enumerate(ids)], 0
    service.search_engine.search = search

    async def structured(prompt, images, deadline):
        task = json.loads(prompt[prompt.index('{'):])
        stage = task['stage']
        calls.append(stage)
        if stage == 'planning':
            return {'question': task['question'], 'question_type': 'scene_search', 'operation': 'count_visible',
                    'answer_scope': 'scene', 'subject': 'person', 'information_needed': 'count',
                    'known_conditions': ['man carrying a pot of broccoli'], 'queries': ['man carrying broccoli']}
        if stage == 'observing':
            out = []
            for item in images:
                if item.image_id == '00.jpg':   # loose frame: passes, 3 people
                    out.append({'image_id': '00.jpg', 'context_match': 'yes', 'observation': 'three men',
                                'checks': [{'condition_id': 1, 'status': 'met'}], 'count_visibility': 'complete',
                                'regions': [{'label': f'P{i}', 'box': [.1 + .2 * i, .1, .2 + .2 * i, .9]} for i in range(3)]})
                elif item.image_id == '01.jpg':  # true frame: right count, condition skipped
                    out.append({'image_id': '01.jpg', 'context_match': 'unclear', 'observation': 'four people',
                                'count_visibility': 'complete',
                                'regions': [{'label': f'P{i}', 'box': [.05 + .2 * i, .1, .15 + .2 * i, .9]} for i in range(4)]})
                else:
                    out.append({'image_id': item.image_id, 'context_match': 'no', 'observation': 'other',
                                'checks': [{'condition_id': 1, 'status': 'violated'}]})
            return {'observations': out}
        if stage == 'choosing':
            assert {f['image_id'] for f in task['frames']} == {'00.jpg', '01.jpg'}
            return {'image_id': '01.jpg'}
        if stage == 'verifying':
            return {'supported': True, 'conflicts': False}
        raise AssertionError(stage)
    service.client.structured = structured
    req = QARequest(question='How many people are there when a man carries broccoli?', answer_language='en')
    versions = QAVersions(model='test', prompt='test', corpus='test', region=None)
    response = run_inline(reasoner.answer(req, 'user', 'r1', time.monotonic()+30, versions))
    assert 'choosing' in calls
    assert response.status == 'answered' and response.short_answer == '4 people'
    assert '01.jpg' in response.answer and '00.jpg (3)' in response.answer


def _roster_setup(tmp_path, monkeypatch, people):
    from app.image_metadata import ImageMetadata
    reasoner, service, ids = make_reasoner(tmp_path, monkeypatch, 6, 128)
    service.client.distinct_people = True
    service.settings.qa_people_roster = True
    service.settings.qa_roster_frames = 4
    crowd = {'02.jpg': 3}
    wear = {'00.jpg': 'red plaid shirt', '01.jpg': 'white top', '02.jpg': 'blue shirt', '03.jpg': 'grey sweater',
            '04.jpg': 'red plaid shirt', '05.jpg': 'white top'}
    records = {x: ImageMetadata.model_validate({'caption': 'k', 'people_count_estimate': crowd.get(x, 1),
                'people': [{'description': 'man', 'upper_clothing': wear[x]}] * crowd.get(x, 1)}) for x in ids}
    service.search_engine.image_metadata = SimpleNamespace(get=records.get)

    async def search(query, k, detail=None):  # the older estimate's retrieval, used when the roster is rejected
        return [{'id': x, 'score': 1.0} for x in ids], 0
    service.search_engine.search = search
    original = service.client.structured
    seen = {}

    async def structured(prompt, images, deadline, **options):
        task = json.loads(prompt[prompt.index('{'):])
        if task['stage'] == 'rostering':
            seen['frames'] = task['frames']
            seen['options'] = options
            return {'people': people(task['frames'])}
        return await original(prompt, images, deadline)
    service.client.structured = structured
    return reasoner, seen


def test_people_roster_answers_the_session_count(tmp_path, monkeypatch):
    reasoner, seen = _roster_setup(tmp_path, monkeypatch, lambda frames: [
        {'label': f'P{i}', 'appearance': 'someone', 'image_ids': [frames[i % len(frames)], 'made_up.jpg']}
        for i in range(4)])
    req = QARequest(question='How many different people took part?', answer_language='en')
    versions = QAVersions(model='test', prompt='test', corpus='test', region=None)
    response = run_inline(reasoner.answer(req, 'user', 'r1', time.monotonic()+30, versions))
    assert seen['frames'][0] == '02.jpg' and len(seen['frames']) <= 4  # crowded frame first, then clothing groups
    assert seen['options'] == {'max_tokens': 12000}
    assert response.status == 'answered' and response.short_answer == '4'
    assert all('made_up.jpg' not in c.evidence_ids for c in response.claims) and len(response.claims) == 4


def test_people_roster_smaller_than_one_crowded_frame_is_not_used(tmp_path, monkeypatch):
    reasoner, seen = _roster_setup(tmp_path, monkeypatch, lambda frames: [
        {'label': 'P1', 'appearance': 'someone', 'image_ids': [frames[0]]}])
    req = QARequest(question='How many different people took part?', answer_language='en')
    versions = QAVersions(model='test', prompt='test', corpus='test', region=None)
    response = run_inline(reasoner.answer(req, 'user', 'r1', time.monotonic()+30, versions))
    assert 'frames' in seen and response.short_answer != '1'
