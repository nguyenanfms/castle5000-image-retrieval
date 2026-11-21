"""Exercise orchestration without a network-backed Windows event loop."""
import json
import time
from types import SimpleNamespace

from app.qa_reasoning import ReasoningQA, normalize_observations
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

    async def structured(self, prompt, images, deadline):
        task = json.loads(prompt[prompt.index('{'):])
        stage = task['stage']
        self.calls.append((stage, [item.image_id for item in images]))
        if stage == 'planning':
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
    # Every image ties, so the response safely declines a list larger than its output limit.
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
    req = QARequest(question='Which image has the most people?', answer_language='en')
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
