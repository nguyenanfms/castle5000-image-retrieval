import contextlib
import io
import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import httpx
import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from castle.app import create_app
from castle.auto_answer import AutoAnswer, Decision
from castle.catalog import scan, to_item_name
from castle.config import Config
from castle.dres import DresGateway, DresError
from castle.index import build, SearchIndex


class FakeEncoder:
    calls = 0
    device = "test-fixture"

    def __init__(self, cfg):
        pass

    def images(self, images):
        FakeEncoder.calls += 1
        v = np.asarray([np.array(im).mean(axis=(0, 1)) for im in images], dtype=np.float32)
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)

    def text(self, query):
        return {"red": np.array([1, 0, 0], np.float32), "green": np.array([0, 1, 0], np.float32)}.get(query, np.array([0, 0, 1], np.float32))


class SystemTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cfg = Config(self.root, self.root/"data", self.root/"artifacts", self.root/"cache", None)
        for batch, filename, color in [("Batch1", "08_2.jpg", "red"), ("Batch1", "08_10.jpg", "green"), ("Batch2", "08_11.jpg", "blue")]:
            p = self.cfg.data_dir / batch / filename
            p.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (40, 30), color).save(p)
        (self.root/"web").mkdir()
        (self.root/"web"/"index.html").write_text("<h1>Test UI</h1>")

    def tearDown(self):
        self.tmp.cleanup()

    def build(self, factory=FakeEncoder):
        with contextlib.redirect_stdout(io.StringIO()):
            return build(self.cfg, factory)

    def index(self):
        self.build()
        return SearchIndex(self.cfg, FakeEncoder)

    def test_submission_names_and_windows_paths(self):
        self.assertEqual(to_item_name(r"D:\data\Batch1\08_262.jpg"), "day1_Kitchen_08_262")
        self.assertEqual(to_item_name("day1_Kitchen_08_262.jpg"), "day1_Kitchen_08_262")
        self.assertIsNone(to_item_name("arbitrary.jpg"))
        self.assertIsNone(to_item_name("28_1.jpg"))

    def test_recursive_scan_and_numeric_order(self):
        idx = self.index()
        self.assertEqual([r["filename"] for r in idx.search(k=10)], ["08_2.jpg", "08_10.jpg", "08_11.jpg"])

    def test_ambiguous_dres_ids_are_flagged(self):
        duplicate = self.cfg.data_dir/"Batch2"/"08_2.jpg"
        Image.new("RGB", (30, 30), "white").save(duplicate)
        self.assertEqual(sum(r["ambiguous_item"] for r in scan(self.cfg)), 2)

    def test_exact_retrieval_and_filters(self):
        idx = self.index()
        self.assertEqual(idx.search("green", k=1)[0]["filename"], "08_10.jpg")
        self.assertEqual(idx.search("red", batch="Batch2")[0]["filename"], "08_11.jpg")
        self.assertEqual(idx.search("red", hour=9), [])

    def test_neighbor_can_cross_partition_and_stays_same_hour(self):
        idx = self.index()
        key = next(r["key"] for r in idx.records if r["filename"] == "08_10.jpg")
        self.assertEqual([r["filename"] for r in idx.neighbors(key)], ["08_2.jpg", "08_10.jpg", "08_11.jpg"])

    def test_image_similarity(self):
        idx = self.index()
        rec = next(r for r in idx.records if r["filename"] == "08_11.jpg")
        self.assertEqual(idx.search(image_key=rec["key"], k=1)[0]["key"], rec["key"])

    def test_resume_reuses_features(self):
        FakeEncoder.calls = 0
        self.build()
        calls = FakeEncoder.calls
        self.build()
        self.assertEqual(FakeEncoder.calls, calls)

    def test_changed_source_is_reencoded(self):
        self.build()
        Image.new("RGB", (50, 30), "blue").save(self.cfg.data_dir/"Batch1"/"08_2.jpg")
        FakeEncoder.calls = 0
        self.build()
        self.assertGreater(FakeEncoder.calls, 0)

    def test_failed_build_preserves_current(self):
        self.build()
        current = (self.cfg.artifacts_dir/"CURRENT.json").read_bytes()
        Image.new("RGB", (50, 30), "blue").save(self.cfg.data_dir/"Batch1"/"08_3.jpg")
        class Broken(FakeEncoder):
            def images(self, images):
                raise RuntimeError("test failure")
        with self.assertRaises(RuntimeError):
            self.build(Broken)
        self.assertEqual((self.cfg.artifacts_dir/"CURRENT.json").read_bytes(), current)

    def test_oom_reduces_batch(self):
        class SmallBatch(FakeEncoder):
            def images(self, images):
                if len(images)>1:
                    raise RuntimeError("CUDA out of memory")
                return super().images(images)
            def clear_oom(self):
                pass
        self.assertEqual(self.build(SmallBatch)["indexed"], 3)

    def test_exclusion_and_corrupt_image(self):
        (self.cfg.data_dir/"exclude.txt").write_text("Batch1/08_2.jpg\n")
        (self.cfg.data_dir/"Batch2"/"08_12.jpg").write_text("corrupt")
        stats = self.build()
        self.assertEqual((stats["excluded"], stats["indexed"], len(stats["errors"])), (1, 2, 1))

    def test_metadata_uses_authoritative_image_id(self):
        meta=self.root/"metadata.csv"
        meta.write_text("relative_path,imageID,description\nBatch1/08_2.jpg,day2_Kitchen_08_2,test\n")
        self.cfg.metadata_csv=meta
        row=next(r for r in scan(self.cfg) if r["filename"]=="08_2.jpg")
        self.assertEqual(row["item_name"], "day2_Kitchen_08_2")
        self.assertEqual(row["source"], "day2_Kitchen")

    def test_model_change_requires_rebuild(self):
        self.build()
        self.cfg.model_name="changed"
        with self.assertRaises(ValueError):
            SearchIndex(self.cfg, FakeEncoder)

    def test_api_search_image_and_mock_submission(self):
        self.build()
        app = create_app(self.cfg, index_factory=lambda cfg: SearchIndex(cfg, FakeEncoder))
        with TestClient(app) as c:
            self.assertTrue(c.get("/api/status").json()["ready"])
            record=c.post("/api/search", json={"query":"red","k":1}).json()["results"][0]
            self.assertEqual(c.get(record["thumb"]).status_code,200)
            self.assertEqual(c.get('/api/image/'+record["key"]).status_code,200)
            self.assertEqual(c.post("/api/dres/submit",json={"key":record["key"],"confirmed":True}).status_code,401)
            c.post("/api/dres/login",json={"mode":"mock"}).raise_for_status()
            c.post("/api/dres/select",json={"evaluation":"local-practice"}).raise_for_status()
            result=c.post("/api/dres/submit",json={"key":record["key"],"confirmed":True})
            self.assertEqual(result.json()["verdict"],"MOCK_ACCEPTED")
            self.assertEqual(c.post("/api/dres/submit",json={"key":record["key"],"confirmed":True}).status_code,409)
            self.assertEqual(c.get('/api/image/not-a-key').status_code,404)
            self.assertEqual(c.post('/api/search',json={"k":-1}).status_code,422)
            self.assertEqual(c.post('/api/search',json={},headers={"Origin":"https://foreign.example"}).status_code,403)

    def test_no_index_ui_still_loads(self):
        with TestClient(create_app(self.cfg)) as c:
            self.assertEqual(c.get('/').status_code,200)
            self.assertFalse(c.get('/api/status').json()["ready"])
            self.assertEqual(c.post('/api/search',json={}).status_code,503)

    def test_hybrid_search_uses_metadata_and_preserves_filters_and_image_search(self):
        self.build()

        class Matcher:
            coverage = 3

            def __init__(self, records):
                self.bonuses = np.asarray([3.0 if r["filename"] == "08_11.jpg" else 0.0
                                           for r in records], dtype=np.float32)

            def bonus(self, query, clip_scores, weight):
                return self.bonuses * weight

        class Answerer:
            provider = "fixture"

            def close(self):
                pass

            def search_query(self, kind, hint):
                return hint

            def verify(self, kind, hint, records):
                return Decision(records[0], "Fixture evidence.", "")

        app = create_app(self.cfg, index_factory=lambda cfg: SearchIndex(cfg, FakeEncoder),
                         auto_answer_factory=lambda cfg: Answerer())
        with patch("castle.index.load_metadata_matcher", side_effect=lambda cfg, records: Matcher(records)):
            with TestClient(app) as c:
                self.assertEqual(c.get("/api/status").json()["metadata"]["coverage"], 3)
                old = c.post("/api/search", json={"query": "red"}).raise_for_status().json()
                self.assertEqual((old["search_mode"], old["results"][0]["filename"]), ("clip", "08_2.jpg"))
                hybrid = c.post("/api/search", json={"query": "red", "mode": "hybrid"}).raise_for_status().json()
                self.assertEqual((hybrid["search_mode"], hybrid["results"][0]["filename"]), ("hybrid", "08_11.jpg"))
                filtered = c.post("/api/search", json={"query": "red", "mode": "hybrid", "batch": "Batch1"}
                                  ).raise_for_status().json()
                self.assertEqual({row["batch"] for row in filtered["results"]}, {"Batch1"})
                same = c.post("/api/search", json={"query": "red", "mode": "hybrid",
                                                  "image_key": old["results"][0]["key"]}).raise_for_status().json()
                self.assertEqual((same["search_mode"], same["results"][0]["filename"]), ("clip", "08_2.jpg"))
                evaluation = "local-practice"
                c.post("/api/dres/login", json={"mode": "mock"}).raise_for_status()
                c.post("/api/dres/select", json={"evaluation": evaluation}).raise_for_status()
                ref = c.get("/api/dres/task").raise_for_status().json()["ref"]
                c.post("/api/dres/auto/mode", json={"evaluation": evaluation, "enabled": True}).raise_for_status()
                prepared = c.post("/api/dres/auto/prepare", json={"task_ref": ref, "hint": "red", "mode": "hybrid"}
                                  ).raise_for_status().json()
                self.assertEqual(prepared["search_mode"], "hybrid")
                self.assertEqual(prepared["preview"]["evidence"]["filename"], "08_11.jpg")

    def test_missing_partner_checkout_falls_back_to_clip(self):
        self.build()
        self.cfg.image_metadata_dir = self.root / "descriptions"
        self.cfg.partner_repo_dir = self.root / "missing-repo"
        app = create_app(self.cfg, index_factory=lambda cfg: SearchIndex(cfg, FakeEncoder))
        with TestClient(app) as c:
            status = c.get("/api/status").json()
            self.assertTrue(status["ready"])
            self.assertFalse(status["metadata"]["available"])
            self.assertTrue(status["metadata"]["error"])
            result = c.post("/api/search", json={"query": "red", "mode": "hybrid"}).raise_for_status().json()
            self.assertEqual((result["search_mode"], result["results"][0]["filename"]), ("clip", "08_2.jpg"))

    def test_hydrogen_uses_text_and_image_with_structured_output(self):
        requests = []

        def provider(request):
            self.assertEqual(request.url.path, "/v1beta/models/gemini-4-hydrogen:generateContent")
            self.assertEqual(request.headers["x-goog-api-key"], "fixture-key")
            payload = json.loads(request.content)
            config = payload["generationConfig"]
            self.assertEqual(config["thinkingConfig"], {"thinkingLevel": "HIGH"})
            self.assertEqual(config["responseMimeType"], "application/json")
            self.assertNotIn("maxOutputTokens", config)
            self.assertNotIn("temperature", config)
            parts = payload["contents"][0]["parts"]
            schema = config["responseJsonSchema"]["title"]
            if schema == "RetrievalQuery":
                self.assertEqual(len(parts), 1)
                parsed = {"query": "a red object"}
            else:
                self.assertEqual(schema, "ImageVerdict")
                self.assertEqual(parts[-1]["inlineData"]["mimeType"], "image/jpeg")
                self.assertTrue(parts[-1]["inlineData"]["data"])
                parsed = {"image": 1, "observation": "The image is red.", "answer": "red"}
            requests.append(schema)
            return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": json.dumps(parsed)}]}}]})

        with patch.dict("os.environ", {"CASTLE_AUTO_PROVIDER": "gemini", "GEMINI_API_KEY": "fixture-key",
                                       "CASTLE_GEMINI_MODEL": "gemini-4-hydrogen"}):
            answerer = AutoAnswer(self.cfg)
        answerer.close()
        answerer.client = httpx.Client(base_url="https://generativelanguage.googleapis.com/",
                                       transport=httpx.MockTransport(provider))
        self.addCleanup(answerer.close)
        self.assertEqual(answerer.search_query("QA", "What color is the object?"), "a red object")
        red = next(record for record in scan(self.cfg) if record["filename"] == "08_2.jpg")
        self.assertEqual(answerer.verify("QA", "What color is the object?", [red]).answer, "red")
        self.assertEqual(requests, ["RetrievalQuery", "ImageVerdict"])

    def test_auto_api_retrieves_and_sends_kis_qa_once(self):
        self.build()

        def provider(request):
            body = json.loads(request.content)
            schema = body["format"]["title"]
            if schema == "SceneObservation":
                self.assertEqual(len(body["messages"][0]["images"]), 1)
                self.assertNotIn("màu đỏ", body["messages"][0]["content"])
                answer = {"observation": "A red image."}
            elif schema == "EvidenceMatch":
                prompt = body["messages"][0]["content"]
                answer = {"matches": not any(clue in prompt for clue in ("wrong action", "unclear"))}
            elif body["messages"][0].get("images"):
                self.assertIn(len(body["messages"][0]["images"]), (2, 3))
                answer = {"image": 1, "observation": "Ảnh đầu màu đỏ.", "answer": "  đỏ  "}
                if "unclear" in body["messages"][0]["content"]:
                    answer = {"image": 0, "observation": "", "answer": ""}
            else:
                answer = {"query": "red"}
            return httpx.Response(200, json={"message": {"content": json.dumps(answer)}})

        def answerer_factory(cfg):
            answerer = AutoAnswer(cfg)
            answerer.close()
            answerer.provider = "ollama"
            answerer.client = httpx.Client(base_url="http://127.0.0.1:11434/",
                                           transport=httpx.MockTransport(provider))
            return answerer

        app = create_app(self.cfg, index_factory=lambda cfg: SearchIndex(cfg, FakeEncoder),
                         auto_answer_factory=answerer_factory)
        with TestClient(app) as c:
            c.post("/api/dres/login", json={"mode": "mock"}).raise_for_status()
            for evaluation, kind in (("local-practice", "KIS"), ("local-practice-qa", "QA")):
                c.post("/api/dres/select", json={"evaluation": evaluation}).raise_for_status()
                self.assertFalse(c.get("/api/dres/status").json()["auto_enabled"])
                ref = c.get("/api/dres/task").json()["ref"]

                def prepare(hint):
                    return c.post("/api/dres/auto/prepare", json={"hint": hint, "task_ref": ref})

                self.assertEqual(prepare("red").status_code, 409)
                c.post("/api/dres/auto/mode", json={"enabled": True, "evaluation": evaluation}).raise_for_status()
                self.assertIsNone(prepare("unclear").raise_for_status().json()["ticket"])
                if kind == "KIS":
                    self.assertIsNone(prepare("wrong action").raise_for_status().json()["ticket"])
                old = prepare("red").raise_for_status().json()
                current = prepare("màu đỏ").raise_for_status().json()
                self.assertEqual(c.post("/api/dres/auto/commit", json={"ticket": old["ticket"]}).status_code, 409)
                entry = c.post("/api/dres/auto/commit", json={"ticket": current["ticket"]}).raise_for_status().json()
                self.assertEqual((entry["kind"], entry["verdict"], entry["automatic"]), (kind, "MOCK_ACCEPTED", True))
                self.assertEqual(entry["evidence_item"], "day1_Kitchen_08_2")
                if kind == "QA":
                    self.assertEqual(entry["text"], "đỏ")
                self.assertEqual(c.post("/api/dres/auto/commit", json={"ticket": current["ticket"]}).status_code, 409)
                self.assertEqual(prepare("another hint").status_code, 409)
            self.assertEqual(len(c.get("/api/dres/status").json()["history"]), 2)

    def test_kis_rechecks_vision_choice_against_first_retrieval_candidate(self):
        records = [{"key": str(number)} for number in range(3)]
        seen = []

        class Answerer(AutoAnswer):
            def _review(self, kind, hint, candidates):
                seen.append(candidates)
                if len(candidates) == 3:
                    return Decision(candidates[2], "initial evidence", "")
                return Decision(candidates[0], "checked evidence", "")

            def _confirm_kis(self, hint, record):
                return Decision(record, "independent evidence", "")

        answerer = Answerer(self.cfg)
        self.addCleanup(answerer.close)
        result = answerer.verify("KIS", "clue", records)
        self.assertIs(result.record, records[0])
        self.assertEqual(seen, [records, [records[0], records[2]]])

    def test_mock_qa_requires_confirmation_and_records_trimmed_text_without_index(self):
        with TestClient(create_app(self.cfg)) as c:
            login = c.post('/api/dres/login', json={"mode": "mock"})
            self.assertIn("local-practice-qa", [e["id"] for e in login.json()["evaluations"]])
            c.post('/api/dres/select', json={"evaluation": "local-practice-qa"}).raise_for_status()
            task = c.get('/api/dres/task').json()
            self.assertEqual(task["kind"], "QA")
            body = {"text": "  đỏ  ", "task_ref": task["ref"], "confirmed": True}
            self.assertEqual(c.post('/api/dres/submit-text', json={**body, "confirmed": False}).status_code, 400)
            self.assertEqual(c.post('/api/dres/submit-text', json={**body, "text": "   "}).status_code, 400)
            self.assertEqual(c.post('/api/dres/submit-text', json={**body, "key": "image"}).status_code, 422)
            self.assertEqual(c.get('/api/dres/status').json()["history"], [])
            response = c.post('/api/dres/submit-text', json=body)
            response.raise_for_status()
            entry = response.json()
            self.assertEqual((entry["kind"], entry["text"], entry["verdict"]), ("QA", "đỏ", "MOCK_ACCEPTED"))
            self.assertNotIn("item_name", entry)
            self.assertEqual(c.post('/api/dres/submit-text', json={**body, "text": "đỏ"}).status_code, 409)
            self.assertEqual(c.get('/api/dres/status').json()["history"], [entry])

    def test_model_initialization_failure_can_retry_and_browse(self):
        self.build()
        attempts = []
        def factory(cfg):
            attempts.append(1)
            if len(attempts) == 1:
                raise OSError("checkpoint temporarily unavailable")
            return FakeEncoder(cfg)
        app = create_app(self.cfg, index_factory=lambda cfg: SearchIndex(cfg, factory))
        with TestClient(app) as c:
            self.assertEqual(c.post('/api/search', json={"query": "red"}).status_code, 503)
            self.assertEqual(len(c.post('/api/search', json={}).json()["results"]), 3)
            self.assertTrue(c.get('/api/status').json()["ready"])
            self.assertEqual(c.post('/api/search', json={"query": "red"}).status_code, 200)
            self.assertEqual(len(attempts), 2)

    def test_encoding_failure_preserves_image_search_and_next_query(self):
        self.build()
        class FlakyEncoder(FakeEncoder):
            def text(self, query):
                if query == "fail":
                    raise RuntimeError("CUDA out of memory")
                return super().text(query)
        app = create_app(self.cfg, index_factory=lambda cfg: SearchIndex(cfg, FlakyEncoder))
        with TestClient(app) as c:
            self.assertEqual(c.post('/api/search', json={"query": "fail"}).status_code, 503)
            record = c.post('/api/search', json={}).json()["results"][0]
            self.assertEqual(c.get('/api/image/' + record["key"]).status_code, 200)
            self.assertEqual(c.post('/api/search', json={"image_key": record["key"]}).status_code, 200)
            self.assertEqual(c.post('/api/search', json={"query": "green"}).status_code, 200)

    def gateway(self, response_code=200, verdict="CORRECT", timeout=False, kind="KIS"):
        self.requests=[]
        self.remote_task = {"name": f"{kind}01", "taskId": "task-A",
                            "taskType": "LSC Question & Answer Text" if kind == "QA" else "LSC Known-Item Search"}
        def handler(request):
            self.requests.append(request)
            path=request.url.path
            if path.endswith('/login'):
                return httpx.Response(200,json={"sessionId":"private-session"})
            if path.endswith('/list'):
                return httpx.Response(200,json=[{"id":"run","status":"ACTIVE"}])
            if '/currentTask/' in path:
                return httpx.Response(200, json=self.remote_task)
            if '/submit/' in path:
                if timeout:
                    raise httpx.ReadTimeout("test",request=request)
                return httpx.Response(response_code,json={"submission":verdict})
            return httpx.Response(404)
        gateway=DresGateway(self.cfg,httpx.MockTransport(handler))
        self.addCleanup(gateway.close)
        token=gateway.login('real','user','secret-password')
        session=gateway.session(token)
        gateway.select(session,'run')
        return gateway,session,scan(self.cfg)[0]

    def test_real_submit_exact_payload_and_no_password_retained(self):
        gateway,session,record=self.gateway()
        self.assertEqual(gateway.submit(session,record)["verdict"],"CORRECT")
        sent=json.loads(self.requests[-1].content)
        answer=sent["answerSets"][0]["answers"][0]
        self.assertEqual(answer,{"mediaItemName":record["item_name"],"mediaItemCollectionName":"castle5000"})
        self.assertNotIn("secret-password",str(session.__dict__))
        self.assertNotIn("private-session",(self.cfg.artifacts_dir/'submissions.jsonl').read_text())

    def test_wrong_is_not_http_success_correct(self):
        gateway,session,record=self.gateway(verdict="WRONG")
        self.assertEqual(gateway.submit(session,record)["verdict"],"WRONG")

    def test_pending_202(self):
        gateway,session,record=self.gateway(response_code=202,verdict="INDETERMINATE")
        self.assertEqual(gateway.submit(session,record)["verdict"],"INDETERMINATE")

    def test_qa_sends_only_text_and_keeps_pending_202(self):
        gateway, session, _ = self.gateway(response_code=202, verdict="INDETERMINATE", kind="QA")
        ref = gateway.task_status(session)["ref"]
        result = gateway.submit_text(session, "  3  ", ref)
        self.assertEqual(result["verdict"], "INDETERMINATE")
        self.assertEqual(json.loads(self.requests[-1].content), {"answerSets": [{"answers": [{"text": "3"}]}]})
        self.assertTrue(self.requests[-1].url.path.endswith('/submit/run'))

    def test_submit_blocks_wrong_kind_and_stale_task(self):
        gateway, session, record = self.gateway(kind="QA")
        ref = gateway.task_status(session)["ref"]
        with self.assertRaises(DresError) as wrong_kind:
            gateway.submit(session, record)
        self.assertEqual(wrong_kind.exception.status, 409)
        self.remote_task.update(name="QA02", taskId="task-B")
        with self.assertRaises(DresError) as stale_task:
            gateway.submit_text(session, "3", ref)
        self.assertEqual(stale_task.exception.status, 409)
        self.remote_task.update(name="KIS01", taskType="LSC Known-Item Search")
        with self.assertRaises(DresError) as wrong_kind:
            gateway.submit_text(session, "3", gateway.task_status(session)["ref"])
        self.assertEqual(wrong_kind.exception.status, 409)
        self.assertFalse(any('/submit/' in r.url.path for r in self.requests))

    def test_qa_timeout_keeps_unknown_and_blocks_duplicate(self):
        gateway, session, _ = self.gateway(timeout=True, kind="QA")
        ref = gateway.task_status(session)["ref"]
        with self.assertRaises(DresError) as timed_out:
            gateway.submit_text(session, "3", ref)
        self.assertEqual(timed_out.exception.status, 504)
        self.assertEqual(session.history[0]["verdict"], "UNKNOWN")
        with self.assertRaises(DresError) as duplicate:
            gateway.submit_text(session, "3", ref)
        self.assertEqual(duplicate.exception.status, 409)
        self.assertEqual(sum('/submit/' in r.url.path for r in self.requests), 1)

    def test_expired_remote_session(self):
        gateway,session,record=self.gateway(response_code=401)
        with self.assertRaises(DresError) as exc:
            gateway.submit(session,record)
        self.assertEqual(exc.exception.status,401)
        self.assertEqual(session.expires,0)

    def test_timeout_never_retries_and_keeps_unknown(self):
        gateway,session,record=self.gateway(timeout=True)
        with self.assertRaises(DresError):
            gateway.submit(session,record)
        self.assertEqual(session.history[0]["verdict"],"UNKNOWN")
        with self.assertRaises(DresError):
            gateway.submit(session,record)
        self.assertEqual(sum('/submit/' in r.url.path for r in self.requests),1)

    def test_ambiguous_id_cannot_be_submitted(self):
        gateway,session,record=self.gateway()
        record['ambiguous_item']=True
        with self.assertRaises(DresError):
            gateway.submit(session,record)
        self.assertFalse(any('/submit/' in r.url.path for r in self.requests))

    def test_auto_rechecks_task_and_disabling_invalidates_ticket(self):
        gateway, session, record = self.gateway()
        ref = gateway.task_status(session)["ref"]
        decision = Decision(record, "visible detail", "")
        gateway.set_auto(session, "run", True)
        ticket, kind = gateway.begin_auto(session, ref)
        gateway.save_auto(session, ticket, ref, kind, decision)
        gateway.set_auto(session, "run", False)
        with self.assertRaises(DresError):
            gateway.commit_auto(session, ticket)
        gateway.set_auto(session, "run", True)
        ticket, kind = gateway.begin_auto(session, ref)
        gateway.save_auto(session, ticket, ref, kind, decision)
        self.remote_task["taskId"] = "task-B"
        with self.assertRaises(DresError):
            gateway.commit_auto(session, ticket)
        self.assertFalse(any('/submit/' in r.url.path for r in self.requests))

    def test_automatic_and_manual_race_sends_only_one_qa(self):
        gateway, session, record = self.gateway(response_code=202, kind="QA")
        ref = gateway.task_status(session)["ref"]
        gateway.set_auto(session, "run", True)
        ticket, kind = gateway.begin_auto(session, ref)
        gateway.save_auto(session, ticket, ref, kind, Decision(record, "red cup", "đỏ"))

        def attempt(send):
            try:
                return send()["verdict"]
            except DresError as exc:
                return exc.status

        with ThreadPoolExecutor(max_workers=2) as workers:
            automatic = workers.submit(attempt, lambda: gateway.commit_auto(session, ticket))
            manual = workers.submit(attempt, lambda: gateway.submit_text(session, "xanh", ref))
            self.assertCountEqual([automatic.result(), manual.result()], ["INDETERMINATE", 409])
        self.assertEqual(sum('/submit/' in r.url.path for r in self.requests), 1)
        self.assertEqual(len(session.history), 1)

    def test_automatic_timeout_stays_unknown_and_stops_next_hint(self):
        gateway, session, record = self.gateway(timeout=True, kind="QA")
        ref = gateway.task_status(session)["ref"]
        gateway.set_auto(session, "run", True)
        ticket, kind = gateway.begin_auto(session, ref)
        gateway.save_auto(session, ticket, ref, kind, Decision(record, "red cup", "đỏ"))
        with self.assertRaises(DresError) as timeout:
            gateway.commit_auto(session, ticket)
        self.assertEqual(timeout.exception.status, 504)
        self.assertEqual(session.history[0]["verdict"], "UNKNOWN")
        with self.assertRaises(DresError):
            gateway.begin_auto(session, ref)
        posts = [json.loads(r.content) for r in self.requests if '/submit/' in r.url.path]
        self.assertEqual(posts, [{"answerSets": [{"answers": [{"text": "đỏ"}]}]}])


if __name__=='__main__':
    unittest.main()
