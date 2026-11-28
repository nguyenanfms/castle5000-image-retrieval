import json
import hashlib
import secrets
import threading
import time
from dataclasses import dataclass, field
from urllib.parse import quote

import httpx


PRACTICE_TASKS = {
    "local-practice": ("KIS · Luyện tìm ảnh và nộp thử", "LSC Known-Item Search"),
    "local-practice-qa": ("QA · Luyện nộp đáp án chữ", "LSC Question & Answer Text"),
}


def task_kind(task):
    task_type = str(task.get("taskType", "")).strip().casefold()
    if task_type == "lsc known-item search":
        return "KIS"
    if task_type == "lsc question & answer text":
        return "QA"
    if not task_type:
        name = str(task.get("name", "")).strip().upper()
        for kind in ("KIS", "QA"):
            if name.startswith(kind):
                return kind
    return None


def task_ref(evaluation, task):
    identity = [evaluation, {k: task.get(k) for k in ("id", "taskId", "name", "taskGroup", "taskType")}]
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()


class DresError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


@dataclass
class Session:
    mode: str
    remote: str = ""
    username: str = "local"
    evaluation: str = ""
    expires: float = 0
    attempted: set = field(default_factory=set)
    history: list = field(default_factory=list)
    lock: object = field(default_factory=threading.RLock)
    auto_enabled: bool = False
    auto_ticket: str = ""
    auto_draft: tuple | None = None
    auto_sent: set = field(default_factory=set)


class DresGateway:
    def __init__(self, cfg, transport=None):
        self.cfg = cfg
        self.sessions = {}
        self.client = (httpx.Client(base_url=cfg.dres_url.rstrip("/") + "/", timeout=12,
                                   transport=transport, follow_redirects=False) if transport else None)

    def close(self):
        if self.client:
            self.client.close()

    def _request(self, method, path, session=None, body=None):
        try:
            if self.client is None:
                self.client = httpx.Client(base_url=self.cfg.dres_url.rstrip("/")+"/", timeout=12,
                                           follow_redirects=False)
            response = self.client.request(method, path,
                params={"session": session.remote} if session else None, json=body)
        except httpx.TimeoutException:
            raise DresError("DRES qua thoi gian cho. Neu vua nop: ket qua chua ro, xem Viewer truoc khi nop tiep.", 504) from None
        except httpx.HTTPError:
            raise DresError("Khong ket noi duoc DRES. Kiem tra mang va dres_url.", 502) from None
        except ImportError:
            raise DresError("Proxy thieu dependency. Cai requirements.txt (httpx[socks]) roi thu lai.", 503) from None
        if response.status_code == 401:
            if session:
                session.expires = 0
            raise DresError("Phien DRES het han. Dang nhap lai.", 401)
        if response.status_code not in (200, 202):
            explanations = {
                400: "DRES 400: chua co task dang chay hoac request khong hop le.",
                404: "DRES 404: sai evaluation hoac tai khoan khong thuoc buoi thi.",
                412: "DRES 412: sai loại bài, bài trùng hoặc nhóm đã có bài đúng. Kiểm tra Viewer.",
                500: "DRES 500: kiểm tra loại bài, mã ảnh (với KIS) hoặc tình trạng máy chủ.",
            }
            raise DresError(explanations.get(response.status_code, f"DRES HTTP {response.status_code}"),
                            response.status_code if response.status_code < 500 else 502)
        try:
            return response.status_code, response.json()
        except ValueError:
            raise DresError("DRES tra ve du lieu khong phai JSON. Khong tu dong nop lai.", 502) from None

    def login(self, mode, username="", password=""):
        if mode not in ("mock", "real"):
            raise DresError("Che do khong hop le.")
        remote = ""
        if mode == "real":
            if not username.strip() or not password:
                raise DresError("Nhap tai khoan va mat khau DRES.")
            _, payload = self._request("POST", "login", body={"username": username.strip(), "password": password})
            remote = payload.get("sessionId", "")
            if not remote:
                raise DresError("DRES khong tra ve sessionId.", 502)
        token = secrets.token_urlsafe(32)
        # Bound local session storage; passwords are never retained.
        self.sessions = {k: s for k, s in self.sessions.items() if s.expires > time.time()}
        self.sessions[token] = Session(mode=mode, remote=remote, username=username.strip() or "local",
                                       expires=time.time()+8*3600)
        return token

    def session(self, token):
        session = self.sessions.get(token)
        if not session or session.expires < time.time():
            raise DresError("Chua dang nhap hoac phien da het han.", 401)
        return session

    def evaluations(self, session):
        if session.mode == "mock":
            return [{"id": evaluation, "name": name, "status": "ACTIVE"}
                    for evaluation, (name, _) in PRACTICE_TASKS.items()]
        _, payload = self._request("GET", "client/evaluation/list", session)
        if not isinstance(payload, list):
            raise DresError("Danh sach evaluation khong dung dinh dang.", 502)
        return [{"id": str(e["id"]), "name": str(e.get("name", e["id"])),
                 "status": str(e.get("status", "UNKNOWN"))} for e in payload]

    def select(self, session, evaluation):
        if not any(e["id"] == evaluation for e in self.evaluations(session)):
            raise DresError("Tai khoan khong co evaluation nay.", 404)
        with session.lock:
            if evaluation != session.evaluation:
                session.auto_enabled = False
                session.auto_ticket = ""
                session.auto_draft = None
            session.evaluation = evaluation

    def set_auto(self, session, evaluation, enabled):
        with session.lock:
            if not session.evaluation or evaluation != session.evaluation:
                raise DresError("Buổi thi đã đổi. Chọn lại evaluation.", 409)
            session.auto_enabled = enabled
            session.auto_ticket = ""
            session.auto_draft = None
            return enabled

    def begin_auto(self, session, expected_task):
        with session.lock:
            if not session.auto_enabled:
                raise DresError("Chưa bật tự nộp cho buổi thi này.", 409)
            task = self.task_status(session)
            self._check_auto_task(session, task, expected_task)
            session.auto_ticket = secrets.token_urlsafe(24)
            session.auto_draft = None
            return session.auto_ticket, task["kind"]

    def save_auto(self, session, ticket, expected_task, kind, decision):
        with session.lock:
            if not session.auto_enabled or session.auto_ticket != ticket:
                raise DresError("Gợi ý đã đổi hoặc tự nộp đã tắt.", 409)
            if decision is not None:
                session.auto_draft = (expected_task, kind, decision)

    def commit_auto(self, session, ticket):
        with session.lock:
            if not session.auto_enabled or session.auto_ticket != ticket or session.auto_draft is None:
                raise DresError("Đáp án tự động đã hết hiệu lực.", 409)
            expected_task, kind, decision = session.auto_draft
            session.auto_ticket = ""
            session.auto_draft = None
            if kind == "KIS":
                return self.submit(session, decision.record, expected_task, automatic=decision)
            return self.submit_text(session, decision.answer, expected_task, automatic=decision)

    @staticmethod
    def _check_auto_task(session, task, expected_task):
        if task["ref"] != expected_task or task["kind"] not in ("KIS", "QA"):
            raise DresError("Task đã đổi hoặc không xác định được loại bài.", 409)
        if expected_task in session.auto_sent or any(ref == expected_task for ref, _, _ in session.attempted):
            raise DresError("Task này đã có lượt nộp. Xem lịch sử.", 409)

    def task(self, session):
        if not session.evaluation:
            return None
        if session.mode == "mock":
            name, task_type = PRACTICE_TASKS[session.evaluation]
            return {"name": name, "taskId": session.evaluation, "taskType": task_type, "duration": None}
        try:
            _, payload = self._request("GET", "client/evaluation/currentTask/" + quote(session.evaluation, safe=""), session)
        except DresError as exc:
            if exc.status == 404:
                return None
            raise
        return payload if isinstance(payload, dict) else None

    def task_status(self, session):
        with session.lock:
            task = self.task(session)
            return {"task": task, "kind": task_kind(task) if task else None,
                    "ref": task_ref(session.evaluation, task) if task else None}

    def submit(self, session, record, expected_task=None, *, automatic=None):
        if not record["item_name"]:
            raise DresError("Ten file khong dung mau; can metadata imageID truoc khi nop.")
        if record["ambiguous_item"]:
            raise DresError("Ma DRES bi trung giua cac anh. Sua metadata truoc khi nop.")
        answer = {"mediaItemName": record["item_name"], "mediaItemCollectionName": self.cfg.collection}
        entry = {"item_name": record["item_name"], "filename": record["filename"]}
        return self._submit(session, "KIS", answer, record["item_name"], entry, expected_task, automatic)

    def submit_text(self, session, text, expected_task, *, automatic=None):
        text = text.strip()
        if not text:
            raise DresError("Nhập đáp án trước khi nộp.")
        if not expected_task:
            raise DresError("Cập nhật task QA trước khi nộp.", 409)
        return self._submit(session, "QA", {"text": text}, text, {"text": text}, expected_task, automatic)

    def _submit(self, session, kind, answer, value, details, expected_task, automatic):
        with session.lock:
            if not session.evaluation:
                raise DresError("Chon evaluation truoc khi nop.")
            active = any(e["id"] == session.evaluation and e["status"] == "ACTIVE" for e in self.evaluations(session))
            if not active:
                raise DresError("Evaluation chua ACTIVE.", 409)
            current = self.task_status(session)
            if not current["task"]:
                raise DresError("Chưa có task đang chạy. Kiểm tra DRES Viewer.", 409)
            if current["kind"] != kind:
                raise DresError(f"Chỉ nộp loại bài này khi task {kind} đang chạy. Kiểm tra DRES Viewer.", 409)
            if expected_task is not None and current["ref"] != expected_task:
                raise DresError("Task đã đổi. Kiểm tra lại câu hỏi trước khi nộp.", 409)
            if automatic is not None:
                self._check_auto_task(session, current, expected_task)
                session.auto_sent.add(expected_task)
                details = {**details, "automatic": True, "evidence": automatic.observation,
                           "evidence_item": automatic.record["item_name"]}
            elif session.auto_enabled and current["ref"] in session.auto_sent:
                raise DresError("Task đã có lượt tự nộp. Xem lịch sử trước khi tắt chế độ và nộp tay.", 409)
            guard = (current["ref"], kind, value)
            if guard in session.attempted:
                raise DresError("Bài này đã gửi trong task hiện tại. Kiểm tra lịch sử hoặc DRES Viewer.", 409)
            session.attempted.add(guard)
            entry = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "mode": session.mode,
                     "evaluation": session.evaluation, "kind": kind, "task_name": current["task"].get("name"),
                     **details, "verdict": "UNKNOWN"}
            try:
                if session.mode == "mock":
                    entry.update(verdict="MOCK_ACCEPTED", message="Da mo phong nop. Khong gui DRES va chua cham dung/sai.")
                else:
                    status, payload = self._request("POST", "submit/" + quote(session.evaluation, safe=""), session,
                        {"answerSets": [{"answers": [answer]}]})
                    verdict = payload.get("submission") if isinstance(payload, dict) else None
                    if status == 202:
                        verdict = "INDETERMINATE"
                    if verdict not in ("CORRECT", "WRONG", "INDETERMINATE", "UNDECIDABLE"):
                        raise DresError("DRES trả kết quả chưa xác định. Kiểm tra Viewer trước khi nộp tiếp.", 502)
                    entry.update(verdict=verdict, message={"CORRECT": "Bài nộp đúng.", "WRONG": "Bài nộp sai.",
                        "INDETERMINATE": "Đã nhận, đang chờ chấm.", "UNDECIDABLE": "Cần trọng tài chấm."}[verdict])
            except DresError as exc:
                entry["message"] = str(exc)
                if exc.status < 500:
                    session.attempted.discard(guard)
                    entry["verdict"] = "REJECTED"
                # Timeout/5xx/invalid JSON remains UNKNOWN; never retry automatically.
                raise
            finally:
                session.history.insert(0, entry)
                try:
                    self.cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)
                    with (self.cfg.artifacts_dir / "submissions.jsonl").open("a", encoding="utf-8") as f:
                        f.write(json.dumps(entry, ensure_ascii=False)+"\n")
                except OSError:
                    entry["log_warning"] = "Khong ghi duoc lich su ra dia; ket qua DRES van giu trong phien."
            return entry
