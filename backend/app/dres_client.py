from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

import httpx

from .config import Settings

# "submitted": accepted by DRES for manual grading (QA text answers); no verdict ever comes back.
Verdict = Literal["correct", "wrong", "pending", "submitted", "rejected", "error"]

# ApiVerdictStatus enum values, from DRES's own OpenAPI spec (dres-dev/DRES,
# doc/oas.json) -> normalized verdicts used by this app's /api/submit.
# INDETERMINATE/UNDECIDABLE both mean "no automatic verdict yet", so both
# collapse to "pending" here.
_VERDICT_MAP: dict[str, Verdict] = {
    "CORRECT": "correct",
    "WRONG": "wrong",
    "INDETERMINATE": "pending",
    "UNDECIDABLE": "pending",
}


class DresError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class DresAuthError(DresError):
    """DRES rejected the session (401). Caller must require a fresh login;
    we never cache the student's password, so we cannot silently retry."""

    def __init__(self, message: str = "DRES session expired, please log in again"):
        super().__init__("dres_unauthorized", message)


@dataclass
class DresUser:
    id: str
    username: str
    role: str
    session_id: str


@dataclass
class DresTask:
    name: str
    task_group: str
    task_type: str
    duration_s: int | None


class DresClient:
    """Thin wrapper over the DRES v2 API.

    Endpoint paths and field names were verified against the public DRES
    OpenAPI spec (github.com/dres-dev/DRES, doc/oas.json) on 2026-10-06:
      POST /api/v2/login                                   -> ApiUser
      GET  /api/v2/logout?session=...                       -> SuccessStatus
      GET  /api/v2/client/evaluation/currentTask/{evalId}   -> ApiClientTaskTemplateInfo
      POST /api/v2/submit/{evalId}?session=...              -> SuccessfulSubmissionsStatus
      GET  /api/v2/status/time                               -> CurrentTime

    Re-verify against the spec served by the actual deployed DRES instance
    before go-live (versions can drift) -- this is exactly the check the
    technical note asks for in section 6.
    """

    def __init__(self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None,
                 base_url: str | None = None):
        self._settings = settings
        self._client = httpx.AsyncClient(
            base_url=base_url or settings.dres_base_url,
            timeout=settings.dres_timeout_s,
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def login(self, username: str, password: str) -> DresUser:
        resp = await self._client.post(
            "/api/v2/login", json={"username": username, "password": password}
        )
        if resp.status_code == 200:
            data = resp.json()
            return DresUser(
                id=data["id"],
                username=data["username"],
                role=data.get("role", "PARTICIPANT"),
                session_id=data["sessionId"],
            )
        if resp.status_code in (400, 401):
            raise DresAuthError(_error_message(resp, "Sai tên đăng nhập hoặc mật khẩu"))
        raise DresError("dres_error", f"DRES login failed: HTTP {resp.status_code}")

    async def logout(self, session_id: str) -> None:
        await self._client.get("/api/v2/logout", params={"session": session_id})

    async def current_task(self, evaluation_id: str, session_id: str) -> DresTask | None:
        resp = await self._client.get(
            f"/api/v2/client/evaluation/currentTask/{evaluation_id}",
            params={"session": session_id},
        )
        if resp.status_code == 404:
            return None
        if resp.status_code == 401:
            raise DresAuthError()
        if resp.status_code != 200:
            raise DresError("dres_error", f"DRES currentTask failed: HTTP {resp.status_code}")
        data = resp.json()
        return DresTask(
            name=data["name"],
            task_group=data["taskGroup"],
            task_type=data["taskType"],
            duration_s=data.get("duration"),
        )

    async def submit(
        self,
        evaluation_id: str,
        session_id: str,
        media_item_name: str,
        collection_name: str | None = None,
    ) -> tuple[Verdict, str]:
        answer: dict[str, str] = {"mediaItemName": media_item_name}
        if collection_name:
            answer["mediaItemCollectionName"] = collection_name
        return await self._submit_answer(evaluation_id, session_id, answer)

    async def submit_text(self, evaluation_id: str, session_id: str, text: str) -> tuple[Verdict, str]:
        """QA answers are graded by hand, so DRES never reports correct/wrong for them: any accepted
        submission is just "submitted", whatever placeholder verdict DRES returns."""
        verdict, message = await self._submit_answer(evaluation_id, session_id, {"text": text})
        if verdict in ("correct", "wrong", "pending"):
            return "submitted", "Đã nộp. Giáo viên chấm tay nên không có kết quả trả về."
        return verdict, message

    async def _submit_answer(self, evaluation_id: str, session_id: str, answer: dict[str, str]) -> tuple[Verdict, str]:
        body = {"answerSets": [{"answers": [answer]}]}

        resp = await self._client.post(
            f"/api/v2/submit/{evaluation_id}",
            params={"session": session_id},
            json=body,
        )

        if resp.status_code in (200, 202):
            data = resp.json()
            if resp.status_code == 202:
                return "pending", data.get("description", "Đã nhận, đang chờ chấm")
            verdict = _VERDICT_MAP.get(data.get("submission", ""), "pending")
            return verdict, data.get("description", "")
        if resp.status_code == 401:
            raise DresAuthError()
        return _explain_failure(resp)

    async def evaluations(self, session_id: str) -> list[dict]:
        resp = await self._client.get('/api/v2/client/evaluation/list', params={'session': session_id})
        if resp.status_code == 401:
            raise DresAuthError()
        if resp.status_code != 200:
            raise DresError('dres_error', f'DRES evaluations: HTTP {resp.status_code}')
        return resp.json()

    async def server_time(self) -> int:
        resp = await self._client.get("/api/v2/status/time")
        resp.raise_for_status()
        return resp.json()["timeStamp"]


def is_qa_task(task_type: str | None) -> bool:
    """Task types are named by the organisers ("LSC Question & Answer Text", "QA", "Q&A", ...)."""
    t = (task_type or "").casefold()
    return "question" in t or "answer" in t or bool(re.search(r"(^|[^a-z])q\s*&?\s*a([^a-z]|$)", t))


def _explain_failure(resp: httpx.Response) -> tuple[Verdict, str]:
    """Turn a refused submission into what the team should do (DRES guide, sections 7.3 and 9.2)."""
    raw = _error_message(resp, "")
    low = raw.casefold()
    code = resp.status_code
    if "media item" in low:
        verdict, text = "error", "Sai tên ảnh hoặc collection (phải là day1_Kitchen_HH_frame trong castle5000)"
    elif code == 400 and "state" in low:
        verdict, text = "rejected", "Không có task đang chạy (chưa bắt đầu hoặc đã hết giờ)"
    elif code == 404:
        verdict, text = "error", "Sai Evaluation ID, hoặc tài khoản không thuộc buổi thi này"
    elif code == 412:
        verdict, text = "rejected", "Bị từ chối: đã nộp đáp án này rồi, hoặc team đã có bài đúng cho task này"
    else:
        verdict, text = "error", f"DRES trả lỗi HTTP {code}"
    return verdict, f"{text} (DRES: {raw})" if raw else text


def _error_message(resp: httpx.Response, fallback: str) -> str:
    try:
        return resp.json().get("description", fallback)
    except Exception:
        return fallback
