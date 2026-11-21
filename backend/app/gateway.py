from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from .config import Settings
from .dres_client import DresAuthError, DresClient, DresError, DresTask, Verdict
from .sessions import LocalSession, SessionStore


class GatewayError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


class DresGateway:
    """Single interface between a browser session cookie and DRES.

    Owns the session lookup and the DRES-to-HTTP error translation (expired
    DRES session -> forget it and demand a fresh login; any other DRES
    failure -> 502), so request handlers never repeat that logic.
    """

    def __init__(self, settings: Settings, client: DresClient, sessions: SessionStore,
                 demo_client: DresClient | None = None, demo_evaluation_id: str | None = None):
        self._settings = settings
        self._client = client
        self._clients = {"real": client}
        if demo_client is not None:
            self._clients["demo"] = demo_client
        self._demo_evaluation_id = demo_evaluation_id
        self._sessions = sessions

    @property
    def modes(self) -> list[str]:
        return list(self._clients)

    def _client_for(self, session: LocalSession) -> DresClient:
        return self._client if session.mode == "real" else self._clients[session.mode]

    def require_session(self, cookie: str | None) -> LocalSession:
        session = self._sessions.resolve(cookie)
        if session is None:
            raise GatewayError(401, "unauthorized", "Chưa đăng nhập hoặc phiên đã hết hạn")
        return session

    async def login(self, username: str, password: str, mode: str = "real") -> tuple[str, LocalSession]:
        client = self._clients.get(mode)
        if client is None:
            raise GatewayError(400, "mode_unavailable", "Chế độ đăng nhập này chưa được bật")
        try:
            user = await client.login(username, password)
        except DresAuthError as exc:
            raise GatewayError(401, exc.code, exc.message)
        except DresError as exc:
            raise GatewayError(502, exc.code, exc.message)
        cookie = self._sessions.create(
            dres_session_id=user.session_id,
            dres_user_id=user.id,
            username=user.username,
            role=user.role,
            mode=mode,
        )
        return cookie, self._sessions.resolve(cookie)

    async def logout(self, cookie: str | None) -> None:
        session = self._sessions.resolve(cookie)
        if session is not None:
            await self._client_for(session).logout(session.dres_session_id)
        self._sessions.destroy(cookie)

    async def current_task(self, cookie: str | None) -> tuple[DresTask | None, int]:
        session = self.require_session(cookie)
        with self._dres_errors(cookie):
            task = await self._client_for(session).current_task(
                self.evaluation_id(session), session.dres_session_id
            )
            server_time_ms = await self._client_for(session).server_time()
        return task, server_time_ms

    async def submit(self, cookie: str | None, image_id: str) -> tuple[Verdict, str]:
        session = self.require_session(cookie)
        with self._dres_errors(cookie):
            return await self._client_for(session).submit(
                self.evaluation_id(session), session.dres_session_id, "day1_Kitchen_" + image_id.rsplit(".", 1)[0], "castle5000"
            )

    async def submit_text(self, cookie: str | None, text: str) -> tuple[Verdict, str]:
        session = self.require_session(cookie)
        evaluation_id = self.evaluation_id(session)
        with self._dres_errors(cookie):
            task = await self._client_for(session).current_task(evaluation_id, session.dres_session_id)
            if task is None:
                raise GatewayError(409, "no_active_task", "Chưa có task đang chạy trong buổi thi.")
            task_type = task.task_type.casefold()
            if "question" not in task_type or "answer" not in task_type:
                raise GatewayError(409, "wrong_task_type", "Task hiện tại không phải QA hoặc chưa xác định được loại task.")
            return await self._client_for(session).submit_text(evaluation_id, session.dres_session_id, text)

    async def evaluations(self, cookie: str | None) -> list[dict]:
        session = self.require_session(cookie)
        with self._dres_errors(cookie):
            return await self._client_for(session).evaluations(session.dres_session_id)

    def has_evaluation(self, session: LocalSession) -> bool:
        return bool(session.evaluation_id or (self._demo_evaluation_id if session.mode == "demo"
                                              else self._settings.dres_evaluation_id))

    def evaluation_id(self, session: LocalSession) -> str:
        if session.evaluation_id:
            return session.evaluation_id
        if session.mode == "demo" and self._demo_evaluation_id:
            return self._demo_evaluation_id
        return self._require_evaluation_id()

    def _require_evaluation_id(self) -> str:
        evaluation_id = self._settings.dres_evaluation_id
        if not evaluation_id:
            raise GatewayError(409, "no_evaluation", "Chưa cấu hình DRES_EVALUATION_ID")
        return evaluation_id

    @contextmanager
    def _dres_errors(self, cookie: str | None) -> Iterator[None]:
        try:
            yield
        except DresAuthError:
            # The password is never cached, so we cannot silently re-login.
            self._sessions.destroy(cookie)
            raise GatewayError(
                401, "session_expired", "Phiên DRES đã hết hạn, hãy đăng nhập lại"
            )
        except DresError as exc:
            raise GatewayError(502, exc.code, exc.message)
