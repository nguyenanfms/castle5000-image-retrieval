from contextlib import asynccontextmanager
import time
from typing import Literal

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from .auto_answer import AutoAnswer, AutoAnswerError
from .config import load_config
from .dres import DresGateway, DresError
from .index import SearchIndex


class SearchBody(BaseModel):
    query: str = Field(default="", max_length=2000)
    k: int = Field(default=60, ge=1, le=200)
    batch: str = ""
    hour: int | None = Field(default=None, ge=0, le=23)
    image_key: str | None = None
    mode: Literal["clip", "hybrid"] = "clip"


class LoginBody(BaseModel):
    mode: str = "mock"
    username: str = Field(default="", max_length=200)
    password: str = Field(default="", max_length=1000)


class SelectionBody(BaseModel):
    evaluation: str = Field(max_length=200)


class SubmitBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str = Field(max_length=100)
    confirmed: bool = False
    task_ref: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")


class SubmitTextBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(max_length=2000)
    confirmed: bool = False
    task_ref: str = Field(pattern=r"^[a-f0-9]{64}$")


class AutoModeBody(SelectionBody):
    enabled: bool


class AutoHintBody(BaseModel):
    hint: str = Field(min_length=1, max_length=2000)
    task_ref: str = Field(pattern=r"^[a-f0-9]{64}$")
    mode: Literal["clip", "hybrid"] = "clip"


class AutoCommitBody(BaseModel):
    ticket: str = Field(pattern=r"^[a-zA-Z0-9_-]{32}$")


def create_app(cfg=None, index_factory=SearchIndex, gateway_factory=DresGateway,
               auto_answer_factory=AutoAnswer):
    cfg = cfg or load_config()

    @asynccontextmanager
    async def lifespan(app):
        try:
            app.state.index = index_factory(cfg)
            app.state.index_error = None
        except (OSError, ValueError, KeyError) as exc:
            app.state.index = None
            app.state.index_error = str(exc)
        app.state.gateway = gateway_factory(cfg)
        app.state.auto_answer = None
        app.state.auto_error = None
        try:
            app.state.auto_answer = auto_answer_factory(cfg)
        except ValueError as exc:
            app.state.auto_error = str(exc)
        try:
            yield
        finally:
            app.state.gateway.close()
            if app.state.auto_answer is not None:
                app.state.auto_answer.close()

    app = FastAPI(title="Castle Local · DS314", lifespan=lifespan)

    @app.middleware("http")
    async def same_origin(request, call_next):
        origin = request.headers.get("origin")
        if request.method == "POST" and origin and origin != str(request.base_url).rstrip("/"):
            return Response("Origin khong hop le", status_code=403)
        return await call_next(request)

    def index():
        if app.state.index is None:
            raise HTTPException(503, "Chua co index hop le. Chay python -m castle build, sau do khoi dong lai.")
        return app.state.index

    def gateway_session(request):
        return app.state.gateway.session(request.cookies.get("castle_session"))

    @app.exception_handler(DresError)
    async def dres_exception(request, exc):
        from fastapi.responses import JSONResponse
        return JSONResponse({"detail": str(exc)}, status_code=exc.status)

    @app.get("/api/status")
    def status():
        idx = app.state.index
        matcher = idx.metadata_matcher if idx else None
        return {"ready": idx is not None, "index_error": app.state.index_error,
                "model": cfg.model_id, "device_setting": cfg.device,
                "device": idx.encoder.device if idx and idx.encoder else "Chua nap model",
                "stats": idx.stats if idx else None,
                "batches": sorted({r["batch"] for r in idx.records}) if idx else [],
                "metadata": {"available": matcher is not None,
                             "coverage": matcher.coverage if matcher else 0,
                             "error": idx.metadata_error if idx else None},
                "dres_url": cfg.dres_url, "collection": cfg.collection,
                "auto_provider": app.state.auto_answer.provider if app.state.auto_answer else None}

    @app.post("/api/search")
    def search(body: SearchBody):
        start = time.perf_counter()
        idx = index()
        try:
            results = idx.search(**body.model_dump())
        except KeyError:
            raise HTTPException(404, "Khong tim thay anh.") from None
        except (RuntimeError, ValueError, ImportError, OSError) as exc:
            raise HTTPException(503, "Khong nap/ma hoa duoc model. Chay doctor va xem terminal: " + str(exc)[:300]) from None
        mode = idx.effective_mode(body.mode, body.query, body.image_key)
        notice = ("CLIP + BM25: diem anh va diem metadata dung de xep hang." if mode == "hybrid" else
                  "CLIP baseline: nen dung mo ta tieng Anh ngan; diem cosine khong phai xac suat dung.")
        return {"results": results, "took_ms": round((time.perf_counter()-start)*1000),
                "search_mode": mode, "notice": notice}

    @app.get("/api/neighbors/{key}")
    def neighbors(key: str):
        try:
            return {"results": index().neighbors(key)}
        except KeyError:
            raise HTTPException(404, "Khong tim thay anh.") from None

    @app.get("/api/image/{key}")
    def image(key: str, thumb: bool = False):
        try:
            record = index().record(key)
        except KeyError:
            raise HTTPException(404, "Khong tim thay anh.") from None
        source = (cfg.data_dir / record["path"]).resolve()
        if not source.is_relative_to(cfg.data_dir.resolve()):
            raise HTTPException(403)
        path = cfg.artifacts_dir / "thumbs" / (record["fingerprint"]+".jpg") if thumb else source
        if not path.is_file():
            raise HTTPException(404, "Anh da bi di chuyen. Build lai index.")
        return FileResponse(path, headers={"Cache-Control": "private, max-age=3600"})

    @app.post("/api/dres/login")
    def login(body: LoginBody, request: Request, response: Response):
        token = app.state.gateway.login(body.mode, body.username, body.password)
        previous = request.cookies.get("castle_session")
        if previous:
            app.state.gateway.sessions.pop(previous, None)
        response.set_cookie("castle_session", token, httponly=True, samesite="strict", max_age=8*3600)
        session = app.state.gateway.session(token)
        return {"mode": session.mode, "username": session.username,
                "evaluations": app.state.gateway.evaluations(session)}

    @app.get("/api/dres/status")
    def dres_status(request: Request):
        try:
            session = gateway_session(request)
        except DresError:
            return {"logged_in": False}
        return {"logged_in": True, "mode": session.mode, "username": session.username,
                "evaluation": session.evaluation, "history": session.history[:30],
                "auto_enabled": session.auto_enabled}

    @app.get("/api/dres/evaluations")
    def evaluations(request: Request):
        return app.state.gateway.evaluations(gateway_session(request))

    @app.post("/api/dres/select")
    def select(body: SelectionBody, request: Request):
        session = gateway_session(request)
        app.state.gateway.select(session, body.evaluation)
        return {"evaluation": session.evaluation}

    @app.get("/api/dres/task")
    def task(request: Request):
        return app.state.gateway.task_status(gateway_session(request))

    @app.post("/api/dres/submit")
    def submit(body: SubmitBody, request: Request):
        if not body.confirmed:
            raise HTTPException(400, "Can xac nhan anh truoc khi nop.")
        session = gateway_session(request)
        try:
            record = index().record(body.key)
        except KeyError:
            raise HTTPException(404, "Khong tim thay anh.") from None
        return app.state.gateway.submit(session, record, body.task_ref)

    @app.post("/api/dres/submit-text")
    def submit_text(body: SubmitTextBody, request: Request):
        if not body.confirmed:
            raise HTTPException(400, "Cần xác nhận đáp án trước khi nộp.")
        return app.state.gateway.submit_text(gateway_session(request), body.text, body.task_ref)

    @app.post("/api/dres/auto/mode")
    def auto_mode(body: AutoModeBody, request: Request):
        return {"auto_enabled": app.state.gateway.set_auto(
            gateway_session(request), body.evaluation, body.enabled)}

    @app.post("/api/dres/auto/prepare")
    def auto_prepare(body: AutoHintBody, request: Request):
        hint = body.hint.strip()
        if not hint:
            raise HTTPException(400, "Nhập gợi ý hoặc câu hỏi đang hiển thị.")
        session = gateway_session(request)
        answerer = app.state.auto_answer
        if answerer is None:
            raise HTTPException(503, app.state.auto_error or "Chưa cấu hình model.")
        idx = index()
        gateway = app.state.gateway
        ticket, kind = gateway.begin_auto(session, body.task_ref)
        try:
            query = answerer.search_query(kind, hint)
            gateway.save_auto(session, ticket, body.task_ref, kind, None)
            matches = idx.search(query, k=24, mode=body.mode)
            candidates = [idx.record(match["key"]) for match in matches
                          if match["item_name"] and not match["ambiguous_item"]][:8]
            decision = answerer.verify(kind, hint, candidates)
        except (AutoAnswerError, RuntimeError, OSError, ValueError, ImportError) as exc:
            raise HTTPException(503, str(exc)[:300]) from None
        gateway.save_auto(session, ticket, body.task_ref, kind, decision)
        preview = None
        if decision is not None:
            preview = {"evidence": idx.public(decision.record), "observation": decision.observation,
                       "answer": decision.answer if kind == "QA" else decision.record["item_name"]}
        return {"ticket": ticket if preview else None, "kind": kind, "query": query,
                "search_mode": idx.effective_mode(body.mode, query),
                "results": matches, "preview": preview}

    @app.post("/api/dres/auto/commit")
    def auto_commit(body: AutoCommitBody, request: Request):
        return app.state.gateway.commit_auto(gateway_session(request), body.ticket)

    @app.post("/api/dres/logout")
    def logout(request: Request, response: Response):
        # Local logout only, avoiding invalidating the same session in DRES Viewer.
        app.state.gateway.sessions.pop(request.cookies.get("castle_session"), None)
        response.delete_cookie("castle_session")
        return {"ok": True}

    app.mount("/", StaticFiles(directory=cfg.root / "web", html=True), name="web")
    return app
