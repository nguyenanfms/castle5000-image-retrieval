from __future__ import annotations

import asyncio
import json
import httpx
import logging
from logging.handlers import RotatingFileHandler
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import get_settings
from .dres_client import DresClient
from .gateway import DresGateway, GatewayError
from .metadata import MetadataStore
from .qa import QAError, QAService
from .ratelimit import RateLimiter
from .schemas import (
    ImageMetadataResponse,
    LoginRequest,
    LoginResponse,
    QARequest,
    QAResponse,
    QAStatusResponse,
    SearchRequest,
    SearchPlan,
    SearchPlanRequest,
    ParseQueryRequest,
    SearchResponse,
    SubmitRequest,
    SubmitTextRequest,
    SubmitResponse,
    TaskResponse,
)
from .search import IndexNotBuiltError, SearchEngine
from .search_plan import SearchPlanError, SearchPlanner
from .query_parser import split_query
from .sessions import LocalSession, SessionStore

logger = logging.getLogger("retrieval")

settings = get_settings()


class _TaskCache:
    def __init__(self, ttl_s: float = 2.0):
        self._ttl = ttl_s
        self._value: TaskResponse | None = None
        self._expires_at = 0.0

    def get(self) -> TaskResponse | None:
        if time.time() < self._expires_at:
            return self._value
        return None

    def set(self, value: TaskResponse) -> None:
        self._value = value
        self._expires_at = time.time() + self._ttl


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Persist timings and error codes only: no prompts, images, API keys or URLs.
    qa_log = logging.getLogger('retrieval.qa')
    qa_log_path = Path(__file__).resolve().parents[2] / 'output' / 'qa-api.log'
    log_handler = None
    try:
        qa_log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handler = RotatingFileHandler(qa_log_path, maxBytes=2_000_000, backupCount=2, encoding='utf-8')
        log_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s %(message)s'))
        qa_log.addHandler(log_handler)
        qa_log.setLevel(logging.INFO)
    except OSError:
        logger.warning('Could not create QA diagnostic log')
    app.state.qa_local_mode = settings.qa_local_mode
    if settings.using_default_secret:
        logger.warning(
            "SESSION_SECRET is using the insecure dev default; set it via env in production."
        )
    app.state.search_engine = SearchEngine(settings)
    app.state.metadata_store = MetadataStore(settings.metadata_path)
    app.state.dres_client = DresClient(settings)
    app.state.gateway = DresGateway(
        settings, app.state.dres_client, SessionStore(settings)
    )
    app.state.search_limiter = RateLimiter(settings.rate_limit_per_min)
    app.state.plan_limiter = RateLimiter(settings.rate_limit_per_min)
    app.state.search_planner = SearchPlanner(settings)
    app.state.submit_limiter = RateLimiter(settings.rate_limit_per_min)
    app.state.qa_limiter = RateLimiter(settings.qa_rate_limit_per_min)
    try:
        app.state.qa_service = QAService(
            settings, app.state.search_engine, app.state.metadata_store
        )
    except Exception as exc:
        logger.warning("QAService failed to initialize and is unavailable: %s", exc)
        app.state.qa_service = None
    app.state.task_cache = _TaskCache()
    app.state.ready = True
    logger.info("Loaded index with %d images", app.state.search_engine.num_images)
    yield
    if getattr(app.state, "qa_service", None) is not None:
        await app.state.qa_service.aclose()
    await app.state.search_planner.aclose()
    await app.state.dres_client.aclose()
    if log_handler is not None:
        qa_log.removeHandler(log_handler)
        log_handler.close()


app = FastAPI(title="Randomly Retrieval - From text to random image =)))", lifespan=lifespan)


def _error(
    code: str, message: str, status_code: int, headers: dict[str, str] | None = None
) -> JSONResponse:
    return JSONResponse(
        {"code": code, "message": message}, status_code=status_code, headers=headers
    )


@app.exception_handler(httpx.RequestError)
async def on_connection_error(request: Request, exc: httpx.RequestError):
    return _error('dres_connection', 'Không kết nối được DRES. Kiểm tra mạng và thử lại.', 502)


@app.exception_handler(RequestValidationError)
async def on_validation_error(request: Request, exc: RequestValidationError):
    return _error("invalid_request", "Dữ liệu gửi lên không hợp lệ", 422)


@app.exception_handler(GatewayError)
async def on_gateway_error(request: Request, exc: GatewayError):
    return _error(exc.code, exc.message, exc.status)


@app.exception_handler(QAError)
async def on_qa_error(request: Request, exc: QAError):
    return _error(exc.code, exc.message, exc.status, headers=exc.headers)


@app.exception_handler(HTTPException)
async def on_http_exception(request: Request, exc: HTTPException):
    detail = exc.detail
    if isinstance(detail, dict) and "code" in detail:
        return _error(detail["code"], detail["message"], exc.status_code)
    return _error("error", str(detail), exc.status_code)


def _cookie(request: Request) -> str | None:
    return request.cookies.get(settings.session_cookie_name)


def _require_session(request: Request) -> LocalSession:
    if getattr(request.app.state, 'qa_local_mode', False):
        if not request.client or request.client.host not in {'127.0.0.1', '::1'}:
            raise HTTPException(403, 'Chế độ thử QA chỉ dùng trên localhost')
        return LocalSession(
            dres_session_id='', dres_user_id='local-qa',
            username='Thử QA trên máy', role='LOCAL_QA', created_at=time.time(),
        )
    return request.app.state.gateway.require_session(_cookie(request))


def _require_competition(request: Request) -> None:
    if getattr(request.app.state, 'qa_local_mode', False):
        raise HTTPException(409, {'code': 'competition_disabled', 'message': 'Chế độ thử QA không kết nối DRES'})


@app.get("/health")
async def health(request: Request):
    if not getattr(request.app.state, "ready", False):
        raise HTTPException(503, {"code": "not_ready", "message": "Đang nạp index"})
    return {"status": "ok", "images": request.app.state.search_engine.num_images, "region_ready": request.app.state.search_engine.region_index is not None}


@app.get('/api/region-status')
async def region_status(request: Request):
    _require_session(request)
    engine = request.app.state.search_engine
    await asyncio.to_thread(engine.refresh_regions)
    progress = {}
    try:
        progress = json.loads((settings.region_index_dir / 'progress.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        pass
    return {'ready': engine.region_index is not None, 'processed': progress.get('processed', 0), 'total': progress.get('total', engine.num_images), 'errors': len(progress.get('errors', []))}


@app.post("/api/login", response_model=LoginResponse)
async def login(req: LoginRequest, request: Request, response: Response):
    if getattr(request.app.state, 'qa_local_mode', False):
        session = _require_session(request)
        return LoginResponse(username=session.username, role=session.role)
    cookie_value, session = await request.app.state.gateway.login(req.username, req.password)
    response.set_cookie(
        settings.session_cookie_name,
        cookie_value,
        httponly=True,
        secure=settings.session_cookie_secure,
        samesite="lax",
        max_age=settings.session_max_age_s,
    )
    return LoginResponse(username=session.username, role=session.role)


@app.get("/api/me", response_model=LoginResponse)
async def me(request: Request):
    session = _require_session(request)
    return LoginResponse(username=session.username, role=session.role)


@app.post("/api/logout")
async def logout(request: Request, response: Response):
    if not getattr(request.app.state, 'qa_local_mode', False):
        await request.app.state.gateway.logout(_cookie(request))
    response.delete_cookie(settings.session_cookie_name)
    return {"status": "ok", "description": "logged out"}


@app.post('/api/parse-query')
async def parse_query(req: ParseQueryRequest, request: Request):
    _require_session(request)
    return {'detail_queries': split_query(req.query), 'method': 'english_clauses'}


@app.post("/api/search", response_model=SearchResponse)
async def search(req: SearchRequest, request: Request):
    session = _require_session(request)

    if not request.app.state.search_limiter.allow(session.username):
        raise HTTPException(
            429, {"code": "rate_limited", "message": "Tìm kiếm quá nhanh, vui lòng chờ"}
        )

    engine: SearchEngine = request.app.state.search_engine
    details = req.detail_queries or (split_query(req.query) if req.auto_split else [])
    if req.mode != 'legacy':
        results, took_ms, mode, version, checked = await engine.search_demo(
            req.query or '. '.join(details), req.k, details, req.mode, req.conditions,
        )
        notice = None
        if req.mode == 'clip_region' and mode == 'clip':
            notice = ('Chưa có điều kiện hoặc cụm chi tiết để kiểm tra.' if not details and not req.conditions
                      else 'Dữ liệu kiểm tra vùng chưa sẵn sàng.')
        elif req.conditions and mode == 'clip_region' and len(results) < req.k:
            notice = (f'Còn {len(results)}/{req.k} ảnh sau khi kiểm tra {checked} ứng viên và lọc vi phạm.'
                      if results else f'Không còn ảnh sau khi kiểm tra {checked} ứng viên và lọc vi phạm.')
        return SearchResponse(
            results=[{**row, 'thumb': f"/thumbs/{row['id']}"} for row in results],
            took_ms=took_ms, detail_queries=details, region_ready=version is not None,
            mode=mode, notice=notice, index_version=version,
            candidate_k=checked,
            region_threshold=settings.search_region_threshold if mode == 'clip_region' else None,
        )
    results, took_ms = await engine.search(req.query, req.k, details)
    return SearchResponse(
        results=[
            {**r, "thumb": f"/thumbs/{r['id']}"} for r in results
        ],
        took_ms=took_ms,
        detail_queries=details,
        region_ready=engine.region_index is not None,
        notice=engine.region_notice if details else None,
    )


@app.post('/api/search/plan', response_model=SearchPlan)
async def plan_search(req: SearchPlanRequest, request: Request):
    session = _require_session(request)
    if not request.app.state.plan_limiter.allow(session.username):
        return _error('rate_limited', 'Đã đạt giới hạn lượt tạo mô tả. Bạn có thể nhập tiếng Anh.', 429)
    try:
        return await request.app.state.search_planner.plan(req.query, req.language)
    except SearchPlanError as exc:
        return _error('search_plan_unavailable', str(exc), 503)


@app.get("/api/qa/status", response_model=QAStatusResponse)
async def qa_status(request: Request):
    _require_session(request)
    qa_service: QAService | None = getattr(request.app.state, "qa_service", None)
    if qa_service is None:
        return QAStatusResponse(
            enabled=bool(settings.qa_enabled),
            ready=False,
            supported_modes=[],
            max_question_length=settings.qa_question_max_len,
        )
    return qa_service.status_info()


def _qa_owner(request: Request, session: LocalSession) -> str:
    import hashlib
    return hashlib.sha256(((session.username or "") + ":" + (_cookie(request) or "local")).encode()).hexdigest()


@app.get("/api/qa/progress/{conversation_id}")
async def qa_progress(conversation_id: str, request: Request):
    session = _require_session(request)
    service = getattr(request.app.state, "qa_service", None)
    if service is None:
        return {"stage": "idle"}
    service.reasoning._clean()
    progress = service.reasoning.progress.get((_qa_owner(request, session), conversation_id), {})
    return {"stage": progress.get("stage", "idle"), "request_id": progress.get("request_id"),
            "image_ids": progress.get("image_ids", []), "examined_ids": progress.get("examined_ids", [])}


@app.post("/api/qa", response_model=QAResponse)
async def qa_endpoint(req: QARequest, request: Request):
    session = _require_session(request)
    qa_limiter: RateLimiter = request.app.state.qa_limiter
    if not qa_limiter.allow(session.username):
        retry_after = qa_limiter.retry_after(session.username)
        raise QAError(
            429,
            "qa_rate_limited",
            f"Bạn đã gửi quá nhiều câu hỏi. Vui lòng chờ {retry_after} giây.",
            headers={"Retry-After": str(retry_after)},
        )

    qa_service: QAService | None = getattr(request.app.state, "qa_service", None)
    if qa_service is None:
        raise QAError(503, "qa_unavailable", "Dịch vụ hỏi đáp hiện không khả dụng")

    return await qa_service.answer(req, access_scope=_qa_owner(request, session))


@app.get('/api/evaluations')
async def evaluations(request: Request):
    session = _require_session(request)
    if session.role == 'LOCAL_QA':
        return []
    with request.app.state.gateway._dres_errors(_cookie(request)):
        return await request.app.state.dres_client.evaluations(session.dres_session_id)


@app.post('/api/evaluation/{evaluation_id}')
async def select_evaluation(evaluation_id: str, request: Request):
    session = _require_session(request)
    _require_competition(request)
    from uuid import UUID
    try:
        evaluation_id = str(UUID(evaluation_id))
    except ValueError:
        raise HTTPException(422, 'Run ID phải là mã UUID của buổi thi trên DRES')
    session.evaluation_id = evaluation_id
    return {'status': 'ok'}


@app.get("/api/task", response_model=TaskResponse)
async def get_task(request: Request):
    session = _require_session(request)
    if session.role == 'LOCAL_QA':
        return TaskResponse(running=False)

    if not (_require_session(request).evaluation_id or settings.dres_evaluation_id):
        return TaskResponse(running=False)

    cached = None  # Task depends on the evaluation selected by this session.
    if cached is not None:
        return cached

    task, server_time_ms = await request.app.state.gateway.current_task(_cookie(request))

    if task is None:
        result = TaskResponse(running=False, server_time_ms=server_time_ms)
    else:
        result = TaskResponse(
            running=True,
            name=task.name,
            task_group=task.task_group,
            task_type=task.task_type,
            duration_s=task.duration_s,
            server_time_ms=server_time_ms,
        )
    request.app.state.task_cache.set(result)
    return result


@app.post("/api/submit", response_model=SubmitResponse)
async def submit(req: SubmitRequest, request: Request):
    session = _require_session(request)
    _require_competition(request)

    if not request.app.state.submit_limiter.allow(session.username):
        raise HTTPException(
            429, {"code": "rate_limited", "message": "Submit quá nhanh, vui lòng chờ"}
        )

    engine: SearchEngine = request.app.state.search_engine
    if not engine.has_id(req.id):
        raise HTTPException(
            400, {"code": "unknown_id", "message": "ID ảnh không hợp lệ"}
        )

    verdict, message = await request.app.state.gateway.submit(_cookie(request), req.id)
    return SubmitResponse(verdict=verdict, message=message)


@app.post("/api/submit/text", response_model=SubmitResponse)
async def submit_text(req: SubmitTextRequest, request: Request):
    session = _require_session(request)
    _require_competition(request)
    if not request.app.state.submit_limiter.allow(session.username):
        raise HTTPException(429, {"code": "rate_limited", "message": "Submit quá nhanh, vui lòng chờ"})
    verdict, message = await request.app.state.gateway.submit_text(_cookie(request), req.text)
    return SubmitResponse(verdict=verdict, message=message)


@app.get("/api/metadata/{image_id}", response_model=ImageMetadataResponse)
async def get_metadata(image_id: str, request: Request):
    _require_session(request)
    engine: SearchEngine = request.app.state.search_engine
    if not engine.has_id(image_id):
        raise HTTPException(404, {"code": "not_found", "message": "Không có ảnh này"})
    store: MetadataStore = request.app.state.metadata_store
    meta = store.get(image_id)
    if meta is None:
        raise HTTPException(404, {"code": "no_metadata", "message": "Không có metadata cho ảnh này"})
    return ImageMetadataResponse(
        owner=meta.owner,
        day=meta.day,
        hour=meta.hour,
        frame=meta.frame,
        classes=meta.classes,
        class_count=meta.class_count,
        description=meta.description,
    )


@app.get('/api/regions/{image_id}')
async def inspect_regions(image_id: str, request: Request):
    _require_session(request)
    engine = request.app.state.search_engine
    if not engine.has_id(image_id):
        raise HTTPException(404, 'Không có ảnh này')
    await asyncio.to_thread(engine.refresh_regions)
    if engine.region_index is None:
        raise HTTPException(409, 'Chỉ mục vùng chưa sẵn sàng')
    regions = [r for r in engine.region_index.records if r['id'] == image_id]
    detections = {}
    for region in regions:
        for obj in region.get('objects', []):
            key = (obj['label'], tuple(round(v, 6) for v in obj['box']))
            if key not in detections or obj['score'] > detections[key]['score']:
                detections[key] = obj
    return {'id': image_id, 'regions': regions,
            'objects': sorted(detections.values(), key=lambda obj: -obj['score']),
            'detector': getattr(engine.region_index, 'manifest', {}).get('detector', 'Grounding DINO Tiny'),
            'encoder': getattr(engine.region_index, 'manifest', {}).get('model', 'CLIP ViT-B/32')}


@app.get("/images/{image_id}")
async def get_image(image_id: str, request: Request):
    _require_session(request)
    engine: SearchEngine = request.app.state.search_engine
    if not engine.has_id(image_id):
        raise HTTPException(404, {"code": "not_found", "message": "Không có ảnh này"})
    path = settings.images_dir / image_id
    if not path.exists():
        raise HTTPException(404, {"code": "not_found", "message": "Không có ảnh này"})
    return FileResponse(path, media_type="image/jpeg")


@app.get("/thumbs/{image_id}")
async def get_thumb(image_id: str, request: Request):
    _require_session(request)
    engine: SearchEngine = request.app.state.search_engine
    if not engine.has_id(image_id):
        raise HTTPException(404, {"code": "not_found", "message": "Không có ảnh này"})
    path = settings.thumbs_dir / image_id
    if not path.exists():
        raise HTTPException(404, {"code": "not_found", "message": "Không có ảnh này"})
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=604800"})


_frontend_dir = Path(__file__).resolve().parents[2] / "frontend"
if _frontend_dir.exists():
    @app.get('/search-v2/')
    async def search_v2():
        return FileResponse(_frontend_dir / 'index.html', media_type='text/html',
                            headers={'Cache-Control': 'no-cache'})

    @app.get('/search-v2', include_in_schema=False)
    async def redirect_search_v2():
        return Response(status_code=307, headers={'Location': '/search-v2/'})

    app.mount("/", StaticFiles(directory=_frontend_dir, html=True), name="frontend")
