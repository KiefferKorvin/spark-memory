import asyncio
import hmac
import json
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.types import ASGIApp

from .config import Settings
from .llm import ProviderError
from .models import IngestRequest, QueryRequest
from .service import Memory


class BodyLimit:
    def __init__(self, app: ASGIApp, limit: int):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        # Buffer only up to the limit before JSON parsing, including chunked requests.
        chunks, size = [], 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            size += len(message.get("body", b""))
            if size > self.limit:
                return await JSONResponse({"detail": "Request body too large"}, 413)(scope, receive, send)
            chunks.append(message)
            if not message.get("more_body", False):
                break
        async def replay():
            return chunks.pop(0) if chunks else await receive()
        await self.app(scope, replay, send)


def create_app(memory=None):
    settings = memory.settings if memory else Settings()

    @asynccontextmanager
    async def lifespan(app):
        app.state.memory = memory or Memory.from_settings(settings)
        await app.state.memory.initialize()
        try:
            yield
        finally:
            await app.state.memory.close()

    async def authorize(request: Request):
        expected = settings.memory_api_token.get_secret_value()
        if expected and not hmac.compare_digest(request.headers.get("authorization", ""), "Bearer " + expected):
            raise HTTPException(401, "Bearer token required")

    app = FastAPI(title="Progressive Graph Memory", version="0.1.0", lifespan=lifespan,
                  dependencies=[Depends(authorize)])
    app.add_middleware(BodyLimit, limit=settings.max_source_bytes*2 + 16384)

    def service(request):
        return request.app.state.memory

    @app.exception_handler(ValueError)
    async def invalid(request, exc):
        return JSONResponse({"detail": str(exc)[:300]}, status_code=422)

    @app.exception_handler(ProviderError)
    async def provider_failed(request, exc):
        return JSONResponse({"detail": "Model provider unavailable or returned invalid data"}, status_code=502)

    @app.get("/memory/health")
    async def health():
        return {"mode": settings.memory_mode, "status": "ready"}

    @app.post("/memory/ingest")
    async def ingest(body: IngestRequest, request: Request):
        return await service(request).ingest(body)

    @app.get('/memory/recipes')
    async def recipe_lookup(request: Request, q: str = Query('',max_length=2000), limit: int = Query(100,ge=1,le=200)):
        from .recipes import lookup
        return await lookup(service(request).repository,q,limit)

    @app.post('/memory/recipes')
    async def recipe_ingest(body: IngestRequest, request: Request):
        from .recipes import ingest
        return await ingest(service(request).repository,body)

    @app.post("/memory/query", status_code=202)
    async def query(body: QueryRequest, request: Request):
        try:
            query_id = await service(request).submit(body)
        except RuntimeError as exc:
            raise HTTPException(429, str(exc)) from exc
        return {"query_id": query_id, "status": "running"}

    @app.get("/memory/query/{query_id}")
    async def query_result(query_id: str, request: Request):
        result = await service(request).repository.read_record("query", query_id)
        if result is None:
            raise HTTPException(404, "Unknown query")
        return result

    @app.get("/memory/query/{query_id}/events")
    async def events(query_id: str, request: Request, after: int = Query(0, ge=0)):
        await query_result(query_id, request)
        return await service(request).events(query_id, after)

    @app.get("/memory/query/{query_id}/stream")
    async def stream(query_id: str, request: Request, after: int = Query(0, ge=0)):
        await query_result(query_id, request)
        try:
            after = max(after, int(request.headers.get("last-event-id", "0")))
        except ValueError as exc:
            raise HTTPException(422, "Invalid Last-Event-ID") from exc
        async def generate():
            cursor = after
            while not await request.is_disconnected():
                # Read terminal status first, then drain events so completion cannot hide the last event.
                result = await service(request).repository.read_record("query", query_id)
                batch = await service(request).events(query_id, cursor)
                for event in batch:
                    cursor = event["sequence"]
                    yield f"id: {cursor}\ndata: {json.dumps(event)}\n\n"
                if result["status"] != "running":
                    yield "event: done\ndata: {}\n\n"
                    return
                yield ": heartbeat\n\n"
                await asyncio.sleep(0.25)
        return StreamingResponse(generate(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/memory/nodes/{node_id}")
    async def node(node_id: str, request: Request):
        result = await service(request).repository.get(node_id)
        if result is None:
            raise HTTPException(404, "Unknown node")
        return {**result.model_dump(exclude={"embedding"}), "provenance": await service(request).repository.provenance(node_id)}

    @app.get("/memory/nodes/{node_id}/neighbors")
    async def neighbors(node_id: str, request: Request, limit: int = Query(100, ge=1, le=500)):
        await node(node_id, request)
        nodes, edges = await service(request).repository.neighbors(node_id, limit)
        return {"nodes": [n.model_dump(exclude={"embedding"}) for n in nodes],
                "edges": [{**e.model_dump(), "id": e.id} for e in edges], "limit": limit}

    @app.get("/memory/documents/{node_id}")
    async def document(node_id: str, request: Request):
        result = await node(node_id, request)
        if result["kind"] != "Document":
            raise HTTPException(404, "Unknown document")
        return result

    @app.get("/memory/concepts/{node_id}")
    async def concept(node_id: str, request: Request):
        result = await node(node_id, request)
        if result["kind"] != "Concept":
            raise HTTPException(404, "Unknown concept")
        return result

    @app.post("/memory/nodes/{node_id}/enrich")
    async def enrich(node_id: str, request: Request):
        return [n.model_dump(exclude={"embedding"}) for n in await service(request).enrichment.enrich(node_id, force=True)]

    @app.get("/memory/taxonomy")
    async def taxonomy(request: Request):
        repository = service(request).repository
        return {"authority": settings.primary_ontology, "available": await repository.records("taxonomy"), "manifest": await repository.read_record("taxonomy", settings.primary_ontology),
                "roots": [n.model_dump(exclude={"embedding"}) for n in await repository.taxonomy_roots(settings.primary_ontology)]}

    @app.get("/memory/metrics")
    async def metrics(request: Request):
        return await service(request).metrics()

    @app.post("/memory/demo/seed")
    async def seed(request: Request):
        if settings.memory_mode not in ("demo", "online_demo"):
            raise HTTPException(404)
        from .demo import seed
        return await seed(service(request))

    return app
