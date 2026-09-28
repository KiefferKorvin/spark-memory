"""HTTP API of the clinical deployment (docs/CLINICAL.md): uvicorn graph_memory.clinical_api:create_app --factory.

A separate app on the shared core: none of PAKT's memory endpoints (online search, recipes, images, episodes) exist here.
Every route needs the bearer token, and startup refuses unsafe settings (ClinicalSettings.check).
"""
import base64
import binascii
import hmac
from contextlib import asynccontextmanager
import datetime as dt
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Path, Request
from pydantic import Field

from .api import BodyLimit
from .clinical import Clinical, ClinicalSettings, ModelError
from .models import Strict
from .parsing import extract

PID = Path(pattern=r"^[A-Za-z0-9._-]{1,64}$")
DOC = Path(pattern=r"^[0-9a-f]{16}$")
ENCOUNTER = Field(None, pattern=r"^[A-Za-z0-9._-]{1,64}$")


class NoteRequest(Strict):
    text: str = Field(min_length=1, max_length=200_000)
    date: dt.date
    title: str = Field("", max_length=300)
    author: str = Field("", max_length=200)
    encounter: str | None = ENCOUNTER


class FhirRequest(Strict):
    resources: list[dict] = Field(min_length=1, max_length=500)
    encounter: str | None = ENCOUNTER


class GuidelineRequest(Strict):
    title: str = Field(min_length=1, max_length=300)
    issuer: str = Field("", max_length=100)
    version: str = Field("", max_length=50)
    published: dt.date | None = None
    text: str | None = Field(None, max_length=3_000_000)
    content_base64: str | None = None    # a PDF or DOCX, read with the memory's own parser
    mime: Literal["application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                  "text/plain", "text/markdown"] = "text/plain"


class AskRequest(Strict):
    question: str = Field(min_length=1, max_length=2000)
    encounter: str | None = ENCOUNTER
    answer: bool = True
    date: dt.date | None = None   # the day the question is asked (default today), for time cues and records


def create_app(service=None):
    settings = service.s if service else ClinicalSettings()
    settings.check()

    @asynccontextmanager
    async def lifespan(app):
        app.state.clinical = service or Clinical.from_settings(settings)
        await app.state.clinical.initialize()
        try:
            yield
        finally:
            await app.state.clinical.close()

    async def authorize(request: Request):
        if not hmac.compare_digest(request.headers.get("authorization", ""),
                                   "Bearer " + settings.memory_api_token.get_secret_value()):
            raise HTTPException(401, "Bearer token required")

    app = FastAPI(title="Clinical SPARK", version="0.1.0", lifespan=lifespan, dependencies=[Depends(authorize)])
    app.add_middleware(BodyLimit, limit=settings.max_source_bytes * 2 + 16384)

    def clinical(request: Request) -> Clinical:
        return request.app.state.clinical

    async def guarded(call):
        try:
            return await call
        except ModelError as exc:
            raise HTTPException(502, f"Model provider failed: {exc}") from exc

    @app.get("/health")
    async def health(request: Request):
        c = clinical(request)
        return {"status": "ok", "data_mode": settings.clinical_data_mode, "terminology": c.terms.manifest["source_file"].split("\\")[-1]}

    @app.post("/clinical/patients/{pid}/notes")
    async def note(body: NoteRequest, request: Request, pid: str = PID):
        return await guarded(clinical(request).add_note(pid, body.text, body.date, body.title, body.author, body.encounter))

    @app.post("/clinical/patients/{pid}/fhir")
    async def fhir(body: FhirRequest, request: Request, pid: str = PID):
        return await guarded(clinical(request).add_fhir(pid, body.resources, body.encounter))

    @app.get("/clinical/patients/{pid}/header")
    async def header(request: Request, pid: str = PID):
        _, docs = await clinical(request).graph(f"patient:{pid}")
        return clinical(request).header(docs)

    @app.post("/clinical/patients/{pid}/ask")
    async def ask(body: AskRequest, request: Request, pid: str = PID):
        return await guarded(clinical(request).ask(pid, body.question, body.encounter, body.answer, body.date))

    @app.delete("/clinical/patients/{pid}")
    async def forget_patient(request: Request, pid: str = PID):
        await clinical(request).forget_patient(pid)
        return {"forgotten": pid}

    @app.delete("/clinical/patients/{pid}/documents/{doc}")
    async def forget_document(request: Request, pid: str = PID, doc: str = DOC):
        await clinical(request).forget_document(pid, doc)
        return {"forgotten": doc}

    @app.post("/clinical/guidelines")
    async def guideline(body: GuidelineRequest, request: Request):
        text = body.text
        if body.content_base64:
            try:
                data = base64.b64decode(body.content_base64, validate=True)
                text = extract(data, body.mime, settings.max_source_bytes)
            except (binascii.Error, ValueError) as exc:
                raise HTTPException(422, f"Unreadable guideline: {exc}") from exc
        if not text or not text.strip():
            raise HTTPException(422, "Send text or content_base64")
        return await guarded(clinical(request).add_guideline(text, body.title, body.issuer, body.version, body.published))

    @app.get("/clinical/guidelines")
    async def guidelines(request: Request):
        return await clinical(request).guidelines()

    @app.delete("/clinical/guidelines/{doc}")
    async def forget_guideline(request: Request, doc: str = DOC):
        await clinical(request).forget_guideline(doc)
        return {"forgotten": doc}

    return app
