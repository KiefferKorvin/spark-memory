import asyncio
import logging
from uuid import uuid4

from .config import Settings
from .evidence import MemoryEnrichmentService
from .external import build_retriever
from .graph import InMemoryGraph, Neo4jGraph
from .ingestion import IngestionEngine
from .llm import OpenRouter
from .models import IngestRequest, QueryRequest, now
from .retrieval import RetrievalEngine
from .trace import TraceService

logger = logging.getLogger(__name__)


class Memory:
    def __init__(self, settings, repository, models, external=None):
        self.settings, self.repository, self.models = settings, repository, models
        self.ingestion = IngestionEngine(settings, repository, models)
        self.retrieval = RetrievalEngine(settings, repository, models, self.ingestion, external)
        self.enrichment = MemoryEnrichmentService(settings, repository, models)
        self.tasks = {}

    @classmethod
    def from_settings(cls, settings=None):
        settings = settings or Settings()
        settings.validate_live()
        if settings.memory_mode == "demo":
            from .demo import DemoExternal, DemoModels
            return cls(settings, InMemoryGraph(), DemoModels(), DemoExternal())
        # Every networked mode searches online for gaps in memory; only the offline demo uses fixtures.
        external = build_retriever(settings.external_retrievers, settings.max_source_bytes)
        if settings.memory_mode == "online_demo":
            from .demo import OnlineDemoModels
            repository = InMemoryGraph()
            return cls(settings, repository, OnlineDemoModels(OpenRouter(settings, repository)), external)
        repository = Neo4jGraph(settings)
        return cls(settings, repository, OpenRouter(settings, repository), external)

    async def initialize(self):
        await self.repository.initialize()
        if self.settings.unesco_thesaurus_path:
            from .unesco import import_thesaurus
            await import_thesaurus(self.repository, self.settings.unesco_thesaurus_path, self.settings.unesco_label_language)
        if self.settings.memory_mode == "live" and self.settings.unesco_required and not await self.repository.read_record("taxonomy", self.settings.primary_ontology):
            raise ValueError("Import the configured PRIMARY_ONTOLOGY before starting live memory")
        # MVP supports one API process. Durable traces survive; interrupted tasks are explicit failures.
        for query in await self.repository.records("query"):
            if query.get("status") == "running":
                await self.repository.record("query", query["query_id"], {**query, "status": "failed", "error": "Service restarted during query"})

    async def close(self):
        tasks = [*self.tasks.values(), *self.ingestion.background]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await self.models.close()
        await self.repository.close()

    async def ingest(self, request):
        return await self.ingestion.ingest(request if isinstance(request, IngestRequest) else IngestRequest.model_validate(request))

    async def execute(self, query_id, request):
        trace = TraceService(self.repository, query_id)
        try:
            async with asyncio.timeout(self.settings.query_timeout_seconds):
                result = await self.retrieval.query(request, trace)
                if self.settings.enrichment_policy == "frequent":
                    for item in result["evidence"]:
                        node_id = item["source_node_id"]
                        count = await self.repository.hit(node_id, success=False)
                        if count >= self.settings.enrichment_hit_threshold and item["source_type"] in ("chunk", "document"):
                            try:
                                await self.enrichment.enrich(node_id, query_id)
                            except Exception:
                                await trace.emit("ENRICHMENT_FAILED", node_id=node_id)
                await trace.emit("QUERY_COMPLETED", status="completed", coverage=result["coverage"]["overall_status"])
                await self.repository.record("query", query_id, result)
        except asyncio.CancelledError:
            await trace.emit("QUERY_FAILED", reason="cancelled")
            await self.repository.record("query", query_id, {"query_id": query_id, "status": "failed", "error": "Query cancelled during shutdown"})
            raise
        except Exception as exc:
            # Logs keep server diagnostics; API never exposes credentials/provider response bodies.
            logger.warning("Query %s failed: %s", query_id, type(exc).__name__)
            await trace.emit("QUERY_FAILED", reason=type(exc).__name__)
            await self.repository.record("query", query_id, {"query_id": query_id, "status": "failed",
                "error": "Query timed out" if isinstance(exc, TimeoutError) else "Query failed; check server configuration and logs"})

    async def submit(self, request):
        request = QueryRequest(query=request) if isinstance(request, str) else request
        if len(self.tasks) >= self.settings.max_active_queries:
            raise RuntimeError("Query capacity reached")
        query_id = str(uuid4())
        ready = asyncio.get_running_loop().create_future()
        # Register a task without an await gap so concurrent submissions cannot exceed capacity.
        async def run():
            try:
                await self.repository.record("query", query_id, {"query_id": query_id, "status": "running", "created_at": now()})
            except Exception as exc:
                ready.set_exception(exc)
                return
            ready.set_result(None)
            await self.execute(query_id, request)
        task = asyncio.create_task(run())
        self.tasks[query_id] = task
        task.add_done_callback(lambda _: self.tasks.pop(query_id, None))
        await ready  # Initial durable record must exist before handing back the ID.
        return query_id

    async def query(self, request):
        query_id = await self.submit(request)
        task = self.tasks.get(query_id)
        if task:
            await task
        return await self.repository.read_record("query", query_id)

    async def events(self, query_id, after=0):
        return await self.repository.records("event", query_id + ":", after=f"{query_id}:{after:08}")

    async def metrics(self):
        calls = await self.repository.records("usage")
        return {"calls": len(calls), "input_tokens": sum(c["input_tokens"] or 0 for c in calls),
                "output_tokens": sum(c["output_tokens"] or 0 for c in calls),
                "reported_cost": sum(c["estimated_cost"] or 0 for c in calls),
                "calls_without_cost": sum(c["estimated_cost"] is None for c in calls)}
