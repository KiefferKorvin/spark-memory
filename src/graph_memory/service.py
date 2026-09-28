import asyncio
import logging
import time
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from .config import Settings
from .evidence import MemoryEnrichmentService
from .external import build_retriever
from .graph import InMemoryGraph, Neo4jGraph
from .images import ImageMemory
from .spark import SparkMemory
from .ingestion import IngestionEngine
from .llm import AccountError, OpenRouter, ProviderError
from .models import FactProposals, IngestRequest, QueryRequest, now
from .procedures import host
from .retrieval import RetrievalEngine, url_key
from .trace import TraceService

logger = logging.getLogger(__name__)


class Memory:
    def __init__(self, settings, repository, models, external=None):
        self.settings, self.repository, self.models = settings, repository, models
        self.ingestion = IngestionEngine(settings, repository, models)
        self.retrieval = RetrievalEngine(settings, repository, models, self.ingestion, external)
        self.enrichment = MemoryEnrichmentService(settings, repository, models)
        self.images = ImageMemory(settings, repository, models)
        self.spark = SparkMemory(settings, repository, models)
        self.tasks, self.swept = {}, 0.0

    @classmethod
    def from_settings(cls, settings=None):
        settings = settings or Settings()
        settings.validate_live()
        if settings.memory_mode == "demo":
            from .demo import DemoExternal, DemoModels
            return cls(settings, InMemoryGraph(), DemoModels(), DemoExternal())
        # Every networked mode searches online for gaps in memory; only the offline demo uses fixtures.
        external = build_retriever(settings.external_retrievers, settings.max_source_bytes,
                                   settings.pakt_web_browser, settings.pakt_obscura_command)
        if settings.memory_mode == "online_demo":
            from .demo import OnlineDemoModels
            repository = InMemoryGraph()
            return cls(settings, repository, OnlineDemoModels(OpenRouter(settings, repository)), external)
        repository = Neo4jGraph(settings)
        return cls(settings, repository, OpenRouter(settings, repository), external)

    async def initialize(self):
        await self.repository.initialize()
        from .recipes import migrate
        await migrate(self.repository)
        if self.settings.unesco_thesaurus_path:
            from .unesco import import_thesaurus
            await import_thesaurus(self.repository, self.settings.unesco_thesaurus_path, self.settings.unesco_label_language)
        if self.settings.memory_mode == "live" and self.settings.unesco_required and not await self.repository.read_record("taxonomy", self.settings.primary_ontology):
            raise ValueError("Import the configured PRIMARY_ONTOLOGY before starting live memory")
        # MVP supports one API process. Durable traces survive; interrupted tasks are explicit failures.
        for query in await self.repository.records("query"):
            if query.get("status") == "running":
                await self.repository.record("query", query["query_id"], {**query, "status": "failed", "error": "Service restarted during query"})
        await self.sweep()

    async def close(self):
        tasks = [*self.tasks.values(), *self.ingestion.background]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await self.models.close()
        await self.repository.close()

    async def ingest(self, request):
        request = request if isinstance(request, IngestRequest) else IngestRequest.model_validate(request)
        result = await self.ingestion.ingest(request)
        await self.changed(request.scope)  # a duplicate still adds a source, which changes provenance-filtered results
        return result

    async def execute(self, query_id, request, created_at=None):
        trace = TraceService(self.repository, query_id)
        created_at = created_at or now()
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
                await trace.emit("QUERY_COMPLETED", status="completed", coverage=(result["coverage"] or {}).get("overall_status"))
                await self.repository.record("query", query_id, {**result, "created_at": created_at})
        except asyncio.CancelledError:
            await trace.emit("QUERY_FAILED", reason="cancelled")
            await self.repository.record("query", query_id, {"query_id": query_id, "status": "failed", "created_at": created_at,
                                                             "error": "Query cancelled during shutdown"})
            raise
        except Exception as exc:
            # Logs keep server diagnostics; API never exposes credentials/provider response bodies. The recorded reason
            # is the exception type, or a provider error's own message (operation, attempts, HTTP status or error type),
            # since container logs do not survive a redeploy.
            logger.warning("Query %s failed", query_id, exc_info=exc)
            reason = str(exc)[:300] if isinstance(exc, (ProviderError, AccountError)) else type(exc).__name__
            await trace.emit("QUERY_FAILED", reason=reason)
            await self.repository.record("query", query_id, {"query_id": query_id, "status": "failed", "reason": reason, "created_at": created_at,
                "error": "Query timed out" if isinstance(exc, TimeoutError) else "Query failed; check server configuration and logs"})
        if time.monotonic() - self.swept > 86400:
            await self.sweep()

    async def submit(self, request):
        request = QueryRequest(query=request) if isinstance(request, str) else request
        if len(self.tasks) >= self.settings.max_active_queries:
            raise RuntimeError("Query capacity reached")
        query_id = str(uuid4())
        ready = asyncio.get_running_loop().create_future()
        # Register a task without an await gap so concurrent submissions cannot exceed capacity.
        async def run():
            created_at = now()
            try:
                await self.repository.record("query", query_id, {"query_id": query_id, "status": "running", "created_at": created_at})
            except Exception as exc:
                ready.set_exception(exc)
                return
            ready.set_result(None)
            await self.execute(query_id, request, created_at)
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

    async def forget(self, reason, document_id=None, url=None, hard=False):
        """Excludes documents from retrieval (reversible with restore) or erases them (hard). Their web addresses are
        remembered, so no query's online search fetches them again. Returns what was forgotten."""
        documents = [document_id] if document_id else await self.repository.documents_from(url)
        urls, at = {url} if url else set(), now()
        for document in documents:
            urls |= {s["uri"] for s in (await self.repository.provenance(document))["sources"] if s.get("uri")}
            if hard:
                await self.repository.delete_document(document)
                continue
            nodes = await self.repository.document_nodes(document)
            for node in nodes:
                node.metadata["excluded"] = {"reason": reason, "at": at}
            await self.repository.update_nodes(nodes)
        for address in urls:
            await self.repository.record("forgotten", url_key(address), {"url": address, "reason": reason, "at": at,
                                                                         "documents": documents, "hard": hard})
            await self.retrieval.knowhow.count("host", host(address), forgotten=1)
        await self.changed("shared")
        return {"documents": documents, "urls": sorted(urls), "hard": hard}

    async def restore(self, document_id):
        """Undoes a (soft) forget: the document is retrievable again and its addresses may be fetched again."""
        nodes = await self.repository.document_nodes(document_id)
        for node in nodes:
            node.metadata.pop("excluded", None)
        await self.repository.update_nodes(nodes)
        for source in (await self.repository.provenance(document_id))["sources"]:
            if source.get("uri"):
                await self.repository.delete_records("forgotten", url_key(source["uri"]))
        await self.changed("shared")
        return {"documents": [document_id] if nodes else []}

    async def erase_scope(self, scope):
        """Erases a private scope: its documents and everything recorded for it (episodes with their query records
        and traces, searches, user facts, images, associative memories). The shared memory is not a scope that can be erased."""
        if scope == "shared":
            raise ValueError("The shared memory cannot be erased as a scope")
        for episode in await self.repository.records("episode", scope + "|"):
            for query_id in episode.get("query_ids", []):
                await self.repository.delete_records("event", query_id + ":")
                await self.repository.delete_records("query", query_id)
        await self.repository.delete_scope(scope)
        for category in ("episode", "search", "fact", "state", "image", "image-data", "spark"):
            await self.repository.delete_records(category, scope + "|")
        self.spark.forget(scope)
        return {"scope": scope, "erased": True}

    async def facts(self, scope):
        """A private scope's user memory: what the client stated and what was inferred, with status and evidence."""
        return sorted(await self.repository.records("fact", scope + "|"), key=lambda f: f["created_at"])

    async def add_fact(self, scope, kind, text, origin="stated", status="active", evidence=()):
        existing = next((f for f in await self.facts(scope) if f["text"].casefold() == text.casefold() and f["status"] != "rejected"), None)
        if existing:
            return existing
        at, fact_id = now(), str(uuid4())
        fact = {"id": fact_id, "scope": scope, "kind": kind, "text": text, "origin": origin, "status": status,
                "evidence": list(evidence), "created_at": at, "updated_at": at}
        await self.repository.record("fact", f"{scope}|{fact_id}", fact)
        if status == "active":
            await self.changed(scope)
        return fact

    async def update_fact(self, scope, fact_id, changes):
        fact = await self.repository.read_record("fact", f"{scope}|{fact_id}")
        if fact is None:
            raise LookupError("Unknown fact")
        fact = {**fact, **{k: v for k, v in changes.items() if v is not None}, "updated_at": now()}
        await self.repository.record("fact", f"{scope}|{fact_id}", fact)
        await self.changed(scope)
        return fact

    async def delete_fact(self, scope, fact_id):
        if not await self.repository.read_record("fact", f"{scope}|{fact_id}"):
            raise LookupError("Unknown fact")
        await self.repository.delete_records("fact", f"{scope}|{fact_id}")
        await self.changed(scope)

    async def infer_facts(self, scope, limit=50):
        """Proposes facts from the scope's recent questions (one model call). Proposals apply only once confirmed."""
        episodes = (await self.episodes(scope, limit=limit))[::-1]
        if len(episodes) < 2:
            return []
        known = [f["text"] for f in await self.facts(scope) if f["status"] != "rejected"]
        found = await self.models.structured("user_facts", {
            "questions": {str(i): e["question"] for i, e in enumerate(episodes, 1)}, "existing_facts": known}, FactProposals)
        proposals = []
        for fact in found.facts:
            evidence = [episodes[i - 1]["question"] for i in fact.evidence if 0 < i <= len(episodes)]
            if evidence:  # a proposal must point at the questions it comes from
                proposals.append(await self.add_fact(scope, fact.kind, fact.text, "inferred", "proposed", evidence))
        return proposals

    async def changed(self, scope):
        """Marks what a scope's queries can read as changed, so no earlier answer is reused over it (episodes)."""
        await self.repository.record("state", scope + "|changed", {"at": now()})

    async def episodes(self, scope="shared", question=None, limit=20):
        """What was asked recently in a scope (latest first), or the episodes closest to a question, with similarity."""
        if question:
            vector = await self.models.embed(question)
            found = [{**e, "similarity": round(s, 3)} for e, s in await self.repository.similar_records("episode", vector, limit * 5)
                     if e["scope"] == scope]
            return found[:limit]
        return sorted(await self.repository.records("episode", scope + "|"), key=lambda e: e["asked_at"], reverse=True)[:limit]

    async def sweep(self):
        """Deletes query records and their traces older than TRACE_RETENTION_DAYS, and episodes not asked for
        EPISODE_RETENTION_DAYS; usage records stay for cost accounting. Runs at startup, then at most daily."""
        self.swept = time.monotonic()
        if self.settings.episode_retention_days:
            cutoff = (datetime.now(timezone.utc) - timedelta(days=self.settings.episode_retention_days)).isoformat()
            for episode in await self.repository.records("episode"):
                if episode["asked_at"] < cutoff:
                    await self.repository.delete_records("episode", episode["key"])
        if not self.settings.trace_retention_days:
            return 0
        cutoff, swept = (datetime.now(timezone.utc) - timedelta(days=self.settings.trace_retention_days)).isoformat(), 0
        for query in await self.repository.records("query"):
            if query.get("status") == "running":
                continue
            created = query.get("created_at") or ((await self.repository.read_record("event", query["query_id"] + ":00000001")) or {}).get("timestamp")
            if created and created < cutoff:
                await self.repository.delete_records("event", query["query_id"] + ":")
                await self.repository.delete_records("query", query["query_id"])
                swept += 1
        return swept

    async def events(self, query_id, after=0):
        return await self.repository.records("event", query_id + ":", after=f"{query_id}:{after:08}")

    async def metrics(self):
        calls, searches = await self.repository.records("usage"), await self.repository.records("search")
        return {"calls": len(calls), "input_tokens": sum(c["input_tokens"] or 0 for c in calls),
                "output_tokens": sum(c["output_tokens"] or 0 for c in calls),
                "reported_cost": sum(c["estimated_cost"] or 0 for c in calls),
                "calls_without_cost": sum(c["estimated_cost"] is None for c in calls),
                # Search memory: needs searched online (latest search per need), searches skipped because a need had
                # been searched recently, and searches whose sources reached an answer's context.
                "searches": {"needs": len(searches), "skipped_as_recent": sum(s.get("reused", 0) for s in searches),
                             "reached_context": sum(bool(s.get("used")) for s in searches)}}
