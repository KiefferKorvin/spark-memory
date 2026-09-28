import asyncio
import logging
import weakref
from collections import Counter, defaultdict

from .models import Edge, Relation, Understanding, SourceMetadata, node_from, stable_id
from .ontology import CONCURRENCY, OntologyService
from .parsing import PARSER_VERSION, extract, parse_structure, truncate
from .sources import SourceService

logger = logging.getLogger(__name__)


def embedding_text(node):
    """The text a node's retrieval embedding encodes; ingestion, enrichment and re-embedding must agree."""
    if node.kind == "Assertion":
        return node.proposition
    if node.kind == "Concept":
        return node.label + " " + node.description
    return getattr(node, "context_header", node.label) + "\n" + node.routing_summary + "\n" + (node.text or node.summary)


class IngestionEngine:
    def __init__(self, settings, repository, models):
        self.settings, self.repository, self.models = settings, repository, models
        self.sources = SourceService(settings)
        self.ontology = OntologyService(repository, models, settings.taxonomy_match_threshold, settings.primary_ontology,
                                        settings.embedding_model, settings.taxonomy_provisional_threshold,
                                        model=settings.semantic_model)
        # Per-document locks avoid duplicate inference for identical content while unrelated
        # documents (e.g. several external sources found by one query) ingest concurrently.
        self.locks = weakref.WeakValueDictionary()
        self.background = set()

    async def ingest(self, request, query_id=None):
        source, data = await self.sources.identify(request)
        scope = request.scope
        if scope != "shared":  # a private copy: identical content in another scope never merges with it
            source.id, source.scope = stable_id("source", f"{scope}:{source.id}"), scope
        document_id = stable_id("document", ("" if scope == "shared" else scope + ":") + f"{PARSER_VERSION}:{source.mime_type}:{source.content_hash}")
        lock = self.locks.setdefault(document_id, asyncio.Lock())
        async with lock:
            existing = await self.repository.get(document_id)
            if existing and existing.metadata.get("bibliography"):
                bibliography = existing.metadata["bibliography"]
                source.label = request.title or bibliography["title"]
                source.author = request.author or bibliography.get("author")
                source.published_at = source.published_at or bibliography.get("published_at")
                source.metadata = {**source.metadata, "bibliography": bibliography, "metadata_method": "reused_structured_metadata"}
                await self.repository.put([source], [Edge(source=source.id, target=document_id, relation=Relation.PROVIDES)])
                return {"source_id": source.id, "document_id": document_id, "duplicate": True}
            text, embedded = await asyncio.to_thread(extract, data, source.mime_type, self.settings.max_source_bytes, True)
            metadata = await self.models.structured("source_metadata", {
                "title": request.title, "filename": source.filename, "url": source.uri,
                "author": source.author, "published_at": source.published_at,
                "mime_type": source.mime_type, "embedded_metadata": embedded,
                "text": truncate(text, min(8000, self.settings.model_input_token_budget // 2)),
            }, SourceMetadata, query_id)
            source.label = request.title or metadata.title
            source.author = request.author or metadata.author
            source.published_at = source.published_at or metadata.published_at
            source.metadata = {**source.metadata, "bibliography": metadata.model_dump(), "metadata_method": "structured_model"}
            if existing:
                await self.repository.put([source], [Edge(source=source.id, target=document_id, relation=Relation.PROVIDES)])
                return {"source_id": source.id, "document_id": document_id, "duplicate": True}
            # Tokenizing and chunking large texts is CPU-bound; keep it off the event loop serving live streams.
            parsed = await asyncio.to_thread(parse_structure, text, source.label, document_id, self.settings)
            parsed.document.metadata.update({"formatted_content": text, "content_format": "text" if source.mime_type == "text/plain" else "markdown", "bibliography": metadata.model_dump()})
            for node in parsed.nodes:
                node.scope = scope
            understood = await self.understand(parsed, source, query_id)
            # The document hierarchy commits atomically first; it is retrievable without concept links.
            await self.repository.put([source, *parsed.nodes], [Edge(source=source.id, target=document_id, relation=Relation.PROVIDES), *parsed.edges])
            result = {"source_id": source.id, "document_id": document_id, "duplicate": False, "nodes_created": len(parsed.nodes)+1}
            if scope != "shared":
                # ponytail: private documents stay out of the shared concept graph (no label of theirs can leak into
                # it); they are found through the indexes, which is where nearly all evidence comes from anyway.
                return {**result, "concepts": "not linked (private scope)"}
            return {**result, **await self.link_concepts(document_id, parsed.nodes, understood, query_id)}

    async def ingest_light(self, request, query_id):
        """Query-time ingestion of a source found online: parse and embed, no other model call. Live queries that
        searched online made ~630 model calls each (median $0.20, 154 s), nearly all summarizing every section and
        classifying every concept of pages mostly never used again. A light document that answers a later query is
        structured then (structure_later). Returns (result, parsed nodes with their embeddings)."""
        source, data = await self.sources.identify(request)
        document_id = stable_id("document", f"{PARSER_VERSION}:{source.mime_type}:{source.content_hash}")
        async with self.locks.setdefault(document_id, asyncio.Lock()):
            result = {"source_id": source.id, "document_id": document_id, "duplicate": True}
            if await self.repository.get(document_id):
                await self.repository.put([source], [Edge(source=source.id, target=document_id, relation=Relation.PROVIDES)])
                return result, []
            text = await asyncio.to_thread(extract, data, source.mime_type, self.settings.max_source_bytes)
            parsed = await asyncio.to_thread(parse_structure, text, source.label, document_id, self.settings)
            parsed.document.metadata.update({"formatted_content": text, "content_format": "text" if source.mime_type == "text/plain" else "markdown",
                                             "light": True, "fetched_for": query_id})
            # ponytail: one embedding request per source; retrievers cap sources at 12k characters, which fits.
            vectors = await self.models.embed_batch([embedding_text(n) for n in parsed.nodes], query_id)
            for node, vector in zip(parsed.nodes, vectors):
                node.embedding = vector
            await self.repository.put([source, *parsed.nodes], [Edge(source=source.id, target=document_id, relation=Relation.PROVIDES), *parsed.edges])
            return {**result, "duplicate": False, "nodes_created": len(parsed.nodes) + 1, "light": True}, parsed.nodes

    def structure_later(self, document_ids, query_id):
        """Structures, in the background, the light documents among these that a query other than the one that
        fetched them has used: a source that proves useful again earns its summary and concepts."""
        async def run():
            for document_id in sorted(document_ids):
                await self.structure(document_id, query_id)
        if document_ids:
            task = asyncio.create_task(run())
            self.background.add(task)
            task.add_done_callback(self.finished)

    async def structure(self, document_id, query_id=None):
        """Document-level understanding and its concepts for a light document: one summary call and a few
        classifications, instead of one per section and one per concept of every section. The bibliographic call
        adds what freshness needs (publication date, author) to the sources of a page worth keeping."""
        async with self.locks.setdefault(document_id, asyncio.Lock()):
            document = await self.repository.get(document_id)
            if not document or not document.metadata.get("light") or document.metadata.get("fetched_for") == query_id:
                return None
            text = document.metadata.get("formatted_content", "")
            sources = [node_from(s) for s in (await self.repository.provenance(document_id))["sources"]]
            understanding, metadata = await asyncio.gather(
                self.models.structured("understanding", {
                    "title": document.label, "context": document.label, "summary_input": False,
                    "text": truncate(text, min(10000, self.settings.model_input_token_budget // 2)),
                    "content_format": document.metadata.get("content_format")}, Understanding),
                self.models.structured("source_metadata", {
                    "title": document.label, "url": next((s.uri for s in sources if s.uri), None),
                    "text": truncate(text, min(8000, self.settings.model_input_token_budget // 2))}, SourceMetadata))
            document.summary, document.routing_summary = understanding.summary, understanding.routing_summary
            document.document_type, document.language = understanding.document_type, understanding.language
            document.metadata.update(temporal_scope=understanding.temporal_scope, light=False, bibliography=metadata.model_dump())
            document.embedding = await self.models.embed(embedding_text(document))
            for source in sources:
                source.author = source.author or metadata.author
                source.published_at = source.published_at or metadata.published_at
                source.metadata = {**source.metadata, "bibliography": metadata.model_dump(), "metadata_method": "structured_model"}
            await self.repository.update_nodes([document, *sources])
            return await self.link_concepts(document_id, [document], {document.id: understanding})

    async def link_concepts(self, document_id, nodes, understood, query_id=None):
        """Resolve each distinct concept once per document (not once per chunk), then attach nodes with ABOUT.
        Skipped concepts are kept for review in the document's classification_review record."""
        specs = {}
        for understanding in understood.values():
            for spec in understanding.concepts:
                specs.setdefault(spec.label.casefold(), spec)
        pending = {}
        resolved, edges, skipped = await self.ontology.link(list(specs.values()), pending, query_id)
        for node in nodes:
            about = {c.id for spec in understood[node.id].concepts for c in resolved.get(spec.label.casefold(), [])}
            edges.extend(Edge(source=node.id, target=i, relation=Relation.ABOUT) for i in sorted(about))
        await self.repository.put(list(pending.values()), edges)
        reasons = Counter(s["reason"] for s in skipped)
        # Provisional concepts this document links (new or reused) await confirmation beside the skips.
        provisional = [{"label": n.label, "concept_id": n.id, "confidence": n.metadata.get("classification_confidence"),
                        "parents": sorted(pending[e.source].label for e in edges if e.relation == Relation.BROADER_THAN
                                          and e.target == n.id and e.source in pending)}
                       for n in pending.values() if n.metadata.get("classification_status") == "provisional"]
        if skipped or provisional:
            await self.repository.record("classification_review", document_id,
                                         {"document_id": document_id, "skipped": skipped, "provisional": provisional})
        if skipped:
            logger.warning("Document %s: %d of %d concepts not classified (%s)", document_id, len(skipped), len(specs),
                           ", ".join(f"{reason} {count}" for reason, count in reasons.most_common()))
        return {"concepts_linked": len(pending), "concepts_extracted": len(specs), "concepts_provisional": len(provisional),
                "unclassified_concepts": [s["label"] for s in skipped], "classification_skips": dict(reasons)}

    def finished(self, task):
        self.background.discard(task)
        if not task.cancelled() and task.exception():
            logger.warning("Background structuring failed: %s", task.exception())

    async def understand(self, parsed, source, query_id):
        """Bottom-up summaries (children before parents) retain coverage of late sections in long documents;
        nodes of equal height are independent and run concurrently."""
        children, height = defaultdict(list), {}
        for edge in parsed.edges:
            children[edge.source].append(edge.target)
        for node in reversed(parsed.nodes):
            height[node.id] = 1 + max((height[c] for c in children[node.id]), default=-1)
        results, gate = {}, asyncio.Semaphore(CONCURRENCY)

        async def one(node):
            async with gate:
                body = "\n".join(results[i].summary for i in children[node.id]) or parsed.texts[node.id]
                understanding = await self.models.structured("understanding", {
                    "title": node.label, "context": getattr(node, "context_header", source.label),
                    "text": truncate(body, min(10000, self.settings.model_input_token_budget // 2)),
                    "summary_input": bool(children[node.id]), "mime_type": source.mime_type,
                }, Understanding, query_id)
                node.summary, node.routing_summary = understanding.summary, understanding.routing_summary
                node.metadata["temporal_scope"] = understanding.temporal_scope
                node.metadata["parser_version"] = PARSER_VERSION
                node.embedding = await self.models.embed(embedding_text(node), query_id)
                if node.kind == "Document":
                    node.document_type, node.language = understanding.document_type, understanding.language
                results[node.id] = understanding

        for level in sorted(set(height.values())):
            await asyncio.gather(*(one(n) for n in parsed.nodes if height[n.id] == level))
        return results
