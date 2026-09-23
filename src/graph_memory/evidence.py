import json

from .models import Assertion, Coverage, CoverageItem, Edge, Enrichment, Evidence, Relation, Relevance, stable_id
from .parsing import token_count, truncate


class EvidenceCollector:
    def __init__(self, repository, models, original_sources_only=False):
        self.repository, self.models = repository, models
        self.original_sources_only = original_sources_only

    async def collect(self, node, need, query_id, question=None):
        if node.kind not in ("Chunk", "Assertion", "Document") or not node.text:
            return None
        if node.kind == "Document" and not node.retrieval_leaf:
            return None
        if self.original_sources_only and node.kind == "Assertion":
            return None
        provenance = await self.repository.provenance(node.id)
        if self.original_sources_only:
            provenance["sources"] = [s for s in provenance["sources"]
                                     if s.get("source_type") != "generated" and not s.get("metadata", {}).get("generated")
                                     and str(s.get("uri") or "").startswith(("https://", "http://"))]
        if not provenance["sources"]:
            return None
        assessment = await self.models.structured("relevance", {
            "question": question, "information_need": need.model_dump(), "text": node.text, "label": node.label,
        }, Relevance, query_id)
        if not assessment.relevant:
            return None
        return Evidence(id=stable_id("evidence", node.id), information_need_ids=[need.id],
            source_node_id=node.id, source_type=node.kind.lower(), text=node.text,
            relevance_score=assessment.confidence, confidence=getattr(node, "confidence", 1.0), provenance=provenance)


class SufficiencyEvaluator:
    def __init__(self, models):
        self.models = models

    async def evaluate(self, needs, evidence, query_id):
        if not evidence:
            return Coverage(coverage=[CoverageItem(information_need_id=n.id, status="MISSING", evidence_ids=[],
                                                   missing=n.description) for n in needs], overall_status="INSUFFICIENT")
        result = await self.models.structured("coverage", {
            "information_needs": [n.model_dump() for n in needs],
            "evidence": [e.model_dump() for e in evidence],
        }, Coverage, query_id)
        # Invalid parts of a verdict are discarded conservatively instead of discarding every need's verdict:
        # invented evidence IDs support nothing, and unassessed needs stay MISSING.
        # Evidence found while exploring one need may legitimately answer another.
        known, verdicts = {e.id for e in evidence}, {}
        for item in result.coverage:
            item.evidence_ids = [i for i in item.evidence_ids if i in known]
            if item.status != "MISSING" and not item.evidence_ids:
                item.status = "MISSING"
                item.missing = item.missing or "No supporting evidence was cited"
            verdicts.setdefault(item.information_need_id, item)
        coverage = [verdicts.get(n.id) or CoverageItem(information_need_id=n.id, status="MISSING", evidence_ids=[],
                                                        missing="Coverage was not assessed") for n in needs]
        return Coverage(coverage=coverage, overall_status="SUFFICIENT" if all(i.status == "COVERED" for i in coverage) else "INSUFFICIENT")


class ContextBuilder:
    def __init__(self, token_budget):
        self.token_budget = token_budget

    def build(self, needs, evidence):
        # Round-robin across needs so one prolific branch cannot consume all context; within a need the most
        # relevant evidence comes first, so truncation drops the weakest and synthesis reads the best first.
        groups = [sorted((e for e in evidence if n.id in e.information_need_ids), key=lambda e: -e.relevance_score) for n in needs]
        ordered, seen = [], set()
        for index in range(max((len(g) for g in groups), default=0)):
            for group in groups:
                if index < len(group) and group[index].id not in seen:
                    item = group[index]
                    ordered.append(item)
                    seen.add(item.id)
                    # Keep material contradictory counterparts adjacent where they were retrieved.
                    for other in evidence:
                        if other.source_node_id in item.provenance.get("contradictions", []) and other.id not in seen:
                            ordered.append(other)
                            seen.add(other.id)
        selected = []
        for item in ordered:
            payload = item.model_dump()
            if token_count(json.dumps([*selected, payload], ensure_ascii=False)) <= self.token_budget:
                selected.append(payload)
                continue
            # Include bounded original text, never silently exceed the metadata-inclusive budget.
            payload["text"] = ""
            overhead = token_count(json.dumps([*selected, payload], ensure_ascii=False))
            remaining = self.token_budget - overhead - 16
            if remaining > 40:
                payload["text"] = truncate(item.text, remaining)
                payload["provenance"] = {**payload["provenance"], "context_truncated": True}
                if token_count(json.dumps([*selected, payload], ensure_ascii=False)) <= self.token_budget:
                    selected.append(payload)
        return selected


class MemoryEnrichmentService:
    def __init__(self, settings, repository, models):
        self.settings, self.repository, self.models = settings, repository, models

    async def enrich(self, node_id, query_id=None, force=False):
        if not self.settings.enrichment_enabled or self.settings.enrichment_policy == "disabled":
            return []
        if not force and self.settings.enrichment_policy == "on_demand":
            return []
        node = await self.repository.get(node_id)
        if not node or not node.text or node.kind not in ("Document", "Chunk"):
            raise ValueError("Enrichment requires an original retrieval leaf")
        if await self.repository.read_record("enriched", node_id):
            return []
        probe = node.summary or node.text[:500]  # stored payloads carry no vector, so embed the probe text itself
        candidates = await self.repository.candidates(probe, await self.models.embed(probe, query_id), 12, kind="Assertion")
        existing = [n for n, _ in candidates]
        output = await self.models.structured("enrichment", {
            "text": node.text, "existing_assertions": [n.model_dump() for n in existing],
        }, Enrichment, query_id)
        nodes, edges = [], []
        neighbors, _ = await self.repository.neighbors(node_id)
        for spec in output.assertions:
            if spec.supporting_quote not in node.text:
                raise ValueError("Assertion quote is absent from its source")
            if any(i not in {n.id for n in existing} for i in spec.contradicts_ids):
                raise ValueError("Unknown contradiction target")
            if spec.valid_from and spec.valid_to and spec.valid_from > spec.valid_to:
                raise ValueError("Assertion temporal range is inverted")
            assertion = Assertion(id=stable_id("assertion", f"{node_id}:{spec.proposition}"),
                label=spec.proposition[:150], text=spec.proposition, proposition=spec.proposition,
                assertion_type=spec.assertion_type, confidence=spec.confidence,
                valid_from=spec.valid_from.isoformat() if spec.valid_from else None,
                valid_to=spec.valid_to.isoformat() if spec.valid_to else None,
                embedding=await self.models.embed(spec.proposition, query_id),
                metadata={"supporting_quote": spec.supporting_quote})
            nodes.append(assertion)
            edges.append(Edge(source=assertion.id, target=node_id, relation=Relation.SUPPORTED_BY))
            edges.extend(Edge(source=assertion.id, target=c.id, relation=Relation.ABOUT) for c in neighbors if c.kind == "Concept")
            edges.extend(Edge(source=assertion.id, target=i, relation=Relation.CONTRADICTS) for i in spec.contradicts_ids)
        await self.repository.put(nodes, edges)
        await self.repository.record("enriched", node_id, {"assertion_ids": [n.id for n in nodes]})
        return nodes
