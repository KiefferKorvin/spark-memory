import asyncio

from .llm import ProviderError
from .models import Concept, ConceptSpec, Edge, ExternalConcept, Relation, Resolution, TaxonomyResolution, stable_id
from .prompts import VERSION

# ponytail: fixed fan-out for classification calls; make it a setting if provider rate limits bite.
CONCURRENCY = 6


class Skip(ValueError):
    """An unsound classification. `record` keeps why, what was proposed and how many candidates were shown.

    Reasons: low_confidence, unknown_reuse_id, unanchored_reuse, no_parent, unknown_parent_id, no_anchor,
    provider_error, invalid_concept."""
    def __init__(self, reason, spec, decision=None, candidates=None, record=None):
        super().__init__(f"{reason}: {spec.label}")
        shown = candidates or {}

        def named(i):
            return {"id": i, "label": getattr(shown.get(i), "label", None)}
        self.record = record or {"label": spec.label, "reason": reason, "confidence": decision.confidence if decision else None,
                                 "reuse": named(decision.reuse_id) if decision and decision.reuse_id else None,
                                 "parents": [named(i) for i in decision.parent_ids] if decision else [], "candidates": len(shown)}


def skip_record(spec, exc):
    if isinstance(exc, Skip):
        return {**exc.record, "label": spec.label}  # a failed broader lookup is reported under its concept
    return Skip("provider_error" if isinstance(exc, ProviderError) else "invalid_concept", spec).record


class ConceptResolver:
    def __init__(self, repository, models):
        self.repository, self.models = repository, models

    async def resolve(self, spec: ConceptSpec, pending: dict, query_id=None):
        for label in [spec.label, *spec.aliases]:
            for concept in pending.values():
                if label.casefold() in {concept.label.casefold(), *(a.casefold() for a in concept.aliases)}:
                    concept.aliases = sorted(set(concept.aliases + spec.aliases + ([spec.label] if spec.label != concept.label else [])))
                    return concept
            existing = await self.repository.exact_concept(label)
            if existing:
                existing.aliases = sorted(set(existing.aliases + spec.aliases + ([spec.label] if spec.label != existing.label else [])))
                pending[existing.id] = existing
                return existing
        vector = await self.models.embed(spec.label + " " + spec.description, query_id)
        candidates = await self.repository.candidates(spec.label + " " + spec.description, vector, 8, kind="Concept")
        plausible = [(n, score) for n, score in candidates if score >= 0.2]
        if plausible:
            records = []
            for node, score in plausible:
                neighbors, _ = await self.repository.neighbors(node.id, 12)
                records.append({"id": node.id, "label": node.label, "description": node.description,
                                "aliases": node.aliases, "neighbors": [n.label for n in neighbors], "score": score})
            decision = await self.models.structured("resolution", {"candidate": spec.model_dump(), "existing": records}, Resolution, query_id)
            if decision.reuse_id:
                matches = [n for n, _ in plausible if n.id == decision.reuse_id]
                if not matches:
                    raise Skip("unknown_reuse_id", spec)
                matches[0].aliases = sorted(set(matches[0].aliases + spec.aliases + [spec.label]))
                pending[matches[0].id] = matches[0]
                return matches[0]
        node = Concept(id=stable_id("concept", spec.label.strip().casefold()), label=spec.label.strip(),
                       preferred_label=spec.label.strip(), aliases=spec.aliases,
                       description=spec.description, summary=spec.description,
                       routing_summary=f"Explore for {spec.label}. {spec.description}", embedding=vector)
        pending[node.id] = node
        return node


class OntologyService:
    """Links extracted concepts. Returns ({label_casefold: [concept, *anchors]}, edges, skip records).

    A concept that cannot be classified soundly is skipped rather than rejecting its whole document:
    the text stays retrievable through fulltext/vector indexes and no orphan concept is created.
    Each skip is a Skip.record, so a review sees the reason instead of a bare label.

    Confidence in [provisional, threshold) with valid proposed parents creates a LOCAL concept marked
    classification_status="provisional" instead of skipping it. Model-decided skips are cached per label
    (MemoryRecord "classification_skip") so later documents do not pay for the same verdict again.
    """
    def __init__(self, repository, models, threshold=0.75, authority="UNESCO", embedding_model="", provisional=0.5, cache=True,
                 model=""):
        self.repository = repository
        self.authority, self.embedding_model, self.model = authority, embedding_model, model
        self.models, self.threshold, self.provisional = models, threshold, min(provisional, threshold)
        self.resolver = ConceptResolver(repository, models)
        self.semantic, self.cache = False, cache

    async def link(self, specs, pending, query_id=None):
        if await self.repository.read_record("taxonomy", self.authority):
            return await self.link_taxonomy(specs, pending, query_id)
        resolved, edges, skipped = {}, [], []
        for spec in specs:
            try:
                concept = await self.resolver.resolve(spec, pending, query_id)
                for label in spec.broader:
                    parent = await self.resolver.resolve(ConceptSpec(label=label), pending, query_id)
                    if parent.id != concept.id:
                        edges.append(Edge(source=parent.id, target=concept.id, relation=Relation.BROADER_THAN))
                for label in spec.related:
                    related = await self.resolver.resolve(ConceptSpec(label=label), pending, query_id)
                    if related.id != concept.id:
                        edges.append(Edge(source=concept.id, target=related.id, relation=Relation.RELATED_TO))
            except (ValueError, ProviderError) as exc:
                skipped.append(skip_record(spec, exc))
                continue
            resolved[spec.label.casefold()] = [concept]
        return resolved, edges, skipped

    async def link_taxonomy(self, specs, pending, query_id=None):
        # Semantic candidates only once the thesaurus is embedded with this model (unesco --embed).
        embedded = await self.repository.read_record("taxonomy_embedding", self.authority)
        self.semantic = bool(embedded) and embedded["model"] == self.embedding_model
        if self.cache:
            await self.skip_cache_scope()
        resolved, edges, skipped, failed = {}, [], [], set()
        remaining, gate = list(specs), asyncio.Semaphore(CONCURRENCY)

        async def classify(spec):
            async with gate:
                return await self.classify(spec, pending, query_id)

        while remaining:
            # Parents extracted from the same text are classified first; a cyclic remainder runs as one layer.
            layer = [s for s in remaining if not any(p.casefold() == o.label.casefold() for p in s.broader for o in remaining if o is not s)] or remaining
            remaining = [s for s in remaining if not any(s is l for l in layer)]
            # A same-document broader concept that failed cannot be a parent: classify its children without it.
            layer = [s.model_copy(update={"broader": [b for b in s.broader if b.casefold() not in failed]})
                     if failed & {b.casefold() for b in s.broader} else s for s in layer]
            for spec, result in zip(layer, await asyncio.gather(*map(classify, layer), return_exceptions=True)):
                if isinstance(result, (ValueError, ProviderError)):
                    skipped.append(skip_record(spec, result))
                    failed.add(spec.label.casefold())
                    # Only a model's verdict is remembered; provider errors are transient.
                    if self.cache and isinstance(result, Skip) and result.record["confidence"] is not None and not result.record.get("cached"):
                        await self.repository.record("classification_skip", self.skip_key(spec), result.record)
                    continue
                if isinstance(result, BaseException):
                    raise result
                concept, anchors, extra, new_edges = result
                edges.extend(new_edges)
                for node in [concept, *anchors, *extra]:
                    pending[node.id] = node
                resolved[spec.label.casefold()] = list({n.id: n for n in [concept, *anchors]}.values())
                if concept.origin == "LOCAL":
                    concept.aliases = sorted(set(concept.aliases + spec.aliases + ([spec.label] if spec.label != concept.label else [])))
                for label in spec.related:
                    related = next((n for n in pending.values() if n.label.casefold() == label.casefold()), None) or await self.repository.exact_concept(label, self.authority)
                    if related and related.id != concept.id and (concept.origin == "LOCAL" or related.origin == "LOCAL"):
                        pending[related.id] = related
                        edges.append(Edge(source=concept.id, target=related.id, relation=Relation.RELATED_TO))
        return resolved, edges, skipped

    async def classify(self, spec, pending, query_id=None):
        """Returns (concept, taxonomy anchors, other nodes to persist, new edges); raises ValueError when unsound."""
        exact = await self.repository.exact_concept(spec.label, self.authority)
        if exact is None:
            exact = next((n for n in pending.values() if spec.label.casefold() in {n.label.casefold(), *(a.casefold() for a in n.aliases)}), None)
        anchors = await self.anchors_for(exact, pending) if exact else []
        if exact and anchors:
            return exact, anchors, [], []
        if self.cache and (cached := await self.repository.read_record("classification_skip", self.skip_key(spec))):
            raise Skip(cached["reason"], spec, record={**cached, "label": spec.label, "cached": True})
        # The label and broader labels name the concept; free-text descriptions mostly add generic words.
        query = " ".join([spec.label, *spec.broader])
        vector = await self.models.embed(". ".join([spec.label, ", ".join(spec.broader), spec.description])[:1000], query_id) if self.semantic else None
        candidates = {n.id: n for n in await self.repository.taxonomy_candidates(query, 20, self.authority, vector)}
        local_matches = [n for n, score in await self.repository.candidates(query, [], 12, kind="Concept") if n.origin == "LOCAL" and score > 0]
        from .graph import rank
        local_matches += sorted(pending.values(), key=lambda n: -rank(n, query, []))[:12]
        for local in local_matches:
            if local.origin == "LOCAL" and await self.anchors_for(local, pending):
                candidates[local.id] = local
        for label in spec.broader:
            parent = next((n for n in pending.values() if n.label.casefold() == label.casefold()), None) or await self.repository.exact_concept(label, self.authority)
            if parent and await self.anchors_for(parent, pending):
                candidates[parent.id] = parent
        if exact:
            candidates[exact.id] = exact
        decision = await self.models.structured("taxonomy_resolution", {
            "authority": self.authority, "candidate": spec.model_dump(), "existing": [dict(id=n.id, label=n.label, aliases=n.aliases, description=n.description, origin=n.origin) for n in candidates.values()]}, TaxonomyResolution, query_id)
        provisional = decision.confidence < self.threshold
        # Reuse means equivalence, so it needs full confidence; a new child of sound parents may be provisional.
        if decision.confidence < self.provisional or provisional and decision.reuse_id:
            raise Skip("low_confidence", spec, decision, candidates)
        if decision.reuse_id:
            if decision.reuse_id not in candidates:
                raise Skip("unknown_reuse_id", spec, decision, candidates)
            concept = candidates[decision.reuse_id]
            anchors = await self.anchors_for(concept, pending)
            if not anchors:
                raise Skip("unanchored_reuse", spec, decision, candidates)
            return concept, anchors, [], []
        if not decision.parent_ids:
            raise Skip("no_parent", spec, decision, candidates)
        if any(i not in candidates for i in decision.parent_ids):
            raise Skip("unknown_parent_id", spec, decision, candidates)
        for parent_id in decision.parent_ids:
            anchors += await self.anchors_for(candidates[parent_id], pending)
        if not anchors:
            raise Skip("no_anchor", spec, decision, candidates)
        concept = exact or Concept(id=stable_id("concept", ("" if self.authority == "UNESCO" else self.authority + ":") + spec.label.strip().casefold()), label=spec.label.strip(), preferred_label=spec.label.strip(), aliases=spec.aliases, description=spec.description, summary=spec.description, routing_summary=spec.description)
        concept.metadata["taxonomy_anchor_ids"] = sorted({n.id for n in anchors})
        if provisional:
            concept.metadata.update(classification_status="provisional", classification_confidence=decision.confidence)
        parents = sorted(set(decision.parent_ids))
        return concept, anchors, [candidates[i] for i in parents], [
            Edge(source=i, target=concept.id, relation=Relation.BROADER_THAN) for i in parents]

    def skip_key(self, spec):
        return f"{self.authority}:{spec.label.strip().casefold()}"

    async def skip_cache_scope(self):
        """Cached verdicts hold for one thesaurus snapshot, classifier model, threshold pair and prompt version;
        a change clears them."""
        manifest = await self.repository.read_record("taxonomy", self.authority) or {}
        scope = {"fingerprint": manifest.get("fingerprint"), "threshold": self.threshold, "model": self.model,
                 "provisional": self.provisional, "prompt_version": VERSION}
        if await self.repository.read_record("classification_skip_scope", self.authority) != scope:
            await self.clear_skip_cache()
            await self.repository.record("classification_skip_scope", self.authority, scope)

    async def clear_skip_cache(self):
        await self.repository.delete_records("classification_skip", self.authority + ":")

    async def anchors_for(self, node, pending):
        if node.origin == self.authority:
            return [node]
        if node.origin != 'LOCAL':
            return []
        anchors = {n.id: n for n in await self.ancestors(node.id)}
        for ident in node.metadata.get('taxonomy_anchor_ids', node.metadata.get('unesco_anchor_ids', [])):
            anchor = pending.get(ident) or await self.repository.get(ident)
            if anchor and anchor.origin == self.authority:
                anchors[anchor.id] = anchor
        return list(anchors.values())

    async def ancestors(self, node_id):
        return await self.repository.taxonomy_ancestors(node_id, self.authority)

    async def align(self, concept_id, system, external_id, label, uri=None, exact=False):
        """Application-supplied ontology alignment; never guesses external identifiers."""
        node = ExternalConcept(id=stable_id("external-concept", f"{system}:{external_id}"),
                               label=label, system=system, external_id=external_id, uri=uri)
        await self.repository.put([node], [Edge(source=concept_id, target=node.id,
            relation=Relation.EXACT_MATCH if exact else Relation.CLOSE_MATCH)])
        return node
