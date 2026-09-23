import asyncio
import heapq
import itertools

from .evidence import ContextBuilder, EvidenceCollector, SufficiencyEvaluator
from .external import source_request
from .llm import ProviderError, cited
from .models import (Answer, Branch, Coverage, CoverageItem, Decision, Decomposition,
                     Evidence, NavigationDecision, Relevance)
from .parsing import truncate


def preview(node, score=0, excerpt_tokens=0):
    # Routing summaries (not content summaries) drive navigation; keep the policy payload bounded.
    view = {"node_id": node.id, "label": node.label, "kind": node.kind,
            "origin": getattr(node, "origin", None), "uri": getattr(node, "uri", None),
            "routing_summary": node.routing_summary,
            "aliases": getattr(node, "aliases", []), "score": score,
            "retrievable": bool(node.text) and (node.kind != "Document" or node.retrieval_leaf),
            "temporal_scope": node.metadata.get("temporal_scope")}
    if excerpt_tokens and node.text and (node.kind in ("Chunk", "Assertion") or node.kind == "Document" and node.retrieval_leaf):
        # A short leaf's opening text shows whether it answers the need far better than a routing summary.
        # The character pre-cut bounds tokenizer work; 8 characters per token is a safe upper bound.
        view["excerpt"] = truncate(node.text[:excerpt_tokens * 8], excerpt_tokens)
    return view


def trace_preview(node, score=0, excerpt_tokens=0):
    """Compact preview for persisted/streamed events; the UI fetches full node details on selection."""
    view = {**preview(node, score, excerpt_tokens), "routing_summary": node.routing_summary[:300],
            "aliases": getattr(node, "aliases", [])[:5]}
    if "excerpt" in view:
        view["excerpt"] = view["excerpt"][:300]
    return view


class ExplorationSupervisor:
    def __init__(self, settings, repository, models, trace, original_sources_only=False, question=None):
        self.settings, self.repository, self.models, self.trace = settings, repository, models, trace
        # The user's own question: decomposition rewrites wording, so search and relevance also see the original.
        self.question, self.question_vector = question, []
        self.collector = EvidenceCollector(repository, models, original_sources_only)
        self.frontier, self.visited, self.evidence = [], set(), {}
        self.branches, self.counter, self.explored = [], itertools.count(), 0

    async def spawn(self, branch):
        if branch.node_id:
            key = (branch.information_need_id, branch.node_id)
            if key in self.visited:
                await self.trace.emit("BRANCH_MERGED", branch, reason="already_scheduled_for_need")
                return
            self.visited.add(key)
        heapq.heappush(self.frontier, (-branch.priority, next(self.counter), branch))
        self.branches.append(branch)
        await self.trace.emit("BRANCH_SPAWNED", branch, path=branch.path, priority=branch.priority)

    async def collect(self, node, need, branch):
        if not node:
            return
        try:
            item = await self.collector.collect(node, need, self.trace.query_id, self.question)
        except ProviderError:
            await self.trace.emit("MODEL_FAILURE", branch, operation="relevance")
            return
        if item:
            if item.id in self.evidence:
                current = self.evidence[item.id]
                current.information_need_ids = sorted(set(current.information_need_ids + item.information_need_ids))
            else:
                self.evidence[item.id] = item
            branch.evidence_found.append(item.id)
            await self.repository.hit(node.id, success=True)
            await self.trace.emit("EVIDENCE_FOUND", branch, evidence=self.evidence[item.id].model_dump())
            # Preserve material contradictions even when they lie outside the chosen route.
            for counterpart_id in ([] if self.collector.original_sources_only else item.provenance.get("contradictions", [])):
                counterpart = await self.repository.get(counterpart_id)
                if counterpart and counterpart.text:
                    opposite = Evidence(id="contradiction:" + counterpart.id, information_need_ids=[need.id],
                        source_node_id=counterpart.id, source_type="assertion", text=counterpart.text,
                        relevance_score=item.relevance_score, confidence=counterpart.confidence,
                        provenance=await self.repository.provenance(counterpart.id))
                    self.evidence[opposite.id] = opposite
                    await self.trace.emit("EVIDENCE_FOUND", branch, node_id=counterpart.id, evidence=opposite.model_dump())

    async def work(self, branch, need, vector):
        current = await self.repository.get(branch.node_id) if branch.node_id else None
        if current:
            branch.visited_nodes.append(current.id)
            await self.repository.hit(current.id)
            await self.trace.emit("NODE_SELECTED" if branch.action == "SELECT" else "NODE_EXPANDED", branch, node=trace_preview(current))
            await self.collect(current, need, branch)
        else:
            await self.trace.emit("ROOT_ENTERED", branch)
        if branch.depth >= self.settings.max_depth:
            branch.status = "COMPLETE" if branch.evidence_found else "DEAD_END"
            await self.trace.emit("BRANCH_COMPLETED", branch, status=branch.status, reason="depth_budget")
            return
        if branch.action == "SELECT" and branch.evidence_found:
            branch.status = "COMPLETE"
            await self.trace.emit("BRANCH_COMPLETED", branch, status=branch.status)
            return
        # Root candidates come from the fulltext/vector indexes; imported taxonomy concepts qualify
        # only once content is attached to them, so empty thesaurus entries cannot absorb the budget.
        candidates = await self.candidates(need, vector, branch.node_id)
        candidates = [(n, score) for n, score in candidates if (need.id, n.id) not in self.visited]
        excerpt = self.settings.navigation_excerpt_tokens
        await self.trace.emit("CANDIDATES_GENERATED", branch, nodes=[trace_preview(n, s, excerpt) for n, s in candidates])
        if not candidates:
            branch.status = "COMPLETE" if branch.evidence_found else "DEAD_END"
            await self.trace.emit("BACKTRACK", branch, reason="no_unvisited_candidates")
            await self.trace.emit("BRANCH_COMPLETED", branch, status=branch.status)
            return
        # A root decision chooses a need's entry points among all index candidates; deeper ones stay narrow.
        breadth = self.settings.max_children_per_decision if branch.node_id else self.settings.max_root_children
        payload = {"information_need": need.model_dump(), "current_path": branch.path,
                   "current_node": preview(current, excerpt_tokens=excerpt) if current else None,
                   "candidate_children": [preview(n, s, excerpt) for n, s in candidates],
                   "evidence_already_found": [{"id": e.id, "source_node_id": e.source_node_id,
                                               "information_need_ids": e.information_need_ids} for e in self.evidence.values()],
                   "remaining_budget": {"nodes": self.ceiling - self.explored,
                                        "depth": self.settings.max_depth - branch.depth}}
        try:
            decision = await self.models.navigate(payload, self.trace.query_id)
            candidate_ids = {n.id for n, _ in candidates}
            if any(d.node_id not in candidate_ids for d in decision.decisions) or len({d.node_id for d in decision.decisions}) != len(decision.decisions):
                raise ValueError("Navigation selected invalid or duplicate IDs")
        except (ProviderError, ValueError):
            await self.trace.emit("MODEL_FAILURE", branch, operation="navigation", fallback="bounded_candidate_order")
            decision = NavigationDecision(decisions=[Decision(node_id=n.id, action="SELECT" if preview(n)["retrievable"] else "EXPAND")
                                                       for n, _ in candidates[:breadth]],
                                          current_node_action="CONTINUE")
        actions = {d.node_id: d.action for d in decision.decisions}
        if decision.current_node_action == "DEAD_END":
            actions = {}
        if not branch.node_id and all(a == "PRUNE" for a in actions.values()) and self.settings.min_root_children:
            # Never explore nothing while candidates exist: the index ranking still carries signal.
            floor = candidates[:self.settings.min_root_children]
            actions = {n.id: "SELECT" if preview(n)["retrievable"] else "EXPAND" for n, _ in floor}
            await self.trace.emit("NAVIGATION_FLOOR", branch, nodes=[trace_preview(n, s) for n, s in floor])
        taken = 0
        for node, score in candidates:
            action = actions.get(node.id, "PRUNE")
            if action == "PRUNE" or taken >= breadth:
                await self.trace.emit("NODE_PRUNED", branch, node_id=node.id,
                                      reason="policy" if action == "PRUNE" else "branch_budget")
                continue
            taken += 1
            child = Branch(information_need_id=need.id, node_id=node.id, parent_branch_id=branch.id,
                           path=[*branch.path, node.id], depth=branch.depth+1, priority=score, action=action)
            if current:
                await self.trace.emit("EDGE_TRAVERSED", child, source=current.id, target=node.id)
            await self.spawn(child)
        branch.status = "COMPLETE" if taken or branch.evidence_found else "DEAD_END"
        if not taken:
            await self.trace.emit("BACKTRACK", branch, reason="policy_dead_end")
        await self.trace.emit("BRANCH_COMPLETED", branch, status=branch.status)

    async def candidates(self, need, vector, parent):
        """Index candidates for the need and for the original question, merged before the limit."""
        limit = self.settings.candidate_limit
        found = await self.repository.candidates(need.description, vector, limit, parent=parent)
        if not self.question or self.question.strip().casefold() == need.description.strip().casefold():
            return found
        best = {n.id: (n, score) for n, score in found}
        for node, score in await self.repository.candidates(self.question, self.question_vector, limit, parent=parent):
            if node.id not in best or score > best[node.id][1]:
                best[node.id] = (node, score)
        return sorted(best.values(), key=lambda pair: (-pair[1], pair[0].id))[:limit]

    async def explore(self, needs, starts=None):
        """Explore from the graph root, or from given start nodes per need (e.g. newly ingested sources).

        Each call gets its own node budget, so sources fetched after a first pass are always examined."""
        self.ceiling = self.explored + self.settings.max_total_nodes_explored
        vectors = {need.id: await self.models.embed(need.description, self.trace.query_id) for need in needs}
        if self.question and not self.question_vector:
            self.question_vector = await self.models.embed(self.question, self.trace.query_id)
        lookup = {n.id: n for n in needs}
        for need in needs:
            for node_id in (starts[need.id] if starts else [None]):
                await self.spawn(Branch(information_need_id=need.id, node_id=node_id,
                                        path=[node_id] if node_id else [], depth=1 if node_id else 0, priority=1 if node_id else 0))
        while self.frontier and self.explored < self.ceiling:
            batch = []
            while self.frontier and len(batch) < self.settings.max_parallel_branches and self.explored < self.ceiling:
                _, _, branch = heapq.heappop(self.frontier)
                if branch.node_id:
                    self.explored += 1  # Reserve budget before launching workers.
                batch.append(branch)
            await asyncio.gather(*(self.work(b, lookup[b.information_need_id], vectors[b.information_need_id]) for b in batch))
        while self.frontier:
            _, _, branch = heapq.heappop(self.frontier)
            branch.status = "PRUNED"
            await self.trace.emit("BRANCH_COMPLETED", branch, status="PRUNED", reason="global_node_budget")
        return list(self.evidence.values())


class RetrievalEngine:
    def __init__(self, settings, repository, models, ingestion, external):
        self.settings, self.repository, self.models = settings, repository, models
        self.ingestion, self.external = ingestion, external
        self.evaluator = SufficiencyEvaluator(models)
        # Coverage and synthesis receive this whole context, so it must fit the model input budget.
        self.context = ContextBuilder(min(settings.context_token_budget, settings.model_input_token_budget - 2000))

    async def coverage(self, needs, evidence, trace):
        context = self.context.build(needs, evidence)
        try:
            result = await self.evaluator.evaluate(needs, [Evidence.model_validate(e) for e in context], trace.query_id)
        except (ProviderError, ValueError):
            await trace.emit("MODEL_FAILURE", operation="coverage", fallback="conservative_insufficient")
            result = Coverage(coverage=[CoverageItem(information_need_id=n.id, status="MISSING", evidence_ids=[],
                                                     missing="Coverage could not be verified") for n in needs], overall_status="INSUFFICIENT")
        await trace.emit("SUFFICIENCY_CHECKED", coverage=result.model_dump())
        return result, context

    async def search_external(self, need, limit, discovered, trace, question=None):
        """Search online for one need; relevant sources are ingested concurrently. Returns new document IDs."""
        await trace.emit("EXTERNAL_SEARCH_STARTED", need=need.model_dump())
        try:
            sources = await self.external.search(need.description, limit)
        except Exception as exc:
            await trace.emit("EXTERNAL_SEARCH_FAILED", need_id=need.id, reason=type(exc).__name__)
            return []
        sources = [s for s in sources if s.url not in discovered][:limit]
        discovered.update(s.url for s in sources)
        if not sources:
            await trace.emit("EXTERNAL_SEARCH_EMPTY", need_id=need.id)

        async def one(source):
            await trace.emit("EXTERNAL_SOURCE_FOUND", title=source.title, url=source.url, need_id=need.id,
                             retriever=source.metadata.get("retriever"))
            try:
                relevant = await self.models.structured("relevance", {"question": question, "information_need": need.model_dump(),
                    "text": truncate(source.text, 6000), "label": source.title}, Relevance, trace.query_id)
                if not relevant.relevant:
                    await trace.emit("EXTERNAL_SOURCE_REJECTED", url=source.url, need_id=need.id, reason="not relevant")
                    return None
                await trace.emit("SOURCE_INGESTION_STARTED", url=source.url, need_id=need.id)
                result = await self.ingestion.ingest(source_request(source), trace.query_id, defer_concepts=True)
                await trace.emit("SOURCE_INGESTED", url=source.url, title=source.title, need_id=need.id, **result)
                return result["document_id"]
            except (ValueError, ProviderError) as exc:
                await trace.emit("EXTERNAL_SOURCE_REJECTED", url=source.url, need_id=need.id, reason=str(exc)[:200])
                return None
        return [d for d in await asyncio.gather(*map(one, sources)) if d]

    async def query(self, request, trace):
        await trace.emit("QUERY_STARTED", query=request.query)
        decomposition = await self.models.structured("decomposition", {"query": request.query}, Decomposition, trace.query_id)
        needs = decomposition.information_needs
        await trace.emit("QUERY_DECOMPOSED", information_needs=[n.model_dump() for n in needs])
        supervisor = ExplorationSupervisor(self.settings, self.repository, self.models, trace, request.original_sources_only,
                                           question=request.query)
        evidence = await supervisor.explore(needs)
        coverage, context = await self.coverage(needs, evidence, trace)
        if coverage.overall_status != "SUFFICIENT" and not (request.allow_external and self.external):
            await trace.emit("EXTERNAL_SEARCH_SKIPPED", reason="disabled for this query" if self.external else "no external retriever configured")
        discovered = set()
        for _ in range(self.settings.max_external_rounds):
            quota = self.settings.max_external_sources - len(discovered)
            if coverage.overall_status == "SUFFICIENT" or not request.allow_external or not self.external or quota <= 0:
                break
            missing_ids = {c.information_need_id for c in coverage.coverage if c.status != "COVERED"}
            missing = [n for n in needs if n.id in missing_ids][:quota]
            found = await asyncio.gather(*(self.search_external(n, max(1, quota // len(missing)), discovered, trace, request.query)
                                           for n in missing))
            starts = {n.id: ids for n, ids in zip(missing, found) if ids}
            if not starts:
                break
            # Examine the new sources directly rather than re-navigating the whole graph from its root.
            evidence = await supervisor.explore([n for n in missing if n.id in starts], starts)
            coverage, context = await self.coverage(needs, evidence, trace)
        if coverage.overall_status == "SUFFICIENT":
            await trace.emit("SUFFICIENCY_REACHED")
        await trace.emit("FINAL_CONTEXT_BUILT", evidence_ids=[e["id"] for e in context])
        await trace.emit("ANSWER_GENERATION_STARTED")
        try:
            limit = request.answer_max_words or self.settings.answer_max_words
            answer = await self.models.structured("synthesis", {"query": request.query,
                "coverage": coverage.model_dump(), "evidence": context, **({"answer_max_words": limit} if limit else {})},
                Answer, trace.query_id)
            # Cited IDs are whatever known evidence the answer actually cites inline.
            answer.evidence_ids = cited(answer.answer, context)
            if context and not answer.evidence_ids:
                raise ValueError("Answer must cite supplied evidence")
        except (ValueError, ProviderError):
            # Retrieval work is never discarded because synthesis failed: fall back to the original excerpts.
            await trace.emit("MODEL_FAILURE", operation="synthesis", fallback="original_excerpts")
            answer = Answer(answer="Synthesis unavailable. Evidence excerpts:\n" + "\n".join(
                f"{e['text']} [{e['id']}]" for e in context) if context else "Memory is insufficient to answer this question.",
                evidence_ids=[e["id"] for e in context])
        result = {"query_id": trace.query_id, "status": "completed", "answer": answer.answer,
                  "evidence_ids": answer.evidence_ids, "information_needs": [n.model_dump() for n in needs],
                  "coverage": coverage.model_dump(), "evidence": [e.model_dump() for e in evidence],
                  "context": context, "branches": [b.model_dump() for b in supervisor.branches],
                  "nodes_explored": supervisor.explored}
        return result
