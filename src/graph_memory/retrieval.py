import asyncio
import hashlib
import heapq
import itertools
import re
from datetime import datetime, timedelta, timezone

from .evidence import ContextBuilder, EvidenceCollector, SufficiencyEvaluator, model_view
from .external import source_request
from .graph import rank, terms, visible
from .llm import ProviderError, cited
from .models import (Answer, Branch, Coverage, CoverageItem, Decision, Decomposition,
                     Evidence, NavigationDecision, Need, Relevance, now)
from .parsing import truncate
from .procedures import RetrievalKnowHow, host, score


def retrievable(node):
    return bool(node and node.text) and (node.kind != "Document" or node.retrieval_leaf)


def preview(node, score=0, excerpt_tokens=0):
    # Routing summaries (not content summaries) drive navigation; keep the policy payload bounded.
    view = {"node_id": node.id, "label": node.label, "kind": node.kind,
            "origin": getattr(node, "origin", None), "uri": getattr(node, "uri", None),
            "routing_summary": node.routing_summary,
            "aliases": getattr(node, "aliases", []), "score": score,
            "retrievable": retrievable(node),
            "temporal_scope": node.metadata.get("temporal_scope")}
    if excerpt_tokens and node.text and (node.kind in ("Chunk", "Assertion") or node.kind == "Document" and node.retrieval_leaf):
        # A short leaf's opening text shows whether it answers the need far better than a routing summary.
        # The character pre-cut bounds tokenizer work; 8 characters per token is a safe upper bound.
        view["excerpt"] = truncate(node.text[:excerpt_tokens * 8], excerpt_tokens)
    return view


def answer_words(answer, context):
    """Words of an answer without its inline evidence citations, which the length limit does not count."""
    known = {e["id"] for e in context}
    bare = re.sub(r"\s*\[([^\[\]\n]+)\]", lambda m: "" if all(p.strip() in known for p in m.group(1).split(",")) else m.group(0), answer)
    return len(bare.split())


def real_citations(answer, ids):
    """[E1] or [E1, E2] citations of the model view become real evidence IDs, one bracket each, as clients expect."""
    def swap(match):
        parts = [p.strip() for p in match.group(1).split(",")]
        return "".join(f"[{ids[p]}]" for p in parts) if all(p in ids for p in parts) else match.group(0)
    return re.sub(r"\[([^\[\]\n]+)\]", swap, answer)


def search_key(description, scope="shared"):
    """Search memory key: the scope and the need's content words, so reordered or re-punctuated repeats match."""
    return scope + "|" + " ".join(sorted(terms(description)))[:500]


def episode_options(request):
    return f"{request.mode}|{request.synthesize}|{request.answer_max_words}|{request.allow_external}|{request.original_sources_only}"


def episode_key(request):
    """One episode per distinct question and options in a scope: a repeat updates it (count, latest result)."""
    digest = hashlib.sha256((episode_options(request) + "|" + " ".join(sorted(terms(request.query)))).encode()).hexdigest()[:32]
    return (request.scope or "shared") + "|" + digest


def url_key(url):
    """Forgotten-URL record key: a digest, so deleting one URL's record (a key prefix) never matches another URL."""
    return hashlib.sha256(url.encode()).hexdigest()


def trace_preview(node, score=0, excerpt_tokens=0):
    """Compact preview for persisted/streamed events; the UI fetches full node details on selection."""
    view = {**preview(node, score, excerpt_tokens), "routing_summary": node.routing_summary[:300],
            "aliases": getattr(node, "aliases", [])[:5]}
    if "excerpt" in view:
        view["excerpt"] = view["excerpt"][:300]
    return view


class ExplorationSupervisor:
    def __init__(self, settings, repository, models, trace, original_sources_only=False, question=None, question_vector=None,
                 breadth=False, scopes=("shared",), user_context=()):
        self.settings, self.repository, self.models, self.trace = settings, repository, models, trace
        # breadth (dossiers): expansions also run for needs that already have evidence, to reach every source.
        self.breadth = breadth
        # The user's own question: decomposition rewrites wording, so search and relevance also see the original.
        self.question, self.question_vector, self.vectors = question, question_vector or [], {}
        self.scopes = scopes
        self.collector = EvidenceCollector(repository, models, original_sources_only, user_context)
        self.frontier, self.deferred, self.visited, self.evidence, self.found = [], [], set(), {}, set()
        self.branches, self.counter, self.explored = [], itertools.count(), 0

    async def spawn(self, branch, defer=False):
        if branch.node_id:
            key = (branch.information_need_id, branch.node_id)
            if key in self.visited:
                await self.trace.emit("BRANCH_MERGED", branch, reason="already_scheduled_for_need")
                return
            self.visited.add(key)
        if defer:
            self.deferred.append(branch)
        else:
            heapq.heappush(self.frontier, (-branch.priority, next(self.counter), branch))
        self.branches.append(branch)
        await self.trace.emit("BRANCH_SPAWNED", branch, path=branch.path, priority=branch.priority, **({"deferred": True} if defer else {}))

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
            self.found.add(need.id)
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
        if current and not visible(current, self.scopes):  # a start node forgotten or outside the query's scopes
            branch.status = "DEAD_END"
            await self.trace.emit("BRANCH_COMPLETED", branch, status=branch.status, reason="not_visible")
            return
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
        if branch.action == "SELECT" and (branch.evidence_found or retrievable(current)):
            # A selected leaf is retrieved, not explored: over 150 benchmark queries, navigating the neighborhoods of
            # rejected leaves and expanded concepts took 77 navigation calls (~3 s each) and found 1 evidence item.
            branch.status = "COMPLETE" if branch.evidence_found else "DEAD_END"
            await self.trace.emit("BRANCH_COMPLETED", branch, status=branch.status,
                                  **({} if branch.evidence_found else {"reason": "no_evidence"}))
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
            decision = NavigationDecision(decisions=[Decision(node_id=n.id, action="SELECT" if retrievable(n) else "EXPAND")
                                                       for n, _ in candidates[:breadth]],
                                          current_node_action="CONTINUE")
        actions = {d.node_id: d.action for d in decision.decisions}
        if decision.current_node_action == "DEAD_END":
            actions = {}
        if not branch.node_id and all(a == "PRUNE" for a in actions.values()) and self.settings.min_root_children:
            # Never explore nothing while candidates exist: the index ranking still carries signal.
            floor = candidates[:self.settings.min_root_children]
            actions = {n.id: "SELECT" if retrievable(n) else "EXPAND" for n, _ in floor}
            await self.trace.emit("NAVIGATION_FLOOR", branch, nodes=[trace_preview(n, s) for n, s in floor])
        taken = 0
        for node, score in candidates:
            action = actions.get(node.id, "PRUNE")
            if action == "PRUNE" or taken >= breadth:
                await self.trace.emit("NODE_PRUNED", branch, node_id=node.id,
                                      reason="policy" if action == "PRUNE" else "branch_budget")
                continue
            taken += 1
            if not retrievable(node):
                action = "EXPAND"  # nothing to retrieve from a container or concept
            child = Branch(information_need_id=need.id, node_id=node.id, parent_branch_id=branch.id,
                           path=[*branch.path, node.id], depth=branch.depth+1, priority=score, action=action)
            if current:
                await self.trace.emit("EDGE_TRAVERSED", child, source=current.id, target=node.id)
            await self.spawn(child, defer=action == "EXPAND")
        branch.status = "COMPLETE" if taken or branch.evidence_found else "DEAD_END"
        if not taken:
            await self.trace.emit("BACKTRACK", branch, reason="policy_dead_end")
        await self.trace.emit("BRANCH_COMPLETED", branch, status=branch.status)

    async def candidates(self, need, vector, parent):
        """Index candidates for the need and for the original question, merged before the limit."""
        limit = self.settings.candidate_limit
        found = await self.repository.candidates(need.description, vector, limit, parent=parent, scopes=self.scopes)
        if not self.question or self.question.strip().casefold() == need.description.strip().casefold():
            return found
        best = {n.id: (n, score) for n, score in found}
        for node, score in await self.repository.candidates(self.question, self.question_vector, limit, parent=parent, scopes=self.scopes):
            if node.id not in best or score > best[node.id][1]:
                best[node.id] = (node, score)
        return sorted(best.values(), key=lambda pair: (-pair[1], pair[0].id))[:limit]

    async def explore(self, needs, starts=None):
        """Explore from the graph root, or from given (node ID, action) starts per need (e.g. newly ingested sources).

        Each call gets its own node budget, so sources fetched after a first pass are always examined."""
        self.ceiling = self.explored + self.settings.max_total_nodes_explored
        # Needs with evidence from this pass; an external pass runs for needs the first one left incomplete.
        self.found = set()
        await self.embed(needs)
        lookup = {n.id: n for n in needs}
        for need in needs:
            for node_id, action in (starts[need.id] if starts else [(None, "EXPAND")]):
                await self.spawn(Branch(information_need_id=need.id, node_id=node_id, action=action,
                                        path=[node_id] if node_id else [], depth=1 if node_id else 0, priority=1 if node_id else 0))
        await self.drain(lookup)
        # Expansion waits until the selected leaves are checked and runs only for needs they left without evidence
        # (for every need in a dossier): on the benchmark, 620 of 621 evidence items came from leaves selected among the
        # index candidates. It goes one hop only: released again and again, expansions walked the concept graph up to
        # the node budget (513 navigation calls and 245 s for one dossier).
        ready = [b for b in self.deferred if self.breadth or b.information_need_id not in self.found]
        if ready and self.explored < self.ceiling:
            self.deferred = [b for b in self.deferred if not any(b is r for r in ready)]
            for branch in ready:
                heapq.heappush(self.frontier, (-branch.priority, next(self.counter), branch))
            await self.drain(lookup)
        for branch in self.deferred:
            branch.status = "PRUNED"
            reason = ("need_has_evidence" if branch.information_need_id in self.found and not self.breadth
                      else "global_node_budget" if self.explored >= self.ceiling else "expansion_depth")
            await self.trace.emit("BRANCH_COMPLETED", branch, status="PRUNED", reason=reason)
        self.deferred = []
        while self.frontier:
            _, _, branch = heapq.heappop(self.frontier)
            branch.status = "PRUNED"
            await self.trace.emit("BRANCH_COMPLETED", branch, status="PRUNED", reason="global_node_budget")
        return list(self.evidence.values())

    async def drain(self, lookup):
        while self.frontier and self.explored < self.ceiling:
            batch = []
            while self.frontier and len(batch) < self.settings.max_parallel_branches and self.explored < self.ceiling:
                _, _, branch = heapq.heappop(self.frontier)
                if branch.node_id:
                    self.explored += 1  # Reserve budget before launching workers.
                batch.append(branch)
            await asyncio.gather(*(self.work(b, lookup[b.information_need_id], self.vectors[b.information_need_id]) for b in batch))

    async def embed(self, needs):
        """Query vectors in as few requests as possible: one embedding request takes 0.3 to 14 s. A lone need restates
        the question (see the decomposition prompt), so it shares the question's vector."""
        todo = [n for n in needs if n.id not in self.vectors]
        if len(todo) == len(needs) == 1 and self.question_vector:
            self.vectors[todo[0].id] = self.question_vector
        elif todo:
            ask = bool(self.question and not self.question_vector)
            try:
                vectors = await self.models.embed_batch([n.description for n in todo] + [self.question] * ask, self.trace.query_id)
            except ProviderError:
                await self.trace.emit("MODEL_FAILURE", operation="embedding", fallback="question_vector" if self.question_vector else "lexical_search")
                vectors = [self.question_vector] * (len(todo) + ask)
            self.vectors.update(zip((n.id for n in todo), vectors))
            if ask:
                self.question_vector = vectors[-1]


class RetrievalEngine:
    def __init__(self, settings, repository, models, ingestion, external):
        self.settings, self.repository, self.models = settings, repository, models
        self.ingestion, self.external = ingestion, external
        self.evaluator = SufficiencyEvaluator(models)
        self.knowhow = RetrievalKnowHow(repository, settings.host_min_trials)
        # Coverage and synthesis receive this whole context, so it must fit the model input budget.
        self.context = ContextBuilder(min(settings.context_token_budget, settings.model_input_token_budget - 2000))

    async def coverage(self, needs, evidence, trace, by_source=False):
        context = self.context.build(needs, evidence, by_source)
        view, ids = model_view(context)
        try:
            result = await self.evaluator.evaluate(needs, view, trace.query_id)
            for item in result.coverage:
                item.evidence_ids = [ids[i] for i in item.evidence_ids]
        except (ProviderError, ValueError):
            await trace.emit("MODEL_FAILURE", operation="coverage", fallback="conservative_insufficient")
            result = Coverage(coverage=[CoverageItem(information_need_id=n.id, status="MISSING", evidence_ids=[],
                                                     missing="Coverage could not be verified") for n in needs], overall_status="INSUFFICIENT")
        await trace.emit("SUFFICIENCY_CHECKED", coverage=result.model_dump())
        return result, context

    async def search_external(self, need, limit, discovered, trace, question=None, vector=None, scope="shared", user_context=()):
        """Search online for one need; relevant sources are ingested concurrently. Returns (node ID, action) starts."""
        await trace.emit("EXTERNAL_SEARCH_STARTED", need=need.model_dump())
        try:
            sources = await self.external.search(need.description, limit)
        except Exception as exc:
            await trace.emit("EXTERNAL_SEARCH_FAILED", need_id=need.id, reason=type(exc).__name__)
            return []
        kept = []
        for source in sources:
            if source.url in discovered or await self.repository.read_record("forgotten", url_key(source.url)):
                continue
            if reason := await self.knowhow.skip_reason(source.url):  # procedural memory: hosts that never help
                await trace.emit("EXTERNAL_SOURCE_REJECTED", url=source.url, need_id=need.id, reason=reason)
                continue
            kept.append(source)
        scores = {s.url: score(await self.knowhow.stats("host", host(s.url))) for s in kept}
        sources = sorted(kept, key=lambda s: -scores[s.url])[:limit]
        discovered.update(s.url for s in sources)
        if not sources:
            await trace.emit("EXTERNAL_SEARCH_EMPTY", need_id=need.id)

        async def one(source):
            await trace.emit("EXTERNAL_SOURCE_FOUND", title=source.title, url=source.url, need_id=need.id,
                             retriever=source.metadata.get("retriever"))

            async def tally(**outcome):
                await self.knowhow.count("host", host(source.url), found=1, **outcome)
                await self.knowhow.count("retriever", source.metadata.get("retriever"), found=1, **outcome)
            try:
                relevant = await self.models.structured("relevance", {"question": question, "information_need": need.model_dump(),
                    "text": truncate(source.text, 6000), "label": source.title,
                    **({"user_context": list(user_context)} if user_context else {})}, Relevance, trace.query_id)
                if not relevant.relevant:
                    await tally(rejected=1)
                    await trace.emit("EXTERNAL_SOURCE_REJECTED", url=source.url, need_id=need.id, reason="not relevant")
                    return None
                await trace.emit("SOURCE_INGESTION_STARTED", url=source.url, need_id=need.id)
                result, nodes = await self.ingestion.ingest_light(source_request(source), trace.query_id)
                await tally(accepted=1)
                await trace.emit("SOURCE_INGESTED", url=source.url, title=source.title, need_id=need.id, **result)
                return result, nodes
            except (ValueError, ProviderError) as exc:
                await tally(failed=1)
                await trace.emit("EXTERNAL_SOURCE_REJECTED", url=source.url, need_id=need.id, reason=str(exc)[:200])
                return None
        ingested = [r for r in await asyncio.gather(*map(one, sources)) if r]
        if ingested:  # remembered only once something relevant was found; a fruitless search may be retried
            await self.repository.record("search", search_key(need.description, scope), {
                "need": need.description, "scope": scope, "at": now(), "query_id": trace.query_id, "urls": [s.url for s in sources],
                "documents": [result["document_id"] for result, _ in ingested], "reused": 0, "used": False}, vector=vector or None)
        # A source already in memory is explored from its document. A new one has no summaries to navigate by, but its
        # passages were just embedded, so the ones closest to the need go straight to the relevance check.
        leaves = sorted((n for _, nodes in ingested for n in nodes if retrievable(n)), key=lambda n: -rank(n, need.description, vector or []))
        return ([(result["document_id"], "EXPAND") for result, _ in ingested if result["duplicate"]]
                + [(n.id, "SELECT") for n in leaves[:self.settings.max_root_children]])

    async def recalled_search(self, need, vector, scope="shared"):
        """A search for the same need within SEARCH_MEMORY_DAYS: the same content words, or an embedding at least
        SEARCH_MEMORY_SIMILARITY close. Its relevant sources are already in memory, so the web is not searched again.
        Returns (search record, similarity) or None."""
        if not self.settings.search_memory_days:
            return None
        since = (datetime.now(timezone.utc) - timedelta(days=self.settings.search_memory_days)).isoformat()
        exact = await self.repository.read_record("search", search_key(need.description, scope))
        close = await self.repository.similar_records("search", vector, 10) if vector else []
        return next(((record, score) for record, score in [*([(exact, 1.0)] if exact else []), *close]
                     if score >= self.settings.search_memory_similarity and record["at"] >= since
                     and record.get("scope", "shared") == scope), None)

    async def shorten(self, payload, answer, context, limit, trace, operation="synthesis"):
        """Makes the length limit binding: one rewrite of an answer over 110% of it (longer answers make more claims
        that can go beyond the evidence). The draft stays when the rewrite
        fails, cites nothing or is not shorter, so a long answer is never replaced by the raw-excerpt fallback."""
        before = answer_words(answer.answer, context)
        try:
            short = await self.models.structured(operation, {**payload, "draft_answer": answer.answer}, Answer, trace.query_id)
            short.evidence_ids = cited(short.answer, context)
        except (ValueError, ProviderError):
            short = None
        after = answer_words(short.answer, context) if short and short.evidence_ids else before
        await trace.emit("ANSWER_SHORTENED", limit=limit, words_before=before, words_after=min(after, before))
        return short if after < before else answer

    async def answer(self, request, context, trace, user_context=()):
        """Synthesis reads the model view of the context; its short citations are rewritten to real evidence IDs."""
        await trace.emit("ANSWER_GENERATION_STARTED")
        dossier = request.mode == "dossier"
        if not context:
            return Answer(answer="Memory holds nothing on this topic." if dossier else "Memory is insufficient to answer this question.",
                          evidence_ids=[])
        view, ids = model_view(context)
        operation = "dossier" if dossier else "synthesis"
        try:
            limit = request.answer_max_words or self.settings.answer_max_words
            payload = {"query": request.query, "evidence": view, **({"answer_max_words": limit} if limit else {}),
                       **({"user_context": list(user_context)} if user_context else {})}
            answer = await self.models.structured(operation, payload, Answer, trace.query_id)
            if not cited(answer.answer, view):
                raise ValueError("Answer must cite supplied evidence")
            if limit and answer_words(answer.answer, view) > limit * 1.1:
                answer = await self.shorten(payload, answer, view, limit, trace, operation)
            text = real_citations(answer.answer, ids)
            # Cited IDs are whatever known evidence the answer actually cites inline.
            return Answer(answer=text, evidence_ids=cited(text, context))
        except (ValueError, ProviderError):
            # Retrieval work is never discarded because synthesis failed: fall back to the original excerpts.
            await trace.emit("MODEL_FAILURE", operation=operation, fallback="original_excerpts")
            return Answer(answer="Synthesis unavailable. Evidence excerpts:\n" + "\n".join(f"{e['text']} [{e['id']}]" for e in context),
                          evidence_ids=[e["id"] for e in context])

    async def reuse(self, request, trace, vector=None):
        """The result of a recent episode of the same question: exact (same content words, before any model call) or,
        given the question's vector, a paraphrase at least EPISODE_SIMILARITY close. Only within EPISODE_REUSE_HOURS,
        with the same scope and options, and when nothing the scope reads (ingestion, forgetting, user facts) has
        changed since. Live PAKT queries repeat verbatim within hours; each repeat paid for a full exploration."""
        if not (request.reuse and self.settings.episode_reuse_hours):
            return None
        scope, options = request.scope or "shared", episode_options(request)
        since = (datetime.now(timezone.utc) - timedelta(hours=self.settings.episode_reuse_hours)).isoformat()
        changed = max([(await self.repository.read_record("state", s + "|changed") or {}).get("at", "") for s in {"shared", scope}])
        found = ([(await self.repository.read_record("episode", episode_key(request)), 1.0)] if vector is None
                 else await self.repository.similar_records("episode", vector, 10) if vector else [])
        for episode, similarity in found:
            if not episode or similarity < self.settings.episode_similarity or episode["scope"] != scope \
                    or episode["options"] != options or episode["at"] < since or episode["at"] <= changed:
                continue
            previous = await self.repository.read_record("query", episode["query_id"])
            # A result that found nothing is not worth repeating: asked again, the question gets a fresh search.
            if not previous or previous.get("status") != "completed" or not previous.get("context"):
                continue
            await self.repository.record("episode", episode_key(request) if vector is None else episode["key"],
                                         {**episode, "count": episode["count"] + 1, "asked_at": now()})
            origin = {"query_id": episode["query_id"], "at": episode["at"], "question": episode["question"], "similarity": round(similarity, 3)}
            await trace.emit("EPISODE_REUSED", **origin)
            return {**previous, "query_id": trace.query_id, "reused_from": origin}
        return None

    async def remember(self, request, result, vector, searched):
        """Records the query as an episode: what was asked and when, what it found and which sources it used."""
        key = episode_key(request)
        previous = await self.repository.read_record("episode", key) or {}
        sources = {}
        for item in result["context"]:
            for source in item["provenance"].get("sources", [])[:1]:
                sources.setdefault(source.get("uri") or source.get("label"), source.get("label"))
        at = now()
        await self.repository.record("episode", key, {
            "key": key, "scope": request.scope or "shared", "question": request.query, "mode": request.mode,
            "options": episode_options(request), "at": at, "asked_at": at, "first_at": previous.get("first_at", at),
            "count": previous.get("count", 0) + 1, "query_id": result["query_id"],
            "query_ids": [*previous.get("query_ids", []), result["query_id"]][-50:],
            "needs": [n["description"] for n in result["information_needs"]], "answer": result["answer"][:1000],
            "sources": [{"uri": uri, "title": title} for uri, title in list(sources.items())[:8]],
            "searched_online": bool(searched)}, vector=vector or None)

    async def learn(self, context, searched, query_id):
        """What the final context teaches the memory: whether this query's online searches paid off (search metrics),
        which light documents proved useful again (structured in the background), which hosts and retrievers gave
        evidence that was used (retrieval know-how)."""
        used = {d for e in context for d in e["provenance"].get("documents", [])}
        for key in searched:
            if (record := await self.repository.read_record("search", key)) and used & set(record["documents"]):
                await self.repository.record("search", key, {**record, "used": True})
        self.ingestion.structure_later(used, query_id)
        firsts = [e["provenance"]["sources"][0] for e in context if e["provenance"].get("sources")]
        for name in {host(s.get("uri")) for s in firsts if s.get("uri")}:
            await self.knowhow.count("host", name, used=1)
        for name in {s.get("metadata", {}).get("retriever") for s in firsts} - {None}:
            await self.knowhow.count("retriever", name, used=1)

    async def query(self, request, trace):
        await trace.emit("QUERY_STARTED", query=request.query)
        if reused := await self.reuse(request, trace):
            return reused
        # The question's embedding does not depend on the decomposition, so the two requests run at once.
        decomposition, question_vector = await asyncio.gather(
            self.models.structured("decomposition", {"query": request.query}, Decomposition, trace.query_id),
            self.models.embed(request.query, trace.query_id), return_exceptions=True)
        # Neither may fail the query: 6 of 9 failed live queries died on the decomposition (a DNS failure, then a 402).
        if isinstance(decomposition, (ProviderError, ValueError)):
            await trace.emit("MODEL_FAILURE", operation="decomposition", fallback="question_as_need")
            decomposition = Decomposition(information_needs=[Need(id="N1", description=request.query[:1500])])
        if isinstance(question_vector, (ProviderError, ValueError)):
            await trace.emit("MODEL_FAILURE", operation="embedding", fallback="lexical_search")
            question_vector = []
        for outcome in (decomposition, question_vector):
            if isinstance(outcome, BaseException):
                raise outcome
        if question_vector and (reused := await self.reuse(request, trace, question_vector)):
            return reused
        needs = decomposition.information_needs
        await trace.emit("QUERY_DECOMPOSED", information_needs=[n.model_dump() for n in needs])
        dossier, scope = request.mode == "dossier", request.scope or "shared"
        # User memory: a private scope's active facts judge relevance and shape the answer; they never enter the search
        # text (recipe queries had exclusions glued in: "courgette feta sans Crevettes Caramel Viande rouge à éviter").
        facts = sorted(await self.repository.records("fact", scope + "|"), key=lambda f: f["created_at"]) if scope != "shared" else []
        user_context = [f"{f['kind']}: {f['text']}" for f in facts if f["status"] == "active"]
        if user_context:
            await trace.emit("USER_CONTEXT_APPLIED", facts=len(user_context))
        supervisor = ExplorationSupervisor(self.settings, self.repository, self.models, trace, request.original_sources_only,
                                           question=request.query, question_vector=question_vector, breadth=dossier,
                                           scopes=tuple(dict.fromkeys(("shared", scope))), user_context=user_context)
        evidence = await supervisor.explore(needs)
        # Coverage only decides whether to search online. Without that option it would be a paid no-op: a quarter of a
        # benchmark query's cost and ~2 s. The result then reports coverage as null (not assessed).
        skipped = None
        if not request.allow_external:
            skipped = "disabled for this query"
        elif not self.external:
            skipped = "no external retriever configured"
        elif not (self.settings.max_external_rounds and self.settings.max_external_sources):
            skipped = "no external budget"
        coverage, context = (None, self.context.build(needs, evidence, dossier)) if skipped else await self.coverage(needs, evidence, trace, dossier)
        if skipped:
            await trace.emit("EXTERNAL_SEARCH_SKIPPED", reason=skipped)
        discovered, searched = set(), []
        for _ in range(0 if skipped else self.settings.max_external_rounds):
            quota = self.settings.max_external_sources - len(discovered)
            if coverage.overall_status == "SUFFICIENT" or quota <= 0:
                break
            missing_ids = {c.information_need_id for c in coverage.coverage if c.status != "COVERED"}
            missing = [n for n in needs if n.id in missing_ids][:quota]
            # Search memory: 81% of completed live queries were judged incomplete and searched online, repeats included.
            for need in list(missing):
                if recalled := await self.recalled_search(need, supervisor.vectors.get(need.id), scope):
                    record, similarity = recalled
                    missing.remove(need)
                    await self.repository.record("search", search_key(record["need"], scope), {**record, "reused": record.get("reused", 0) + 1})
                    await trace.emit("EXTERNAL_SEARCH_SKIPPED", need_id=need.id, reason="searched recently", searched_at=record["at"],
                                     searched_need=record["need"], similarity=round(similarity, 3))
            if not missing:
                break
            found = await asyncio.gather(*(self.search_external(n, max(1, quota // len(missing)), discovered, trace, request.query,
                                                                supervisor.vectors[n.id], scope, user_context) for n in missing))
            starts = {n.id: ids for n, ids in zip(missing, found) if ids}
            searched += [search_key(n.description, scope) for n in missing if n.id in starts]
            if not starts:
                break
            # Examine the new sources directly rather than re-navigating the whole graph from its root.
            evidence = await supervisor.explore([n for n in missing if n.id in starts], starts)
            coverage, context = await self.coverage(needs, evidence, trace, dossier)
        if coverage and coverage.overall_status == "SUFFICIENT":
            await trace.emit("SUFFICIENCY_REACHED")
        await trace.emit("FINAL_CONTEXT_BUILT", evidence_ids=[e["id"] for e in context])
        await self.learn(context, searched, trace.query_id)
        # Clients that use only the evidence (PAKT) skip the most expensive call.
        answer = await self.answer(request, context, trace, user_context) if request.synthesize else Answer(answer="", evidence_ids=[])
        result = {"query_id": trace.query_id, "status": "completed", "answer": answer.answer,
                  "evidence_ids": answer.evidence_ids, "information_needs": [n.model_dump() for n in needs],
                  "coverage": coverage.model_dump() if coverage else None, "evidence": [e.model_dump() for e in evidence],
                  "context": context, "branches": [b.model_dump() for b in supervisor.branches],
                  "nodes_explored": supervisor.explored}
        await self.remember(request, result, question_vector, searched)
        return result
