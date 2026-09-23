import asyncio
import base64
import io
import json
import socket

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from graph_memory.api import create_app
from graph_memory.config import Settings
from graph_memory.demo import DemoExternal, DemoModels, DOCUMENTS, QUESTIONS, seed
from graph_memory.evidence import ContextBuilder, SufficiencyEvaluator
from graph_memory.graph import InMemoryGraph
from graph_memory.llm import OpenRouter, ProviderError
from graph_memory.models import (Concept, Coverage, CoverageItem, Edge, Evidence, IngestRequest,
                                 NavigationDecision, Relation, Understanding)
from graph_memory.parsing import chunk_text, extract, parse_structure, token_count
from graph_memory.service import Memory
from graph_memory.sources import SafeFetcher


def settings(**kwargs):
    return Settings(memory_mode="demo", _env_file=None, **kwargs)


def make_memory(**kwargs):
    return Memory(settings(**kwargs), InMemoryGraph(), DemoModels(), DemoExternal())


def test_hierarchy_and_small_document_leaf():
    parsed = parse_structure("Brief source.", "note", "small", settings())
    assert parsed.document.retrieval_leaf and len(parsed.nodes) == 1
    text = "# A\nintro\n## B\nb\n1.2.3.4.5.6.7 Deep\ndeep text\n## C\nlast"
    parsed = parse_structure(text, "Book", "doc", settings())
    sections = [n for n in parsed.nodes if n.kind == "Section"]
    deep = next(n for n in sections if n.level == 7)
    assert deep.section_path == ["A", "B", "1.2.3.4.5.6.7 Deep"]
    assert next(n for n in sections if n.label == "C").section_path == ["A", "C"]
    assert any(n.context_header == "Book > A > B > 1.2.3.4.5.6.7 Deep" for n in parsed.nodes if n.kind == "Chunk")
    code = parse_structure("```python\n# not a heading\n1.2.3 not either\n```", "code", "code", settings())
    assert not any(n.kind == "Section" for n in code.nodes)


@pytest.mark.parametrize("text", ["One sentence. "*300, "你好世界🌍"*200, "X"*10000, "Paragraph one.\n\n"*150], ids=["sentences", "unicode", "long_word", "paragraphs"])
def test_chunk_boundaries_unicode_and_token_limit(text):
    chunks = chunk_text(text, 35, 50)
    assert "".join(chunks) == text
    assert all(token_count(c) <= 50 for c in chunks)


def test_html_json_email_and_docx_parsing():
    assert "bad()" not in extract(b"<h1>Title</h1><p>Body</p><script>bad()</script>", "text/html", 10000)
    assert json.loads(extract(b'{"x":1}', "application/json", 1000)) == {"x": 1}
    email = b"Subject: Test\nFrom: a@example.org\nContent-Type: text/plain; charset=utf-8\n\nA message"
    assert "A message" in extract(email, "message/rfc822", 10000)
    from docx import Document
    doc = Document(); doc.add_heading("Outer", 1); doc.add_heading("Inner", 2); doc.add_paragraph("Original text.")
    output = io.BytesIO(); doc.save(output)
    text = extract(output.getvalue(), "application/vnd.openxmlformats-officedocument.wordprocessingml.document", 100000)
    assert "# Outer" in text and "## Inner" in text and "Original text." in text


async def test_concept_reuse_duplicate_ingestion_multiple_provenance():
    memory = make_memory()
    first = await memory.ingest(IngestRequest(**DOCUMENTS[0]))
    count = len(memory.repository.nodes)
    again = await memory.ingest(IngestRequest(**DOCUMENTS[0]))
    assert again["duplicate"] and again["document_id"] == first["document_id"]
    assert len(memory.repository.nodes) == count
    await memory.ingest(IngestRequest(title="Second source", text=DOCUMENTS[0]["text"], mime_type="text/markdown"))
    assert len((await memory.repository.provenance(first["document_id"]))["sources"]) == 2
    await memory.ingest(IngestRequest(title="Piano", text="Piano and harmony work together."))
    assert len([n for n in memory.repository.nodes.values() if n.kind == "Concept" and n.label == "Piano"]) == 1
    rootless = await memory.repository.exact_concept("Rootless voicings")
    parents = [e for e in memory.repository.edges.values() if e.target == rootless.id and e.relation == Relation.BROADER_THAN]
    assert len(parents) == 2


async def test_alias_resolution_cycle_rollback_and_external_alignment():
    memory = make_memory()
    a = Concept(id="a", label="Keyboard", preferred_label="Keyboard", aliases=["Piano"])
    b = Concept(id="b", label="Instrument", preferred_label="Instrument")
    await memory.repository.put([a, b], [Edge(source="b", target="a", relation=Relation.BROADER_THAN)])
    assert (await memory.repository.exact_concept("piano")).id == "a"
    with pytest.raises(ValueError, match="cycle"):
        await memory.repository.put([Concept(id="c", label="new", preferred_label="new")], [Edge(source="a", target="b", relation=Relation.BROADER_THAN)])
    assert await memory.repository.get("c") is None
    await memory.ingestion.ontology.align("a", "demo", "123", "External keyboard", exact=True)
    assert any(e.relation == Relation.EXACT_MATCH for e in memory.repository.edges.values())


async def test_hybrid_candidate_generation_ranks_before_limit():
    memory = make_memory(candidate_limit=3)
    root = Concept(id="root", label="Root", preferred_label="Root")
    nodes = [Concept(id=f"n{i}", label="unrelated" if i < 100 else "rootless piano", preferred_label=str(i)) for i in range(101)]
    await memory.repository.put([root, *nodes], [Edge(source="root", target=n.id, relation=Relation.BROADER_THAN) for n in nodes])
    result = await memory.repository.candidates("rootless piano", await memory.models.embed("rootless piano"), 3, parent="root")
    assert len(result) == 3 and result[0][0].id == "n100"


async def test_end_to_end_parallel_loop_prevention_pruning_external_ingestion_and_trace():
    memory = make_memory()
    await memory.initialize()
    await seed(memory)
    result = await memory.query(QUESTIONS[3])
    assert result["status"] == "completed", result
    assert result["coverage"]["overall_status"] == "SUFFICIENT", result["coverage"]
    events = await memory.events(result["query_id"])
    types = {e["event_type"] for e in events}
    assert {"NODE_PRUNED", "BRANCH_SPAWNED", "EVIDENCE_FOUND", "EXTERNAL_SEARCH_STARTED", "SOURCE_INGESTED", "QUERY_COMPLETED"} <= types
    assert len({e["event_id"] for e in events}) == len(events)
    assert [e["sequence"] for e in events] == list(range(1, len(events)+1))
    explored = [(e["information_need_id"], e["node_id"]) for e in events if e["event_type"] in {"NODE_SELECTED", "NODE_EXPANDED"}]
    assert len(explored) == len(set(explored))
    assert all(len(e["metadata"]["nodes"]) <= memory.settings.candidate_limit for e in events if e["event_type"] == "CANDIDATES_GENERATED")
    assert any(s["metadata"].get("original_uri") for e in result["evidence"] for s in e["provenance"]["sources"])
    assert result["nodes_explored"] <= memory.settings.max_total_nodes_explored
    await memory.close()


async def test_budget_and_missing_coverage_without_external():
    from graph_memory.models import QueryRequest
    memory = make_memory(max_total_nodes_explored=2, max_depth=1)
    await seed(memory)
    result = await memory.query(QueryRequest(query="unknown orbital period", allow_external=False))
    assert result["nodes_explored"] <= 2
    assert result["coverage"]["overall_status"] == "INSUFFICIENT"
    assert not any(e["event_type"] == "EXTERNAL_SEARCH_STARTED" for e in await memory.events(result["query_id"]))


async def test_enrichment_preserves_support_and_is_idempotent():
    memory = make_memory()
    result = await memory.ingest(IngestRequest(title="Piano", text="Piano has keys."))
    nodes = await memory.enrichment.enrich(result["document_id"], force=True)
    assert len(nodes) == 1 and nodes[0].assertion_type == "claim"
    assert (await memory.repository.provenance(nodes[0].id))["sources"]
    assert await memory.enrichment.enrich(result["document_id"], force=True) == []


async def test_coverage_rejects_hallucinated_evidence_and_context_budget():
    from graph_memory.models import Need
    needs = [Need(id="N1", description="topic")]
    item = Evidence(id="E1", information_need_ids=["N1"], source_node_id="a", source_type="chunk", text="original "*2000,
                    relevance_score=.9, confidence=.8, provenance={"sources": []})
    context = ContextBuilder(300).build(needs, [item, item])
    assert token_count(json.dumps(context, ensure_ascii=False)) <= 300
    assert len(context) == 1 and context[0]["text"] in item.text
    class BadCoverage(DemoModels):
        async def structured(self, operation, payload, schema, query_id=None):
            return Coverage(coverage=[CoverageItem(information_need_id="N1", status="COVERED", evidence_ids=["invented"])], overall_status="SUFFICIENT")
    # An invented ID supports nothing, and a need the model skipped stays MISSING, without discarding valid verdicts.
    bad = await SufficiencyEvaluator(BadCoverage()).evaluate([*needs, Need(id="N2", description="other")], [item], "q")
    assert [(c.status, c.evidence_ids) for c in bad.coverage] == [("MISSING", []), ("MISSING", [])]
    assert bad.overall_status == "INSUFFICIENT"
    class CrossNeed(DemoModels):
        async def structured(self, operation, payload, schema, query_id=None):
            return Coverage(coverage=[CoverageItem(information_need_id=n, status="COVERED", evidence_ids=["E1"]) for n in ("N1", "N2")], overall_status="SUFFICIENT")
    # E1 was found for N1 but also answers N2; that is valid support, not a hallucination.
    both = [*needs, Need(id="N2", description="other")]
    assert (await SufficiencyEvaluator(CrossNeed()).evaluate(both, [item], "q")).overall_status == "SUFFICIENT"


def test_ssrf_checks_public_dns_pinning_and_input_validation(monkeypatch):
    fetcher = SafeFetcher(["allowed.example"], 1024)
    for url in ["http://allowed.example/", "https://127.0.0.1/", "https://allowed.example:444/", "https://user:pass@allowed.example/"]:
        with pytest.raises(ValueError):
            fetcher.validate(url)
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [(2,1,6,"",("127.0.0.1",443))])
    with pytest.raises(ValueError, match="Non-public"):
        fetcher.validate("https://allowed.example")
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [(2,1,6,"",("8.8.8.8",443))])
    assert fetcher.validate("https://allowed.example")[1] == "8.8.8.8"
    with pytest.raises(ValidationError):
        IngestRequest(title="x", text="a", url="https://example.org")
    with pytest.raises(ValidationError):
        NavigationDecision(decisions=[{"node_id":"x", "action":"DELETE"}], current_node_action="CONTINUE")


async def test_openrouter_jev_protocol_schema_repair_and_usage():
    requests = []
    async def handler(request):
        data = json.loads(request.content); requests.append((request.url.path, data))
        if request.url.path.endswith("decisions"):
            return httpx.Response(200, json={"answers": {key:{"type":"choice", "choice":"CONTINUE" if key == "branch_status" else "SELECT"} for key in data["questions"]}, "usage":{"input_tokens":20,"output_tokens":0,"cost":.001}})
        if len([r for r in requests if r[0].endswith("completions")]) == 1:
            return httpx.Response(200, json={"choices":[{"message":{"content":"not json"}}]})
        return httpx.Response(200, json={"choices":[{"message":{"content":json.dumps({"document_type":"note","summary":"s","routing_summary":"r"})}}]})
    repo = InMemoryGraph()
    provider = OpenRouter(settings(provider_retries=1), repo, httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    result = await provider.navigate({"candidate_children":[{"node_id":"n"}]}, "q")
    assert result.decisions[0].action == "SELECT"
    assert requests[0][0] == "/api/alpha/decisions" and requests[0][1]["model"] == "typesafe/jev-1.13"
    await provider.structured("understanding", {"text":"x"}, Understanding)
    assert len(await repo.records("usage")) == 3
    await provider.structured("understanding", {"text":"x"}, Understanding)
    assert len(requests) == 3  # prompt-version/model keyed cache
    await provider.close()


async def test_parallel_workers_are_bounded_and_provider_failure_is_visible():
    class ObservedModels(DemoModels):
        active = 0
        peak = 0
        async def navigate(self, payload, query_id):
            self.active += 1
            self.peak = max(self.peak, self.active)
            try:
                await asyncio.sleep(.01)
                return await super().navigate(payload, query_id)
            finally:
                self.active -= 1
    models = ObservedModels()
    memory = Memory(settings(max_parallel_branches=2, max_total_nodes_explored=7), InMemoryGraph(), models)
    await seed(memory)
    result = await memory.query("How do piano and harmony connect?")
    assert result["status"] == "completed" and result["nodes_explored"] <= 7
    assert models.peak == 2
    class FailedNavigation(DemoModels):
        async def navigate(self, payload, query_id):
            raise ProviderError("unavailable")
    memory.models = FailedNavigation()
    memory.retrieval.models = memory.models
    result = await memory.query("piano keys")
    assert result["status"] == "completed"
    assert any(e["event_type"] == "MODEL_FAILURE" for e in await memory.events(result["query_id"]))


async def test_contradictions_coexist_and_invalid_claim_rolls_back():
    from graph_memory.models import Assertion, Enrichment
    memory = make_memory()
    one = await memory.ingest(IngestRequest(title="source one", text="The value is 10."))
    two = await memory.ingest(IngestRequest(title="source two", text="The value is 20."))
    a = Assertion(id="a", label="First claim", text="The value is 10.", proposition="The value is 10.", confidence=.8)
    b = Assertion(id="b", label="Second claim", text="The value is 20.", proposition="The value is 20.", confidence=.7)
    await memory.repository.put([a,b], [Edge(source="a",target=one["document_id"],relation=Relation.SUPPORTED_BY),
                                      Edge(source="b",target=two["document_id"],relation=Relation.SUPPORTED_BY),
                                      Edge(source="a",target="b",relation=Relation.CONTRADICTS)])
    assert (await memory.repository.get("a")).text != (await memory.repository.get("b")).text
    assert (await memory.repository.provenance("b"))["contradictions"] == ["a"]
    class InvalidQuote(DemoModels):
        async def structured(self, operation, payload, schema, query_id=None):
            return Enrichment(assertions=[{"proposition":"invented", "assertion_type":"claim", "confidence":.9,
                                           "supporting_quote":"not in source", "contradicts_ids":[]}])
    memory.enrichment.models = InvalidQuote()
    count = len(memory.repository.nodes)
    with pytest.raises(ValueError, match="quote"):
        await memory.enrichment.enrich(one["document_id"], force=True)
    assert len(memory.repository.nodes) == count


async def test_concurrent_submit_capacity_and_query_timeout():
    class Slow(DemoModels):
        async def structured(self, *args, **kwargs):
            await asyncio.sleep(2)
            return await super().structured(*args, **kwargs)
    memory = Memory(settings(max_active_queries=1, query_timeout_seconds=1), InMemoryGraph(), Slow())
    query_id = await memory.submit("question")
    with pytest.raises(RuntimeError, match="capacity"):
        await memory.submit("other")
    task = memory.tasks[query_id]
    await task
    result = await memory.repository.read_record("query", query_id)
    assert result["status"] == "failed" and "timed out" in result["error"]


async def test_bad_provider_choices_retry_then_fail():
    calls = []
    async def bad(request):
        calls.append(request)
        return httpx.Response(200, json={"answers":{"node_0":{"type":"choice","choice":"EXECUTE"}}})
    provider = OpenRouter(settings(provider_retries=1), InMemoryGraph(), httpx.AsyncClient(transport=httpx.MockTransport(bad)))
    with pytest.raises(ProviderError):
        await provider.navigate({"candidate_children":[{"node_id":"n"}]}, "q")
    assert len(calls) == 2
    await provider.close()


def test_api_auth_ingestion_query_events_replay_and_limits():
    memory = make_memory(memory_api_token="test-secret", max_source_bytes=1024)
    with TestClient(create_app(memory)) as client:
        assert client.get("/memory/health").status_code == 401
        client.headers["Authorization"] = "Bearer test-secret"
        assert client.post("/memory/ingest", json={"title":"Piano", "text":"Piano keys create notes."}).status_code == 200
        response = client.post("/memory/query", json={"query":"Piano keys", "allow_external":False})
        assert response.status_code == 202
        query_id = response.json()["query_id"]
        with client.stream("GET", f"/memory/query/{query_id}/stream") as response:
            body = response.read().decode()
        assert "QUERY_COMPLETED" in body and "event: done" in body
        events = client.get(f"/memory/query/{query_id}/events").json()
        resumed = client.get(f"/memory/query/{query_id}/events?after=2").json()
        assert resumed == events[2:]
        assert client.get(f"/memory/query/{query_id}").json()["status"] == "completed"
        assert client.post("/memory/ingest", content=b"x"*20000).status_code == 413
        assert client.get("/memory/nodes/missing").status_code == 404
        assert client.post("/memory/ingest", json={"title":"invalid", "content_base64":base64.b64encode(b'bad PDF').decode(), "mime_type":"application/pdf"}).status_code == 422


async def hub_graph():
    """Six retrievable leaves about one concept hub."""
    from graph_memory.models import Document
    repo = InMemoryGraph()
    docs = [Document(id=f"d{i}", label=f"note {i}", text=f"Piano note number {i}.", retrieval_leaf=True) for i in range(6)]
    await repo.put([Concept(id="hub", label="Piano", preferred_label="Piano"), *docs],
                   [Edge(source=d.id, target="hub", relation=Relation.ABOUT) for d in docs])
    return repo


async def test_root_breadth_and_navigation_floor():
    from graph_memory.models import Branch, Decision, Need
    from graph_memory.retrieval import ExplorationSupervisor
    from graph_memory.trace import TraceService

    class Policy(DemoModels):
        def __init__(self, prune_all=False):
            self.prune_all = prune_all
        async def navigate(self, payload, query_id):
            return NavigationDecision(decisions=[Decision(node_id=c["node_id"], action="SELECT" if c["retrievable"] and not self.prune_all else "PRUNE")
                                                 for c in payload["candidate_children"]], current_node_action="DEAD_END" if self.prune_all else "CONTINUE")

    need = Need(id="N1", description="piano note")
    repo = await hub_graph()
    supervisor = ExplorationSupervisor(settings(max_root_children=8, max_children_per_decision=4), repo, Policy(), TraceService(repo, "q"))
    supervisor.ceiling = 100
    root = Branch(information_need_id="N1")
    await supervisor.work(root, need, [])
    assert len([b for b in supervisor.branches if b.parent_branch_id == root.id]) == 6  # all six root SELECTs, cap 8
    supervisor = ExplorationSupervisor(settings(max_root_children=8, max_children_per_decision=4), repo, Policy(), TraceService(repo, "q"))
    supervisor.ceiling = 100
    hub = Branch(information_need_id="N1", node_id="hub", path=["hub"], depth=1)
    await supervisor.work(hub, need, [])
    assert len([b for b in supervisor.branches if b.parent_branch_id == hub.id]) == 4  # below the root the cap stays 4

    repo = await hub_graph()
    supervisor = ExplorationSupervisor(settings(min_root_children=2), repo, Policy(prune_all=True), TraceService(repo, "q"))
    await supervisor.explore([need])
    events = await repo.records("event", "q:")
    floor = [e for e in events if e["event_type"] == "NAVIGATION_FLOOR"]
    assert len(floor) == 1 and len(floor[0]["metadata"]["nodes"]) == 2
    assert supervisor.explored == 2  # the two top-ranked candidates were still explored


async def test_navigation_excerpts_only_for_retrievable_leaves():
    from graph_memory.models import Assertion, Chunk, Document, Need, Section
    from graph_memory.retrieval import ExplorationSupervisor, preview
    from graph_memory.trace import TraceService
    text = "First boil the chicken in cola, then fry it with flour and egg. " * 30
    nodes = [Document(id="leaf", label="d", text=text, retrieval_leaf=True), Document(id="container", label="c", text=text),
             Chunk(id="chunk", label="k", text=text, token_count=1, section_path=[], context_header="", order=0),
             Assertion(id="assertion", label="a", text=text, proposition=text, confidence=.5),
             Section(id="section", label="s", text=text, level=1, order=0, section_path=[]),
             Concept(id="concept", label="Frying", preferred_label="Frying", description=text)]
    views = {n.id: preview(n, excerpt_tokens=80) for n in nodes}
    assert {i for i, v in views.items() if "excerpt" in v} == {"leaf", "chunk", "assertion"}
    assert text.startswith(views["leaf"]["excerpt"]) and 70 <= token_count(views["leaf"]["excerpt"]) <= 80
    assert "excerpt" not in preview(nodes[0])

    class Spy(DemoModels):
        payloads = []
        async def navigate(self, payload, query_id):
            self.payloads.append(payload)
            return await super().navigate(payload, query_id)
    repo = await hub_graph()
    supervisor = ExplorationSupervisor(settings(navigation_excerpt_tokens=5), repo, Spy(), TraceService(repo, "q"))
    await supervisor.explore([Need(id="N1", description="piano note")])
    root = Spy.payloads[0]["candidate_children"]
    assert all(("excerpt" in c) == (c["kind"] == "Document") for c in root) and any(c["kind"] == "Concept" for c in root)
    assert all(token_count(c["excerpt"]) <= 5 for c in root if "excerpt" in c)


async def test_navigation_request_shortens_excerpts_to_fit_budget():
    from graph_memory.parsing import truncate
    sent = []
    async def handler(request):
        body = json.loads(request.content); sent.append(body)
        return httpx.Response(200, json={"answers": {k: {"type": "choice", "choice": "CONTINUE" if k == "branch_status" else "PRUNE"}
                                                    for k in body["questions"]}})
    def provider(budget):
        return OpenRouter(settings(model_input_token_budget=budget), InMemoryGraph(), httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    # Worst case at candidate_limit=20: every candidate carries a full 80-token excerpt.
    excerpt = truncate(" ".join(f"word{i}" for i in range(400)), 80)
    state = {"information_need": {"id": "N1", "description": "crispy breading"}, "candidate_children": [
        {"node_id": f"n{i}", "label": "note", "routing_summary": "Explore for frying notes."} for i in range(20)]}
    await provider(100000).navigate(state, "q")
    base = token_count(json.dumps(sent[-1], ensure_ascii=False))  # the same request without excerpts
    full = {**state, "candidate_children": [{**c, "excerpt": excerpt} for c in state["candidate_children"]]}
    budget = base + 600  # 20 x 80 excerpt tokens cannot fit; 20 x 20 can
    await provider(budget).navigate(full, "q")
    shortened = sent[-1]
    assert token_count(json.dumps(shortened, ensure_ascii=False)) <= budget
    assert all(0 < token_count(c["excerpt"]) < 80 and excerpt.startswith(c["excerpt"]) for c in shortened["state"]["candidate_children"])


async def test_original_question_reaches_candidate_search_and_relevance():
    from graph_memory.models import QueryRequest
    class Repo(InMemoryGraph):
        queries = []
        async def candidates(self, query, vector, limit, parent=None, kind=None):
            self.queries.append((query, bool(vector)))
            return await super().candidates(query, vector, limit, parent, kind)
    class Models(DemoModels):
        relevance = []
        async def structured(self, operation, payload, schema, query_id=None):
            if operation == "relevance":
                self.relevance.append(payload)
            return await super().structured(operation, payload, schema, query_id)
    memory = Memory(settings(), Repo(), Models(), DemoExternal())
    await seed(memory)
    Repo.queries.clear()
    question = "How do piano and harmony connect?"  # the demo decomposition rewrites it into two needs
    result = await memory.query(QueryRequest(query=question, allow_external=False))
    assert question not in [n["description"] for n in result["information_needs"]]
    assert (question, True) in Repo.queries  # searched with its own embedding as well as each need
    assert Models.relevance and all(p["question"] == question for p in Models.relevance)


async def test_answer_length_cap_per_query_overrides_the_setting():
    from graph_memory.models import QueryRequest
    class Models(DemoModels):
        limits = []
        async def structured(self, operation, payload, schema, query_id=None):
            if operation == "synthesis":
                self.limits.append(payload.get("answer_max_words"))
            return await super().structured(operation, payload, schema, query_id)
    memory = Memory(settings(answer_max_words=60), InMemoryGraph(), Models())
    await seed(memory)
    await memory.query(QueryRequest(query="piano keys", allow_external=False))
    await memory.query(QueryRequest(query="piano keys", allow_external=False, answer_max_words=84))
    assert Models.limits == [60, 84]


async def test_reembedding_after_a_dimension_change_is_resumable():
    from graph_memory.ingestion import embedding_text
    from graph_memory.migrate import reembed
    memory = make_memory(embedding_dimensions=64)  # the demo embedder returns 64 dimensions
    await seed(memory)
    repo = memory.repository
    embedded = [n for n in repo.nodes.values() if n.embedding]
    for node in embedded:  # as if written by an 8-dimension model
        repo.nodes[node.id] = node.model_copy(update={"embedding": [1.0] * 8})
    result = await reembed(repo, memory.models, memory.settings, batch=2)
    assert result["nodes"] == len(embedded) and embedded
    for node in embedded:
        assert repo.nodes[node.id].embedding == await memory.models.embed(embedding_text(repo.nodes[node.id]))
    assert (await reembed(repo, memory.models, memory.settings))["nodes"] == 0


async def test_external_sources_are_explored_with_fresh_budget():
    # The first pass exhausts its node budget; newly ingested sources are still examined directly.
    memory = make_memory(max_total_nodes_explored=2)
    await seed(memory)
    result = await memory.query(QUESTIONS[3])
    practice = next(c for c in result["coverage"]["coverage"] if c["information_need_id"] == "N2")
    assert practice["status"] == "COVERED", result["coverage"]
    assert result["nodes_explored"] > 2  # the external pass ran on its own budget
    ingested = next(e["metadata"] for e in await memory.events(result["query_id"]) if e["event_type"] == "SOURCE_INGESTED")
    # Query-time ingestion skips taxonomy classification; it completes in the background afterwards.
    assert ingested["concepts"] == "deferred"
    await asyncio.gather(*memory.ingestion.background)
    assert any(e.relation == Relation.ABOUT and e.source == ingested["document_id"] for e in memory.repository.edges.values())
