r"""Why are extracted concepts not mapped to UNESCO? Replays extraction + classification on 20 sampled documents.

Read-only against the bench graph (link_taxonomy() only returns nodes/edges); costs about $0.10 of model calls.
    .venv\Scripts\python benchmarks\rag_qa_arena\replay_mapping.py [run]    -> runs/<run>/mapping_replay.json
"""
import asyncio
import json
import random
import sys
from collections import Counter
from pathlib import Path

MEMORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MEMORY / "src"))
from graph_memory.config import Settings  # noqa: E402
from graph_memory.graph import Neo4jGraph  # noqa: E402
from graph_memory.llm import OpenRouter, ProviderError  # noqa: E402
from graph_memory.models import Understanding  # noqa: E402
from graph_memory.ontology import OntologyService, skip_record  # noqa: E402
from graph_memory.parsing import truncate  # noqa: E402

RUN = Path(__file__).resolve().parent / "runs" / (sys.argv[1] if len(sys.argv) > 1 else "preview")
OUT = RUN / "mapping_replay.json"


class Spy:
    """Forwards to the real provider and keeps each taxonomy decision with the candidates it was shown."""
    def __init__(self, inner):
        self.inner, self.decisions = inner, {}

    async def structured(self, operation, payload, schema, query_id=None):
        result = await self.inner.structured(operation, payload, schema, query_id)
        if operation == "taxonomy_resolution":
            labels = {e["id"]: f"{e['label']} [{e['origin']}]" for e in payload["existing"]}
            self.decisions[payload["candidate"]["label"]] = {
                "shown": list(labels.values()), "reuse": labels.get(result.reuse_id, result.reuse_id),
                "parents": [labels.get(i, i) for i in result.parent_ids], "confidence": result.confidence}
        return result

    def __getattr__(self, name):
        return getattr(self.inner, name)


async def main():
    settings = Settings(_env_file=MEMORY / ".env", neo4j_uri="bolt://localhost:7688", neo4j_password="bench-local-only")
    repo = Neo4jGraph(settings)
    models = Spy(OpenRouter(settings, repo))
    ontology = OntologyService(repo, models, settings.taxonomy_match_threshold, settings.primary_ontology)
    outcomes = []
    original = ontology.classify

    async def classify(spec, pending, query_id=None):
        try:
            concept, anchors, extra, edges = await original(spec, pending, query_id)
        except (ValueError, ProviderError) as exc:
            # The same reason code ingestion persists in its classification_review records.
            outcomes.append({"label": spec.label, "broader": spec.broader, "outcome": skip_record(spec, exc)["reason"],
                             "decision": models.decisions.get(spec.label)})
            raise
        how = "exact UNESCO label" if concept.origin == "UNESCO" and not models.decisions.get(spec.label) else \
              "reused UNESCO concept" if concept.origin == "UNESCO" else "new LOCAL child"
        outcomes.append({"label": spec.label, "outcome": "linked: " + how, "anchors": [a.label for a in anchors],
                         "decision": models.decisions.get(spec.label)})
        return concept, anchors, extra, edges
    ontology.classify = classify

    corpus = [json.loads(line) for line in (RUN / "corpus.jsonl").read_text(encoding="utf-8").splitlines()]
    ingested = {r["doc_id"]: r["kg_document_id"] for r in map(json.loads, (RUN / "ingested.jsonl").read_text(encoding="utf-8").splitlines()) if "error" not in r}
    rng = random.Random(1)
    sample = [d for domain in ("lifestyle", "recreation", "science", "technology", "writing")
              for d in rng.sample([d for d in corpus if d["domain"] == domain], 4)]
    try:
        for doc in sample:
            node = await repo.get(ingested[doc["id"]])
            understanding = await models.structured("understanding", {
                "title": node.label, "context": getattr(node, "context_header", node.label),
                "text": truncate(node.text or doc["text"], min(10000, settings.model_input_token_budget // 2)),
                "summary_input": False, "mime_type": "text/plain"}, Understanding)
            start = len(outcomes)
            await ontology.link_taxonomy(understanding.concepts, {})
            for o in outcomes[start:]:
                o.update(doc=doc["id"], domain=doc["domain"], title=node.label)
            print(doc["id"], node.label, "->", Counter(o["outcome"].split(":")[0] for o in outcomes[start:]), flush=True)
    finally:
        await models.inner.close()
        await repo.close()
    OUT.write_text(json.dumps(outcomes, indent=1, ensure_ascii=False), encoding="utf-8")
    reasons = Counter(o["outcome"] for o in outcomes)
    print(f"\n{len(outcomes)} concepts from {len(sample)} documents")
    for reason, n in reasons.most_common():
        print(f"  {n:3}  {100*n/len(outcomes):4.0f}%  {reason}")
    print("by domain (linked/total):", {d: f"{sum(o['domain']==d and o['outcome'].startswith('linked') for o in outcomes)}/{sum(o['domain']==d for o in outcomes)}" for d in ("lifestyle", "recreation", "science", "technology", "writing")})


asyncio.run(main())
