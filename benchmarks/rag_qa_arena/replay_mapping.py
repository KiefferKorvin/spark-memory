r"""Why are extracted concepts not mapped to UNESCO? Replays extraction + classification on 20 sampled documents.

Writes nothing to the bench graph but thesaurus embeddings (unesco.embed_taxonomy: idempotent, about $0.01 once);
link_taxonomy() only returns nodes/edges, and the negative verdict cache is bypassed so every concept is decided
afresh. About $0.10 of model calls.
    .venv\Scripts\python benchmarks\rag_qa_arena\replay_mapping.py [run]    -> runs/<run>/mapping_replay.json, mapping_review.csv
    ... replay_mapping.py [run] --probes      classifies four fixed concepts and prints the candidates shown (cents)
    ... replay_mapping.py [run] --calibrate runs\<run>\mapping_review.csv
        precision of accepted links per threshold, from the "accept" column you filled with y/n (free)
"""
import argparse
import asyncio
import csv
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
from graph_memory.models import ConceptSpec, Understanding  # noqa: E402
from graph_memory.ontology import OntologyService, skip_record  # noqa: E402
from graph_memory.parsing import truncate  # noqa: E402
from graph_memory.unesco import embed_taxonomy  # noqa: E402

DOMAINS = ("lifestyle", "recreation", "science", "technology", "writing")
PROBES = [ConceptSpec(label="Crispy Frying Technique", broader=["Frying"]),
          ConceptSpec(label="Grand Prix tennis circuit", broader=["tennis tours"]),
          ConceptSpec(label="Schengen Visa Rejection", broader=["Visa Application Outcomes"]),
          ConceptSpec(label="Frying Temperature", broader=["Frying"])]


class Spy:
    """Forwards to the real provider and keeps each taxonomy decision with the candidates it was shown."""
    def __init__(self, inner):
        self.inner, self.decisions, self.calls = inner, {}, 0

    async def structured(self, operation, payload, schema, query_id=None):
        result = await self.inner.structured(operation, payload, schema, query_id)
        if operation == "taxonomy_resolution":
            self.calls += 1
            labels = {e["id"]: f"{e['label']} [{e['origin']}]" for e in payload["existing"]}
            self.decisions[payload["candidate"]["label"]] = {
                "shown": list(labels.values()), "reuse": labels.get(result.reuse_id, result.reuse_id),
                "parents": [labels.get(i, i) for i in result.parent_ids], "confidence": result.confidence}
        return result

    def __getattr__(self, name):
        return getattr(self.inner, name)


def calibrate(path):
    """Lowest threshold whose accepted links are at least 90% correct according to the reviewer's labels."""
    rows = [r for r in csv.DictReader(open(path, encoding="utf-8-sig")) if r["proposal"] and r["accept"].strip()]
    labeled = [(float(r["confidence"]), r["accept"].strip().lower() in ("y", "yes", "1", "true")) for r in rows]
    if not labeled:
        raise SystemExit(f"{path}: no labeled rows (fill the accept column with y or n)")
    print(f"{len(labeled)} labeled proposals, {sum(ok for _, ok in labeled)} correct")
    print("threshold  accepted  precision")
    chosen = None
    for step in range(50, 100, 5):
        accepted = [ok for confidence, ok in labeled if confidence >= step / 100]
        precision = sum(accepted) / len(accepted) if accepted else None
        print(f"   {step / 100:.2f}   {len(accepted):7}   {'-' if precision is None else f'{precision:.2f}'}")
        if chosen is None and precision is not None and precision >= .9 and len(accepted) >= 5:
            chosen = step / 100
    print(f"proposed TAXONOMY_MATCH_THRESHOLD={chosen}" if chosen else "no threshold reaches 0.90 precision on 5+ links")


async def main(args):
    run = Path(__file__).resolve().parent / "runs" / args.run
    settings = Settings(_env_file=MEMORY / ".env", neo4j_uri="bolt://localhost:7688", neo4j_password="bench-local-only")
    repo = Neo4jGraph(settings)
    models = Spy(OpenRouter(settings, repo))
    ontology = OntologyService(repo, models, settings.taxonomy_match_threshold, settings.primary_ontology,
                               settings.embedding_model, settings.taxonomy_provisional_threshold, cache=False)
    outcomes = []
    original = ontology.classify

    async def classify(spec, pending, query_id=None):
        try:
            concept, anchors, extra, edges = await original(spec, pending, query_id)
        except (ValueError, ProviderError) as exc:
            # The same reason code ingestion persists in its classification_review records.
            outcomes.append({"label": spec.label, "description": spec.description, "broader": spec.broader,
                             "outcome": skip_record(spec, exc)["reason"], "decision": models.decisions.get(spec.label)})
            raise
        how = "exact UNESCO label" if concept.origin == "UNESCO" and not models.decisions.get(spec.label) else \
              "reused UNESCO concept" if concept.origin == "UNESCO" else "new LOCAL child"
        status = "provisional" if concept.metadata.get("classification_status") == "provisional" else "linked"
        outcomes.append({"label": spec.label, "description": spec.description, "outcome": f"{status}: {how}",
                         "anchors": [a.label for a in anchors], "decision": models.decisions.get(spec.label)})
        return concept, anchors, extra, edges
    ontology.classify = classify

    try:
        await repo.initialize()  # creates the ontology vector index when missing
        print("thesaurus embeddings:", await embed_taxonomy(repo, models.inner, settings), flush=True)
        if args.probes:
            for spec in PROBES:
                await ontology.link_taxonomy([spec], {})
                decision = models.decisions.get(spec.label) or {}
                print(f"\n{spec.label} (broader {spec.broader}) -> {outcomes[-1]['outcome']}  confidence {decision.get('confidence')}"
                      f"\n  proposed: reuse {decision.get('reuse')} parents {decision.get('parents')}"
                      f"\n  shown: {', '.join(decision.get('shown', []))}")
            return
        corpus = [json.loads(line) for line in (run / "corpus.jsonl").read_text(encoding="utf-8").splitlines()]
        ingested = {r["doc_id"]: r["kg_document_id"] for r in map(json.loads, (run / "ingested.jsonl").read_text(encoding="utf-8").splitlines()) if "error" not in r}
        rng = random.Random(1)
        sample = [d for domain in DOMAINS for d in rng.sample([d for d in corpus if d["domain"] == domain], 4)]
        per_doc = []
        for doc in sample:
            node = await repo.get(ingested[doc["id"]])
            understanding = await models.structured("understanding", {
                "title": node.label, "context": getattr(node, "context_header", node.label),
                "text": truncate(node.text or doc["text"], min(10000, settings.model_input_token_budget // 2)),
                "summary_input": False, "mime_type": "text/plain"}, Understanding)
            start, calls = len(outcomes), models.calls
            await ontology.link_taxonomy(understanding.concepts, {})
            for o in outcomes[start:]:
                o.update(doc=doc["id"], domain=doc["domain"], title=node.label)
            per_doc.append((len(understanding.concepts), models.calls - calls))
            print(doc["id"], node.label, "->", Counter(o["outcome"].split(":")[0] for o in outcomes[start:]), flush=True)
    finally:
        await models.inner.close()
        await repo.close()
    (run / "mapping_replay.json").write_text(json.dumps(outcomes, indent=1, ensure_ascii=False), encoding="utf-8")
    # Review sheet for threshold calibration: one row per model decision; the reviewer fills "accept" with y/n.
    with open(run / "mapping_review.csv", "w", encoding="utf-8-sig", newline="") as f:
        sheet = csv.writer(f)
        sheet.writerow(["concept", "description", "domain", "proposal", "confidence", "outcome", "accept"])
        for o in outcomes:
            if d := o.get("decision"):
                proposal = f"reuse {d['reuse']}" if d["reuse"] else "parents " + "; ".join(d["parents"]) if d["parents"] else ""
                sheet.writerow([o["label"], o.get("description", ""), o.get("domain", ""), proposal, d["confidence"], o["outcome"], ""])
    reasons = Counter(o["outcome"] for o in outcomes)
    print(f"\n{len(outcomes)} concepts from {len(sample)} documents: {sum(c for c, _ in per_doc) / len(per_doc):.1f} concepts "
          f"and {sum(n for _, n in per_doc) / len(per_doc):.1f} taxonomy_resolution calls per document")
    for reason, n in reasons.most_common():
        print(f"  {n:3}  {100*n/len(outcomes):4.0f}%  {reason}")
    print("by domain (linked+provisional/total):", {d: f"{sum(o['domain'] == d and o['outcome'].startswith(('linked', 'provisional')) for o in outcomes)}"
                                                        f"/{sum(o['domain'] == d for o in outcomes)}" for d in DOMAINS})
    print(f"review sheet: {run / 'mapping_review.csv'} (fill accept with y/n, then --calibrate it)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", nargs="?", default="preview")
    parser.add_argument("--probes", action="store_true", help="classify the four fixed probe concepts only")
    parser.add_argument("--calibrate", type=Path, help="labeled mapping_review.csv: precision per threshold, no model calls")
    arguments = parser.parse_args()
    if arguments.calibrate:
        calibrate(arguments.calibrate)
    else:
        asyncio.run(main(arguments))
