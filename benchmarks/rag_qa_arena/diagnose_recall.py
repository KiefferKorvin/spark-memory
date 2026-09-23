"""Where did the graph memory lose each gold document? Reads a finished run's query traces from the bench Neo4j.

    .venv\\Scripts\\python benchmarks\\rag_qa_arena\\diagnose_recall.py [run]      -> runs/<run>/recall_diagnosis.json
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from neo4j import GraphDatabase

RUN = Path(__file__).resolve().parent / "runs" / (sys.argv[1] if len(sys.argv) > 1 else "preview")
questions = {q["qid"]: q for q in json.loads((RUN / "questions.json").read_text(encoding="utf-8"))}
answers = {(a["method"], a["qid"]): a for a in map(json.loads, (RUN / "answers.jsonl").read_text(encoding="utf-8").splitlines())}
kg = json.loads((RUN / "state.json").read_text(encoding="utf-8"))["config"]["kg"]
cap = kg.get("max_root_children", kg["max_children_per_decision"])  # breadth of root decisions
corpus = {d["id"]: d for d in map(json.loads, (RUN / "corpus.jsonl").read_text(encoding="utf-8").splitlines())}

driver = GraphDatabase.driver("bolt://localhost:7688", auth=("neo4j", "bench-local-only"))
with driver.session() as s:
    doc_nodes, node_doc = defaultdict(set), {}
    for r in s.run("MATCH (src:Source)-[:PROVIDES]->(:Document)-[:CONTAINS*0..]->(n) RETURN src.payload AS p, n.id AS id").data():
        node_doc[r["id"]] = json.loads(r["p"])["metadata"].get("bench_doc_id")
        doc_nodes[node_doc[r["id"]]].add(r["id"])

    stages, examples = Counter(), defaultdict(list)
    wanted, top4, kinds, explored_per_need = [], [], Counter(), []
    for qid, q in questions.items():
        a = answers[("kg_memory", qid)]
        ev = [json.loads(r["p"]) for r in s.run("MATCH (r:MemoryRecord {category:'event'}) WHERE r.key STARTS WITH $p "
                                                  "RETURN r.payload AS p ORDER BY r.key", p=a["query_id"] + ":").data()]
        needs = next(e["metadata"]["information_needs"] for e in ev if e["event_type"] == "QUERY_DECOMPOSED")
        explored_per_need.append(a["nodes_explored"] / len(needs))
        rank, pruned, spawned, found = {}, defaultdict(set), set(), set()
        for e in ev:
            t, m = e["event_type"], e["metadata"]
            if t == "CANDIDATES_GENERATED":
                for i, n in enumerate(m["nodes"]):
                    rank.setdefault(n["node_id"], i + 1)
            elif t == "NODE_PRUNED":
                pruned[e["node_id"]].add(m["reason"])
            elif t == "BRANCH_SPAWNED" and e.get("node_id"):
                spawned.add(e["node_id"])
            elif t == "EVIDENCE_FOUND":
                found.add(m["evidence"]["source_node_id"])
        # Root decisions: how many children the policy wanted versus what the breadth cap allowed.
        for root in {e["branch_id"] for e in ev if e["event_type"] == "BRANCH_SPAWNED" and not e.get("node_id")}:
            order = [c["node_id"] for c in next((e["metadata"]["nodes"] for e in ev if e["event_type"] == "CANDIDATES_GENERATED" and e["branch_id"] == root), [])]
            chosen = [e["node_id"] for e in ev if e["event_type"] == "BRANCH_SPAWNED" and e.get("parent_branch_id") == root]
            capped = [e for e in ev if e["event_type"] == "NODE_PRUNED" and e["branch_id"] == root and e["metadata"]["reason"] == "branch_budget"]
            wanted.append(len(chosen) + len(capped))
            top4 += [order.index(n) < cap for n in chosen if n in order]
        for e in ev:
            if e["event_type"] == "CANDIDATES_GENERATED" and not e.get("node_id"):
                kinds.update(n["kind"] for n in e["metadata"]["nodes"])
        for g in [g for g in q["gold"] if g not in a.get("retrieved", [])]:
            nodes = doc_nodes[g]
            stage = ("evidence found but not in final context" if nodes & found else
                     "explored, relevance check rejected" if nodes & spawned else
                     "candidate, pruned by navigation policy" if any("policy" in pruned[n] for n in nodes) else
                     "candidate, cut by breadth cap" if any("branch_budget" in pruned[n] for n in nodes) else
                     "candidate, never scheduled (global budget)" if nodes & rank.keys() else "never a candidate")
            stages[stage] += 1
            examples[stage].append({"qid": qid, "question": q["question"], "needs": [n["description"] for n in needs], "gold": g,
                                    "best_rank": min((rank[n] for n in nodes if n in rank), default=None),
                                    "dense_found_it": g in answers.get(("dense_rag", qid), {}).get("retrieved", []),
                                    "gold_text": corpus[g]["text"][:300]})
driver.close()

total = sum(len(q["gold"]) for q in questions.values())
print(f"missed gold documents: {sum(stages.values())}/{total}")
for stage, n in stages.most_common():
    print(f"  {n:3}  {stage}  (dense RAG found {sum(i['dense_found_it'] for i in examples[stage])})")
print(f"root candidate kinds: {dict(kinds)}")
print(f"root decisions wanting more children than the root cap ({cap}): {100 * mean(w > cap for w in wanted):.0f}%   "
      f"explored children in the top {cap} by score: {100 * mean(top4):.0f}%   nodes explored per need: {mean(explored_per_need):.1f}")
(RUN / "recall_diagnosis.json").write_text(json.dumps(examples, indent=1, ensure_ascii=False), encoding="utf-8")
