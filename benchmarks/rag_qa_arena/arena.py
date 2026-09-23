"""RAG-QA Arena (Han et al., 2024) for the progressive graph memory versus standard RAG baselines.

Faithful to the benchmark: RobustQA/LoTTE test collections, LFRQA reference answers, the answer template
`ans_generation_v1.cfg`, and the pairwise judge against LFRQA with the benchmark's own system prompt,
few-shot examples, answer ordering and rating parser. Methods share one answer model and one mixed corpus.

Subset: `--per-domain` questions per LoTTE domain; the corpus is their gold documents plus `--negatives`
BM25 hard negatives per question mined from the full domain collection (BM25 retrieves over the same corpus,
so these distractors are deliberately hard for it). Every stage is cached in runs/<run>/ and resumes.

From memory/:
    .venv\\Scripts\\python benchmarks\\rag_qa_arena\\arena.py --run preview
    .venv\\Scripts\\python -m http.server 8777 --directory benchmarks\\rag_qa_arena   # /viewer.html?run=preview
"""
import argparse
import asyncio
import heapq
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
MEMORY = HERE.parents[1]
sys.path.insert(0, str(MEMORY / "src"))
from graph_memory.config import Settings  # noqa: E402
from graph_memory.models import IngestRequest, QueryRequest  # noqa: E402
from graph_memory.service import Memory  # noqa: E402

DOMAINS = ["lifestyle", "recreation", "science", "technology", "writing"]
METHODS = ["closed_book", "bm25_rag", "dense_rag", "oracle_rag", "kg_memory"]
NO_ANSWER = "I couldn't find an answer."
STOP = set("a an the of to in on for and or is are was were be been it its this that with as by at from how what why "
           "when where which who whom do does did can could should would will i you my your me we our not no if so".split())
CLOSED_BOOK = ("Provide a helpful answer to the query. Query is in the <query></query> tags.\n\n<query>\n{q}\n</query>\n\n"
               "First, think step by step, and put your thinking in <thinking> tags. Your thinking must be shorter than "
               "50 words. Then, provide your answer.")


def tokens(text):
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOP]


def template(arena, name):
    text = (arena / "templates" / name).read_text(encoding="utf-8")
    return text.split('"""')[1]


# --- Ported verbatim in behaviour from rag-qa-arena code/utils.py and code/compute_correlation.py ---
def remove_elements(response, open="<thinking>", close="</thinking>"):
    start, end = response.find(open), response.find(close)
    if start >= 0 and end >= 0 and start <= end:
        response = response[end + len(close):]
    return response.strip()


def process_response(response):
    for open, close in (("<thinking>", "</thinking>"), ("Thinking: ", "Answer: "), ("Thoughts: ", "Answer: "), ("Thought: ", "Answer: ")):
        response = remove_elements(response, open, close)
    for tag in ("Answer: ", "Answer:\n", "<answer>"):
        response = response.replace(tag, "").replace("</answer>", "")
    return NO_ANSWER if response == "FAIL TO GENERATE ANS." else response


def parse_vote(pred):
    found = re.findall(r"<rating>.*\d+.*</rating>", pred, flags=re.DOTALL)
    rating, score = found[0] if found else pred, 0
    for i in range(3):
        if str(i) in rating:
            score = i
    return score


# --- Data preparation ---
def collection(lotte, domain):
    base = lotte / domain / "test"
    if (base / "collection.tsv").exists():
        with open(base / "collection.tsv", encoding="utf-8") as f:
            for line in f:
                pid, text, *_ = line.rstrip("\n\r").split("\t")
                yield pid, text
    else:  # The HF mirror ships lifestyle as JSONL with the same line-number ids.
        with open(base / "test_collection.jsonl", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                yield str(row["doc_id"]), row["text"]


def mine(lotte, domain, questions, k):
    """Two streaming BM25 passes over the full collection: gold texts plus top-k non-gold hits per question."""
    qterms = [set(tokens(q["question"])) for q in questions]
    vocab, gold_ids = set().union(*qterms), {g for q in questions for g in q["gold"]}
    df, n, total, gold = Counter(), 0, 0, {}
    for pid, text in collection(lotte, domain):
        if int(pid) != n:  # RobustQA's LoTTE invariant; also catches a truncated or misassembled download.
            raise ValueError(f"{domain} collection corrupt at line {n}: pid {pid}")
        words = tokens(text)
        n, total = n + 1, total + len(words)
        df.update(vocab.intersection(words))
        if pid in gold_ids:
            gold[pid] = text
    idf = {t: math.log(1 + (n - df[t] + .5) / (df[t] + .5)) for t in vocab}
    heaps = [[] for _ in questions]
    for pid, text in collection(lotte, domain):
        words = tokens(text)
        tf = Counter(w for w in words if w in vocab)
        if not tf:
            continue
        norm = 1.2 * (.25 + .75 * len(words) / (total / n))
        for heap, terms, q in zip(heaps, qterms, questions):
            score = sum(idf[t] * tf[t] * 2.2 / (tf[t] + norm) for t in terms if t in tf)
            if score and pid not in q["gold"]:
                item = (score, pid, text)
                heapq.heappush(heap, item) if len(heap) < k else heapq.heappushpop(heap, item)
    return gold, heaps


def prepare(args, run):
    questions, corpus = [], {}
    for domain in args.domains:
        refs = json.loads((args.references / f"{domain}_references.json").read_text(encoding="utf-8"))
        with open(args.arena / "data" / f"annotations_{domain}_with_citation.jsonl", encoding="utf-8") as f:
            pool = [json.loads(line) for line in f]
        pool = [a for a in pool if a["question"] in refs and a["gold_doc_ids"]]
        picked = random.Random(args.seed).sample(pool, args.per_domain)
        batch = [{"qid": a["qid"], "domain": domain, "question": a["question"],
                  "reference": refs[a["question"]]["faithful_answer"], "gold": [str(g) for g in a["gold_doc_ids"]]} for a in picked]
        run.log(f"{domain}: mining BM25 hard negatives over the full test collection")
        started = time.monotonic()
        gold, heaps = mine(args.lotte, domain, batch, args.negatives)
        for q, heap in zip(batch, heaps):
            missing = [g for g in q["gold"] if g not in gold]
            if missing:
                raise ValueError(f"{q['qid']}: gold documents {missing} absent from {domain} collection")
            q["gold"] = [f"{domain}-{g}" for g in q["gold"]]
            for pid in q["gold"]:
                corpus[pid] = {"id": pid, "domain": domain, "text": gold[pid.split("-", 1)[1]], "role": "gold"}
            for _, pid, text in sorted(heap, reverse=True):
                corpus.setdefault(f"{domain}-{pid}", {"id": f"{domain}-{pid}", "domain": domain, "text": text, "role": "negative"})
        questions += batch
        run.log(f"{domain}: {len(batch)} questions, corpus now {len(corpus)} documents ({time.monotonic()-started:.0f}s)")
    with open(run.dir / "corpus.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(d) + "\n" for d in corpus.values())
    # Written last: its presence marks preparation complete.
    (run.dir / "questions.json").write_text(json.dumps(questions, indent=1), encoding="utf-8")


def passages(corpus):
    """RobustQA process_passages: 100 whitespace-separated words per passage."""
    result = []
    for doc in corpus:
        words = doc["text"].split(" ")
        result += [{"doc": doc["id"], "text": " ".join(words[i:i+100])} for i in range(0, len(words), 100)]
    return result


class BM25:
    def __init__(self, texts):
        self.docs = [Counter(tokens(t)) for t in texts]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.lengths) / len(self.docs)
        df = Counter(t for d in self.docs for t in d)
        self.idf = {t: math.log(1 + (len(self.docs) - c + .5) / (c + .5)) for t, c in df.items()}

    def top(self, query, k):
        terms = set(tokens(query))
        scores = [sum(self.idf[t] * d[t] * 2.2 / (d[t] + 1.2 * (.25 + .75 * n / self.avg)) for t in terms if t in d)
                  for d, n in zip(self.docs, self.lengths)]
        return sorted(range(len(scores)), key=lambda i: -scores[i])[:k]


# --- Model access for baselines and judge (the KG pipeline uses its own client) ---
class OpenRouter:
    def __init__(self, key):
        self.http = httpx.AsyncClient(base_url="https://openrouter.ai/api/v1", timeout=180,
                                      headers={"Authorization": f"Bearer {key}", "X-Title": "RAG-QA Arena benchmark"})

    async def post(self, path, body, pace=None):
        error = "no attempt"
        for attempt in range(6):
            if pace:
                await pace()
            try:
                response = await self.http.post(path, json=body)
                if response.status_code == 200 and "error" not in (obj := response.json()):
                    return obj
                error = f"HTTP {response.status_code}: {response.text[:200]}"
                if response.status_code in (400, 401, 402, 403, 404):
                    break
            except (httpx.HTTPError, ValueError) as exc:
                error = type(exc).__name__
            await asyncio.sleep(min(30, 2 ** attempt))
        raise RuntimeError(f"{path}: {error}")

    async def chat(self, model, messages, max_tokens, options=None, pace=None):
        obj = await self.post("/chat/completions", {**(options or {}), "model": model, "messages": messages,
                                                     "max_tokens": max_tokens, "temperature": 0}, pace)
        return obj["choices"][0]["message"].get("content") or "", (obj.get("usage") or {}).get("cost") or 0

    async def embed(self, model, texts):
        vectors = []
        for i in range(0, len(texts), 64):
            obj = await self.post("/embeddings", {"model": model, "input": texts[i:i+64]})
            for row in sorted(obj["data"], key=lambda r: r["index"]):
                norm = math.sqrt(sum(x * x for x in row["embedding"])) or 1
                vectors.append([x / norm for x in row["embedding"]])
        return vectors


def pacer(rpm):
    """Spaces requests under a per-minute cap (OpenRouter limits new accounts to 20 RPM on some models)."""
    lock, gap, slot = asyncio.Lock(), 60 / rpm, [0.0]

    async def wait():
        async with lock:
            now = time.monotonic()
            await asyncio.sleep(max(0, slot[0] - now))
            slot[0] = max(now, slot[0]) + gap
    return wait


# --- Run state: append-only JSONL caches plus state.json for the viewer ---
class Run:
    def __init__(self, path, config):
        self.dir = path
        path.mkdir(parents=True, exist_ok=True)
        self.state = {"run": path.name, "config": config, "phase": "starting", "started": time.time(), "log": [],
                      "progress": {}, "questions": [], "corpus": {}, "ingested": [], "answers": [], "judgments": [],
                      "running": {m: [] for m in METHODS}}
        self.dirty = True

    def rows(self, name):
        path = self.dir / f"{name}.jsonl"
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []

    def add(self, name, row):
        with open(self.dir / f"{name}.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        self.state[name].append(row)
        self.dirty = True

    def log(self, message):
        print(time.strftime("%H:%M:%S"), message, flush=True)
        self.state["log"] = [*self.state["log"][-299:], [time.time(), message]]
        self.dirty = True

    def phase(self, name, **progress):
        self.state["phase"] = name
        self.state["progress"].update(progress)
        self.log(f"phase: {name}")

    def save(self):
        self.state["updated"] = time.time()
        tmp = self.dir / "state.json.tmp"
        tmp.write_text(json.dumps(self.state), encoding="utf-8")
        for _ in range(10):  # Windows refuses to replace a file the viewer's server has open.
            try:
                os.replace(tmp, self.dir / "state.json")
                self.dirty = False
                return
            except PermissionError:
                time.sleep(.05)

    async def autosave(self):
        while True:
            await asyncio.sleep(1.5)
            if self.dirty:
                self.save()


def without_citations(answer, ids):
    def drop(match):
        return "" if all(part.strip() in ids for part in match.group(1).split(",")) else match.group(0)
    return re.sub(r"\s*\[([^\[\]\n]+)\]", drop, answer).strip()


async def main(args):
    config = {k: str(v) for k, v in vars(args).items()}
    run = Run(HERE / "runs" / args.run, config)
    saver = asyncio.create_task(run.autosave())
    if not (run.dir / "questions.json").exists():
        run.phase("prepare")
        await asyncio.to_thread(prepare, args, run)
    questions = json.loads((run.dir / "questions.json").read_text(encoding="utf-8"))
    corpus = run.rows("corpus")
    run.state["questions"] = questions
    run.state["corpus"] = {"documents": len(corpus), "gold": sum(d["role"] == "gold" for d in corpus),
                           "by_domain": Counter(d["domain"] for d in corpus)}

    overrides = dict(kv.split("=", 1) for kv in args.kg_setting)
    settings = Settings(_env_file=MEMORY / ".env", neo4j_uri=args.neo4j_uri, neo4j_password=args.neo4j_password,
                        external_retrievers="", **overrides)
    run.state["config"].update(answer_model=settings.synthesis_model, semantic_model=settings.semantic_model,
                               embedding_model=settings.embedding_model, navigation_model=settings.navigation_model,
                               kg={k: getattr(settings, k) for k in ("max_total_nodes_explored", "max_depth", "max_parallel_branches",
                                   "max_children_per_decision", "candidate_limit", "context_token_budget", "query_timeout_seconds")})
    client = OpenRouter(settings.openrouter_api_key.get_secret_value())
    memory = Memory.from_settings(settings)
    run.log("initializing graph memory (Neo4j schema, UNESCO import)")
    await memory.initialize()
    try:
        await ingest(args, run, memory, corpus)
        await evaluate(args, run, memory, client, settings, questions, corpus)
        run.phase("done")
    except Exception as exc:
        run.log(f"FAILED: {type(exc).__name__}: {exc}")
        run.state["phase"] = "failed"
        raise
    finally:
        saver.cancel()
        run.save()
        await client.http.aclose()
        await memory.close()


async def ingest(args, run, memory, corpus):
    run.state["ingested"] = [r for r in run.rows("ingested") if "error" not in r]
    run.phase("ingest", ingest={"total": len(corpus)})
    gate = asyncio.Semaphore(args.ingest_concurrency)

    async def one(doc):
        async with gate:
            started = time.monotonic()
            try:
                result = await memory.ingest(IngestRequest(text=doc["text"], source_type="api",
                                                           metadata={"bench_doc_id": doc["id"], "domain": doc["domain"]}))
                row = {"doc_id": doc["id"], "kg_document_id": result["document_id"], "duplicate": result["duplicate"],
                       "nodes": result.get("nodes_created"), "concepts": result.get("concepts_linked"),
                       "unclassified": len(result.get("unclassified_concepts") or [])}
            except Exception as exc:
                row = {"doc_id": doc["id"], "error": f"{type(exc).__name__}: {exc}"[:300]}
                run.log(f"ingest failed {doc['id']}: {row['error']}")
            run.add("ingested", {**row, "seconds": round(time.monotonic() - started, 1), "words": len(doc["text"].split())})

    for _ in range(2):  # One in-run retry for transient provider failures.
        done = {r["doc_id"] for r in run.state["ingested"] if "error" not in r}
        todo = [d for d in corpus if d["id"] not in done]
        if todo:  # A resumed run with nothing left keeps no timing, so the viewer shows no bogus rate.
            run.state["progress"]["ingest"].setdefault("started", time.time())
        await asyncio.gather(*map(one, todo))
    usage = [u for u in await memory.repository.records("usage") if not u.get("query_id")]
    run.state["progress"]["ingest"].update(finished=time.time(), calls=len(usage), cost=sum(u.get("estimated_cost") or 0 for u in usage))
    failed = len(corpus) - len({r["doc_id"] for r in run.state["ingested"] if "error" not in r})
    # Continue: the viewer shows missing documents; a rerun retries them before evaluating again.
    run.log(f"ingestion finished: {len(corpus) - failed}/{len(corpus)} documents, {len(usage)} model calls")


async def evaluate(args, run, memory, client, settings, questions, corpus):
    run.phase("index baselines")
    psgs = passages(corpus)
    bm25 = BM25([p["text"] for p in psgs])
    vectors = await client.embed(settings.embedding_model, [p["text"] for p in psgs])
    qvectors = dict(zip([q["qid"] for q in questions], await client.embed(settings.embedding_model, [q["question"] for q in questions])))
    texts = {d["id"]: d["text"] for d in corpus}
    answer_template = template(args.arena, "ans_generation_v1.cfg")
    pair_template = template(args.arena, "pairwise_lfrqa.cfg")
    system = (args.arena / "templates" / "pairwise_lfrqa_system.txt").read_text(encoding="utf-8")
    examples = json.loads((args.arena / "templates" / "pairwise_lfrqa_examples.json").read_text(encoding="utf-8"))

    def pair(query, r1, r2):
        return pair_template.replace("{x.question}", query).replace("{x.response1}", r1).replace("{x.response2}", r2)

    shots = [m for ex in examples for m in ({"role": "user", "content": pair(ex["query"], ex["response_1"], ex["response_2"])},
             {"role": "assistant", "content": f"<thinking>{ex['thinking']}</thinking><rating>{ex['label']}</rating>"})]

    async def rag(q, chosen):
        prompt = answer_template.replace("{x.passages}", "".join(f"<passage{i+1}>\n{p['text']}\n</passage>\n" for i, p in enumerate(chosen)))
        text, cost = await client.chat(settings.synthesis_model, [{"role": "user", "content": prompt.replace("{x.question}", q["question"])}],
                                       args.answer_tokens, settings.openrouter_options)
        return {"pred": process_response(text), "cost": cost, "retrieved": list(dict.fromkeys(p["doc"] for p in chosen))}

    async def closed_book(q):
        text, cost = await client.chat(settings.synthesis_model, [{"role": "user", "content": CLOSED_BOOK.format(q=q["question"])}],
                                       args.answer_tokens, settings.openrouter_options)
        return {"pred": process_response(text), "cost": cost}

    async def bm25_rag(q):
        return await rag(q, [psgs[i] for i in bm25.top(q["question"], args.passages)])

    async def dense_rag(q):
        qv = qvectors[q["qid"]]
        ranked = sorted(range(len(psgs)), key=lambda i: -sum(a * b for a, b in zip(qv, vectors[i])))
        return await rag(q, [psgs[i] for i in ranked[:args.passages]])

    async def oracle_rag(q):
        return await rag(q, [{"doc": g, "text": texts[g]} for g in q["gold"]])

    async def kg_memory(q):
        result = await memory.query(QueryRequest(query=q["question"], allow_external=False))
        usage = [u for u in await memory.repository.records("usage") if u.get("query_id") == result["query_id"]]
        extra = {"query_id": result["query_id"], "status": result.get("status"), "calls": len(usage),
                 "cost": sum(u.get("estimated_cost") or 0 for u in usage)}
        if result.get("status") != "completed":
            return {**extra, "pred": NO_ANSWER, "error": result.get("error")}
        context = result["context"]
        docs = [s.get("metadata", {}).get("bench_doc_id") for e in context for s in e["provenance"].get("sources", [])]
        return {**extra, "pred": without_citations(result["answer"], {e["id"] for e in context}),
                "retrieved": [d for d in dict.fromkeys(docs) if d], "fallback": result["answer"].startswith("Synthesis unavailable"),
                "needs": [n["description"] for n in result["information_needs"]], "coverage": result["coverage"]["overall_status"],
                "nodes_explored": result["nodes_explored"], "evidence": len(result["evidence"]), "context": len(context)}

    answerers = {"closed_book": closed_book, "bm25_rag": bm25_rag, "dense_rag": dense_rag, "oracle_rag": oracle_rag, "kg_memory": kg_memory}
    methods = [m for m in METHODS if m in args.methods]
    for name in ("answers", "judgments"):
        run.state[name] = [r for r in run.rows(name) if r["method"] not in args.redo]
        if args.redo:
            (run.dir / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in run.state[name]), encoding="utf-8")
    answered = {(r["method"], r["qid"]): r for r in run.state["answers"]}
    judged = {(r["method"], r["qid"]) for r in run.state["judgments"]}
    kg_slots = min(args.kg_concurrency, settings.max_active_queries)  # Memory.submit rejects beyond its cap.
    gates = {m: asyncio.Semaphore(kg_slots if m == "kg_memory" else args.concurrency) for m in methods}
    judge_pace = pacer(args.judge_rpm)
    run.phase("evaluate")

    async def solve(method, q):
        key = (method, q["qid"])
        async with gates[method]:
            if key not in answered:
                run.state["running"][method].append(q["qid"])
                run.dirty = True
                started = time.monotonic()
                try:
                    row = await answerers[method](q)
                except Exception as exc:
                    row = {"pred": NO_ANSWER, "error": f"{type(exc).__name__}: {exc}"[:300]}
                    run.log(f"{method} failed on {q['qid']}: {row['error']}")
                finally:
                    run.state["running"][method].remove(q["qid"])
                if "retrieved" in row:
                    row["gold_recall"] = len(set(q["gold"]) & set(row["retrieved"])) / len(q["gold"])
                answered[key] = row = {"method": method, "qid": q["qid"], **row, "seconds": round(time.monotonic() - started, 1)}
                run.add("answers", row)
        if key in judged:
            return
        pred, reference = process_response(answered[key]["pred"]), process_response(q["reference"])
        first = len(q["question"].split(" ")) % 2 == 0  # Benchmark's order rule (LFRQADataProcessor).
        order = {1: method, 2: "LFRQA"} if first else {1: "LFRQA", 2: method}
        r1, r2 = (pred, reference) if first else (reference, pred)
        try:
            text, cost = await client.chat(args.judge, [{"role": "system", "content": system}, *shots,
                                                        {"role": "user", "content": pair(q["question"], r1, r2)}], 256, pace=judge_pace)
        except Exception as exc:
            # Not recorded: the benchmark would score this as a tie; a rerun retries it instead.
            run.log(f"judge failed on {method}/{q['qid']}: {exc}")
            return
        vote = parse_vote(text)
        thinking = re.search(r"<thinking>(.*?)</thinking>", text, flags=re.DOTALL)
        run.add("judgments", {"method": method, "qid": q["qid"], "winner": order[vote] if vote else "tie",
                              "vote": vote, "order": order, "thinking": thinking.group(1).strip() if thinking else text[:500], "cost": cost})
        judged.add(key)

    await asyncio.gather(*(solve(m, q) for q in questions for m in methods))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default="preview")
    parser.add_argument("--domains", nargs="+", default=DOMAINS, choices=DOMAINS)
    parser.add_argument("--per-domain", type=int, default=10)
    parser.add_argument("--negatives", type=int, default=4, help="BM25 hard negatives per question")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    parser.add_argument("--redo", nargs="*", default=[], choices=METHODS, help="discard cached answers/judgments of these methods")
    parser.add_argument("--judge", default="openai/gpt-4-turbo", help="paper: gpt-4-0125-preview")
    parser.add_argument("--judge-rpm", type=float, default=18, help="judge requests per minute (OpenRouter new-account cap is 20)")
    parser.add_argument("--passages", type=int, default=5)
    parser.add_argument("--answer-tokens", type=int, default=512)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--kg-concurrency", type=int, default=4)
    parser.add_argument("--ingest-concurrency", type=int, default=6)
    parser.add_argument("--kg-setting", nargs="*", default=[], help="graph memory Settings overrides, e.g. query_timeout_seconds=600")
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7688")
    parser.add_argument("--neo4j-password", default="bench-local-only")
    parser.add_argument("--arena", type=Path, default=Path("C:/code/benchmarks/rag-qa-arena-src"))
    parser.add_argument("--references", type=Path, default=Path("C:/code/benchmarks/rag-qa-arena/data/data"))
    parser.add_argument("--lotte", type=Path, default=Path("C:/code/benchmarks/robustqa-acl23/data/lotte"))
    asyncio.run(main(parser.parse_args()))
