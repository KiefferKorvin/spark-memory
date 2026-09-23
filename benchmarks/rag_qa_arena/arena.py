"""RAG-QA Arena (Han et al., 2024) for the progressive graph memory versus standard RAG baselines.

Faithful to the benchmark: RobustQA/LoTTE test collections, LFRQA reference answers, the benchmark's answer
templates, and the pairwise judge against LFRQA with the benchmark's own system prompt, few-shot examples,
answer ordering and rating parser. Methods share one answer model and one mixed corpus.

Every method answers under one length standard, because the judge prefers "more truthful or helpful information"
and would otherwise score length instead of retrieval. --answer-standard picks it:
  reference  (default) the benchmark's `ans_generation_v2.cfg` with its "50-60 words" set, per question, to the length
             of that question's LFRQA reference, so no method is out-worded by the reference either
  50-60      `ans_generation_v2.cfg` as published; the graph memory's synthesis is capped at 60 words
  unbounded  the paper's `ans_generation_v1.cfg`, no cap
The graph memory gets the same limit through QueryRequest.answer_max_words.

Subset: `--per-domain` questions per LoTTE domain; the corpus is their gold documents plus `--negatives`
BM25 hard negatives per question mined from the full domain collection (BM25 retrieves over the same corpus,
so these distractors are deliberately hard for it). Every stage is cached in runs/<run>/ and resumes.

From memory/:
    .venv\\Scripts\\python benchmarks\\rag_qa_arena\\arena.py --run preview
    .venv\\Scripts\\python -m http.server 8777 --directory benchmarks\\rag_qa_arena   # /viewer.html?run=preview

--reset-graph   wipes the benchmark graph (in batches) and the run's ingested.jsonl before ingestion; initialize()
                then re-imports UNESCO. Refused for NEO4J_URI of memory\\.env and for port 7687 (the live graph).
--skip-judge    answers and retrieval metrics only; a later run without the flag judges whatever is missing.
ARENA_OPENROUTER_API_KEY (environment or memory\\.env) replaces OPENROUTER_API_KEY for baselines, judges and the
                graph memory, so benchmarks cannot exhaust the live app's spend cap. state.json records which was used.
A spend-cap refusal (HTTP 402, or 403 "Key limit exceeded") stops the run: one log line, no new work, state saved,
exit code 2. Rows in flight when it hit are not recorded; cached rows stay valid, so a rerun resumes.
A run refuses to resume on cached rows made with another answer model, answer standard, embedding model or judge.

Besides the arena's pairwise preference, every answer is graded against its LFRQA reference for correctness
(correctness.jsonl; a shorter answer that agrees with the reference is correct) and, for methods that read passages,
for groundedness (grounding.jsonl). The viewer's "Sourced win" counts an answer as a win only when it is correct and
its passages support it; correct but unsourced (closed book, unsupported claims), incorrect or refused is a loss.
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
from urllib.parse import urlsplit

import httpx
from dotenv import dotenv_values

HERE = Path(__file__).resolve().parent
MEMORY = HERE.parents[1]
sys.path.insert(0, str(MEMORY / "src"))
from graph_memory.config import Settings  # noqa: E402
from graph_memory.models import IngestRequest, QueryRequest  # noqa: E402
from graph_memory.service import Memory  # noqa: E402
from graph_memory.unesco import embed_taxonomy  # noqa: E402

DOMAINS = ["lifestyle", "recreation", "science", "technology", "writing"]
METHODS = ["closed_book", "bm25_rag", "dense_rag", "oracle_rag", "kg_memory", "kg_context"]
GROUNDED = ["bm25_rag", "dense_rag", "oracle_rag", "kg_memory", "kg_context"]  # methods that answer from passages
NO_ANSWER = "I couldn't find an answer."
STOP = set("a an the of to in on for and or is are was were be been it its this that with as by at from how what why "
           "when where which who whom do does did can could should would will i you my your me we our not no if so".split())
V2_LIMIT = "Your answer should not be longer than 50-60 words."  # closing sentence of ans_generation_v2.cfg
# Closed-book prompts mirror the passage templates of the same standard, without passages.
CLOSED_V2 = ("Provide a helpful answer to the query. Query is in the <query></query> tags.\n\n<query>\n{q}\n</query>\n\n"
             "Provide a helpful answer to the query. " + V2_LIMIT)
CLOSED_V1 = ("Provide a helpful answer to the query. Query is in the <query></query> tags.\n\n<query>\n{q}\n</query>\n\n"
             "First, think step by step, and put your thinking in <thinking> tags. Your thinking must be shorter than "
             "50 words. Then, provide your answer.")
# CLI alias -> (name recorded in state.json, answer template, closed-book prompt)
STANDARDS = {"reference": ("reference length (ans_generation_v2.cfg)", "ans_generation_v2.cfg", CLOSED_V2),
             "50-60": ("ans_generation_v2.cfg (50-60 words)", "ans_generation_v2.cfg", CLOSED_V2),
             "unbounded": ("ans_generation_v1.cfg (unbounded)", "ans_generation_v1.cfg", CLOSED_V1)}


def word_limit(standard, reference):
    """The answer length cap of one question under a standard, identical for every method; None means no cap."""
    return len(reference.split()) if standard == "reference" else 60 if standard == "50-60" else None


def bounded(prompt, standard, reference):
    """Sets the v2 length sentence of an answer prompt to this question's limit."""
    if standard != "reference":
        return prompt
    if V2_LIMIT not in prompt:
        raise ValueError("answer prompt lacks the ans_generation_v2.cfg length sentence")
    return prompt.replace(V2_LIMIT, f"Your answer should not be longer than {word_limit(standard, reference)} words.")
# Judges' hidden reasoning effort (--judge-reasoning) and the output cap it needs: reasoning tokens count against
# max_tokens, and GLM 5.3 cannot disable reasoning at all (HTTP 400). Models without reasoning ignore the setting.
JUDGE_TOKENS = {"minimal": 1500, "low": 4096, "medium": 8192, "high": 16384}
# Which cached rows each recorded setting produced: a changed judge invalidates its verdicts, not the answers.
PRODUCED_BY = {"answer_model": ("answers", "judgments", "grounding", "correctness"),
               "answer_standard": ("answers", "judgments", "grounding", "correctness"),
               "embedding_model": ("answers", "judgments", "grounding", "correctness"),
               "judge": ("judgments",), "grounding_judge": ("grounding",), "correctness_judge": ("correctness",),
               "judge_reasoning": ("judgments", "grounding", "correctness")}
JUDGED = ("judgments", "grounding", "correctness")


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


def context_texts(context):
    """Text and benchmark document id (None for non-corpus sources) of each graph-memory final-context item."""
    return [{"doc": next((s["metadata"]["bench_doc_id"] for s in e["provenance"].get("sources", [])
                          if s.get("metadata", {}).get("bench_doc_id")), None), "text": e["text"]} for e in context]


def kg_passages(texts, k):
    """kg_context input: the graph memory's final context cut like the baselines' corpus, first k passages."""
    return passages([{"id": t["doc"], "text": t["text"]} for t in texts])[:k]


# --- Groundedness: an extra measure; the arena's pairwise judge never sees the sources ---
GROUNDING = ("You check an answer against the passages it was written from. Split the answer into its distinct factual "
             "claims (advice, hedges and restatements of the question are not claims). For each claim decide whether the "
             "passages state or directly imply it; general knowledge that the passages lack is unsupported. "
             'Return JSON only: {"claims": [{"claim": "...", "supported": true}]}')


def grounding_input(chosen, answer):
    return "".join(f"<passage{i+1}>\n{p['text']}\n</passage{i+1}>\n" for i, p in enumerate(chosen)) + f"\n<answer>\n{answer}\n</answer>"


def parse_grounding(text):
    """(supported share, unsupported claims); the share is None when the answer makes no claims.
    Malformed output raises, so the item is retried by a rerun instead of being recorded."""
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("no JSON object in grounding verdict")
    claims = [c for c in json.loads(match.group(0))["claims"] if str(c.get("claim", "")).strip()]
    supported = [c.get("supported") is True for c in claims]
    return (sum(supported) / len(claims) if claims else None), [c["claim"] for c, ok in zip(claims, supported) if not ok]


# --- Correctness against the reference: preference judges reward detail, this asks only "is it right?" ---
CORRECTNESS = ("You grade whether an answer to a query is correct, using a reference answer that experts wrote from the "
               "relevant sources. Correct: the answer addresses the query and agrees with the reference; it may be shorter, "
               "omit details, or add details that do not contradict the reference. Incorrect: it contradicts the reference, "
               "gives a wrong or misleading answer, or does not answer the query. "
               'Return JSON only: {"correct": true, "reason": "<one sentence>"}')


def correctness_input(question, reference, answer):
    return f"<query>\n{question}\n</query>\n\n<reference>\n{reference}\n</reference>\n\n<answer>\n{answer}\n</answer>"


def parse_correctness(text):
    """(correct, reason); only a literal true is correct. Malformed output raises, so a rerun retries the item."""
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("no JSON object in correctness verdict")
    verdict = json.loads(match.group(0))
    return verdict.get("correct") is True, str(verdict.get("reason", ""))[:300]


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
class SpendCap(RuntimeError):
    """The key's spend cap or credits are exhausted: fatal for the whole run, not for one item."""


def spend_cap(status, text):
    # 403 also covers moderation refusals; only budget wording is fatal.
    return status == 402 or status == 403 and any(w in text.lower() for w in ("limit", "credit"))


def watch_spend_cap(client, run):
    """Halts the run when the graph memory's own OpenRouter client meets the spend cap; the pipeline itself
    absorbs provider errors (navigation fallback, skipped concepts), so they never reach the harness."""
    async def hook(response):
        if response.status_code in (402, 403):
            await response.aread()
            if spend_cap(response.status_code, response.text):
                run.halt(f"graph memory: HTTP {response.status_code}: {response.text[:120]}")
    client.event_hooks["response"].append(hook)


class OpenRouter:
    def __init__(self, key, transport=None):
        self.http = httpx.AsyncClient(base_url="https://openrouter.ai/api/v1", timeout=180, transport=transport,
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
                if spend_cap(response.status_code, response.text):
                    raise SpendCap(error)
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
                      "grounding": [], "correctness": [], "running": {m: [] for m in METHODS}}
        self.dirty, self.halted = True, None

    def halt(self, reason):
        """Spend cap: stop scheduling work. Rows finishing after this were in flight when it hit and may be
        degraded (e.g. navigation fallback), so callers drop them and a rerun redoes them."""
        if not self.halted:
            self.halted, self.state["phase"] = reason, "stopped"
            self.log(f"STOPPED, spend cap reached: {reason}. Cached rows stay valid; rerun to resume.")

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


def reset_refusal(uri, live_uri):
    """Why --reset-graph must not wipe `uri`, or None. Port 7687 is the live stack's and bolt's default."""
    def endpoint(value):
        parts = urlsplit(value)
        host = parts.hostname or "localhost"
        return ("localhost" if host in ("127.0.0.1", "::1") else host), parts.port or 7687
    if endpoint(uri) == endpoint(live_uri):
        return f"--reset-graph refuses {uri}: it is NEO4J_URI of memory\\.env (the live graph)"
    if endpoint(uri)[1] == 7687:
        return f"--reset-graph refuses {uri}: port 7687 belongs to the live graph"
    return None


def mixed_rows(path, produced, redo, rejudge=False):
    """Why cached rows of this run must not be mixed with new ones, or None: rows made with another answer model,
    answer standard, embedding model or judge would share a leaderboard with rows that are not comparable.
    Only the rows a changed setting produced count, and --rejudge discards every judge's rows."""
    state = path / "state.json"
    previous = json.loads(state.read_text(encoding="utf-8"))["config"] if state.exists() else {}
    changed = {k: f"{previous[k]} -> {v}" for k, v in produced.items() if k in previous and previous[k] != v}
    files = {name for k in changed for name in PRODUCED_BY[k]} - (set(JUDGED) if rejudge else set())
    kept = {json.loads(line)["method"] for name in sorted(files) if (path / f"{name}.jsonl").exists()
            for line in (path / f"{name}.jsonl").read_text(encoding="utf-8").splitlines() if json.loads(line)["method"] not in redo}
    if kept:
        return (f"runs/{path.name} holds {', '.join(sorted(files))} rows for {', '.join(sorted(kept))} made with other settings "
                f"({changed}); start a new --run, --redo those methods, or --rejudge for judge changes")
    return None


async def reset_graph(repository):
    """Deletes every node and relationship in batches, so no single transaction holds the whole graph."""
    while (await repository.run("MATCH (n) WITH n LIMIT 5000 DETACH DELETE n RETURN count(*) AS deleted"))[0]["deleted"]:
        pass


async def main(args):
    if args.reset_graph and (refusal := reset_refusal(args.neo4j_uri, Settings(_env_file=MEMORY / ".env").neo4j_uri)):
        raise SystemExit(refusal)
    overrides = dict(kv.split("=", 1) for kv in args.kg_setting)
    # A dedicated benchmark key keeps a runaway run from exhausting the live app's spend cap.
    key = os.environ.get("ARENA_OPENROUTER_API_KEY") or dotenv_values(MEMORY / ".env").get("ARENA_OPENROUTER_API_KEY")
    if key:
        overrides["openrouter_api_key"] = key
    settings = Settings(_env_file=MEMORY / ".env", neo4j_uri=args.neo4j_uri, neo4j_password=args.neo4j_password,
                        external_retrievers="", **overrides)
    produced = {"answer_model": settings.synthesis_model, "answer_standard": STANDARDS[args.answer_standard][0],
                "embedding_model": settings.embedding_model, "judge": args.judge, "grounding_judge": args.grounding_judge,
                "correctness_judge": args.correctness_judge, "judge_reasoning": args.judge_reasoning}
    if refusal := mixed_rows(HERE / "runs" / args.run, produced, args.redo, args.rejudge):
        raise SystemExit(refusal)
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

    run.state["config"].update(**produced, semantic_model=settings.semantic_model, navigation_model=settings.navigation_model,
                               api_key_source="ARENA_OPENROUTER_API_KEY" if key else "OPENROUTER_API_KEY",
                               kg={k: getattr(settings, k) for k in ("max_total_nodes_explored", "max_depth", "max_parallel_branches",
                                   "max_children_per_decision", "max_root_children", "min_root_children", "candidate_limit", "navigation_excerpt_tokens",
                                   "taxonomy_match_threshold", "taxonomy_provisional_threshold",
                                   "context_token_budget", "query_timeout_seconds")})
    client = OpenRouter(settings.openrouter_api_key.get_secret_value())
    memory = Memory.from_settings(settings)
    watch_spend_cap(memory.models.client, run)
    try:
        if args.reset_graph:
            run.log(f"resetting the benchmark graph at {args.neo4j_uri}")
            await reset_graph(memory.repository)
            (run.dir / "ingested.jsonl").unlink(missing_ok=True)
        run.log("initializing graph memory (Neo4j schema, UNESCO import)")
        await memory.initialize()
        # Semantic classification candidates; idempotent, so only a fresh graph pays (cents).
        run.log(f"thesaurus embeddings: {await embed_taxonomy(memory.repository, memory.models, settings)}")
        await ingest(args, run, memory, corpus)
        if not run.halted:
            await evaluate(args, run, memory, client, settings, questions, corpus)
        if not run.halted:
            run.phase("done")
    except SpendCap as exc:
        run.halt(str(exc))
    except Exception as exc:
        if not run.halted:  # e.g. a ProviderError raised by the spend-cap refusal the hook already reported
            run.log(f"FAILED: {type(exc).__name__}: {exc}")
            run.state["phase"] = "failed"
            raise
    finally:
        saver.cancel()
        run.save()
        await client.http.aclose()
        await memory.close()
    return 2 if run.halted else 0


async def ingest(args, run, memory, corpus):
    run.state["ingested"] = [r for r in run.rows("ingested") if "error" not in r]
    run.phase("ingest", ingest={"total": len(corpus)})
    gate = asyncio.Semaphore(args.ingest_concurrency)

    async def one(doc):
        async with gate:
            if run.halted:
                return
            started = time.monotonic()
            try:
                result = await memory.ingest(IngestRequest(text=doc["text"], source_type="api",
                                                           metadata={"bench_doc_id": doc["id"], "domain": doc["domain"]}))
                row = {"doc_id": doc["id"], "kg_document_id": result["document_id"], "duplicate": result["duplicate"],
                       "nodes": result.get("nodes_created"), "concepts": result.get("concepts_linked"),
                       "unclassified": len(result.get("unclassified_concepts") or []),
                       "extracted": result.get("concepts_extracted"), "skips": result.get("classification_skips"),
                       "provisional": result.get("concepts_provisional")}
            except Exception as exc:
                row = {"doc_id": doc["id"], "error": f"{type(exc).__name__}: {exc}"[:300]}
                if not run.halted:
                    run.log(f"ingest failed {doc['id']}: {row['error']}")
            if not run.halted:
                run.add("ingested", {**row, "seconds": round(time.monotonic() - started, 1), "words": len(doc["text"].split())})

    for _ in range(2):  # One in-run retry for transient provider failures.
        if run.halted:
            break
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
    standard = args.answer_standard
    _, answer_file, closed_prompt = STANDARDS[standard]
    answer_template = template(args.arena, answer_file)
    pair_template = template(args.arena, "pairwise_lfrqa.cfg")
    system = (args.arena / "templates" / "pairwise_lfrqa_system.txt").read_text(encoding="utf-8")
    examples = json.loads((args.arena / "templates" / "pairwise_lfrqa_examples.json").read_text(encoding="utf-8"))

    def pair(query, r1, r2):
        return pair_template.replace("{x.question}", query).replace("{x.response1}", r1).replace("{x.response2}", r2)

    shots = [m for ex in examples for m in ({"role": "user", "content": pair(ex["query"], ex["response_1"], ex["response_2"])},
             {"role": "assistant", "content": f"<thinking>{ex['thinking']}</thinking><rating>{ex['label']}</rating>"})]

    async def rag(q, chosen):
        prompt = answer_template.replace("{x.passages}", "".join(f"<passage{i+1}>\n{p['text']}\n</passage>\n" for i, p in enumerate(chosen)))
        prompt = bounded(prompt.replace("{x.question}", q["question"]), standard, q["reference"])
        text, cost = await client.chat(settings.synthesis_model, [{"role": "user", "content": prompt}],
                                       args.answer_tokens, settings.openrouter_options)
        return {"pred": process_response(text), "cost": cost, "retrieved": [d for d in dict.fromkeys(p["doc"] for p in chosen) if d]}

    async def closed_book(q):
        text, cost = await client.chat(settings.synthesis_model, [{"role": "user", "content": bounded(closed_prompt.format(q=q["question"]), standard, q["reference"])}],
                                       args.answer_tokens, settings.openrouter_options)
        return {"pred": process_response(text), "cost": cost}

    def dense(q):
        qv = qvectors[q["qid"]]
        return sorted(range(len(psgs)), key=lambda i: -sum(a * b for a, b in zip(qv, vectors[i])))[:args.passages]

    # Deterministic, so the grounding judge sees the same passages for cached answers.
    select = {"bm25_rag": lambda q: [psgs[i] for i in bm25.top(q["question"], args.passages)],
              "dense_rag": lambda q: [psgs[i] for i in dense(q)],
              "oracle_rag": lambda q: [{"doc": g, "text": texts[g]} for g in q["gold"]]}

    async def kg_memory(q):
        result = await memory.query(QueryRequest(query=q["question"], allow_external=False,
                                                 answer_max_words=word_limit(standard, q["reference"])))
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
                "nodes_explored": result["nodes_explored"], "evidence": len(result["evidence"]), "context": len(context),
                "context_texts": context_texts(context)}

    async def kg_texts(qid):
        """The graph memory's final context: from its kg_memory row, or (older rows) its persisted query record.
        Never a second graph query, so kg_context differs from kg_memory only in how the answer is written."""
        row = answered.get(("kg_memory", qid))
        if not row or row.get("status") != "completed":
            raise ValueError("no completed kg_memory answer for this question")
        if "context_texts" in row:
            return row["context_texts"]
        record = await memory.repository.read_record("query", row["query_id"])
        if not record or "context" not in record:
            raise ValueError(f"query {row['query_id']} is not in the benchmark graph")
        return context_texts(record["context"])

    async def kg_context(q):
        return await rag(q, kg_passages(await kg_texts(q["qid"]), args.passages))

    async def used(method, q):
        """The passages an answer was written from."""
        if method in select:
            return select[method](q)
        found = await kg_texts(q["qid"])
        return kg_passages(found, args.passages) if method == "kg_context" else found

    answerers = {"closed_book": closed_book, "kg_memory": kg_memory, "kg_context": kg_context,
                 **{m: (lambda pick: lambda q: rag(q, pick(q)))(pick) for m, pick in select.items()}}
    methods = [m for m in METHODS if m in args.methods]
    for name in ("answers", *JUDGED):
        run.state[name] = [] if args.rejudge and name in JUDGED else [r for r in run.rows(name) if r["method"] not in args.redo]
        if args.redo or args.rejudge:
            (run.dir / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in run.state[name]), encoding="utf-8")
    answered = {(r["method"], r["qid"]): r for r in run.state["answers"]}
    judged = {(r["method"], r["qid"]) for r in run.state["judgments"]}
    grounded = {(r["method"], r["qid"]) for r in run.state["grounding"]}
    graded = {(r["method"], r["qid"]) for r in run.state["correctness"]}
    # kg_context reuses the kg_memory result for the same question, so it waits for that answer.
    kg_ready = {q["qid"]: asyncio.Event() for q in questions}
    for q in questions:
        if ("kg_memory", q["qid"]) in answered or "kg_memory" not in methods:
            kg_ready[q["qid"]].set()
    kg_slots = min(args.kg_concurrency, settings.max_active_queries)  # Memory.submit rejects beyond its cap.
    gates = {m: asyncio.Semaphore(kg_slots if m == "kg_memory" else args.concurrency) for m in methods}
    judge_pace, ground_pace, grade_pace = pacer(args.judge_rpm), pacer(args.judge_rpm), pacer(args.judge_rpm)
    judge_options = {"reasoning": {"effort": args.judge_reasoning, "exclude": True}}
    judge_tokens = JUDGE_TOKENS[args.judge_reasoning]
    run.phase("evaluate")

    async def answer(method, q):
        key = (method, q["qid"])
        async with gates[method]:
            if key in answered or run.halted:
                return
            run.state["running"][method].append(q["qid"])
            run.dirty = True
            started = time.monotonic()
            try:
                row = await answerers[method](q)
            except SpendCap as exc:
                run.halt(str(exc))
            except Exception as exc:
                row = {"pred": NO_ANSWER, "error": f"{type(exc).__name__}: {exc}"[:300]}
                if not run.halted:
                    run.log(f"{method} failed on {q['qid']}: {row['error']}")
            finally:
                run.state["running"][method].remove(q["qid"])
            if run.halted:  # In flight when the spend cap hit, so possibly degraded: a rerun redoes it.
                return
            if "retrieved" in row:
                row["gold_recall"] = len(set(q["gold"]) & set(row["retrieved"])) / len(q["gold"])
            answered[key] = row = {"method": method, "qid": q["qid"], **row, "seconds": round(time.monotonic() - started, 1)}
            run.add("answers", row)

    async def judge(method, q):
        key = (method, q["qid"])
        if key in judged or run.halted:
            return
        pred, reference = process_response(answered[key]["pred"]), process_response(q["reference"])
        first = len(q["question"].split(" ")) % 2 == 0  # Benchmark's order rule (LFRQADataProcessor).
        order = {1: method, 2: "LFRQA"} if first else {1: "LFRQA", 2: method}
        r1, r2 = (pred, reference) if first else (reference, pred)
        try:
            text, cost = await client.chat(args.judge, [{"role": "system", "content": system}, *shots,
                                                        {"role": "user", "content": pair(q["question"], r1, r2)}], judge_tokens, judge_options, pace=judge_pace)
        except SpendCap as exc:
            return run.halt(str(exc))
        except Exception as exc:
            # Not recorded: the benchmark would score this as a tie; a rerun retries it instead.
            run.log(f"judge failed on {method}/{q['qid']}: {exc}")
            return
        vote = parse_vote(text)
        thinking = re.search(r"<thinking>(.*?)</thinking>", text, flags=re.DOTALL)
        run.add("judgments", {"method": method, "qid": q["qid"], "winner": order[vote] if vote else "tie",
                              "vote": vote, "order": order, "thinking": thinking.group(1).strip() if thinking else text[:500], "cost": cost})
        judged.add(key)

    async def ground(method, q):
        """Extra column beside the arena's pairwise metric: the share of answer claims its own passages support."""
        key = (method, q["qid"])
        if method not in GROUNDED or key in grounded or run.halted:
            return
        pred = process_response(answered[key]["pred"])
        row = {"method": method, "qid": q["qid"], "applicable": pred != NO_ANSWER}
        if row["applicable"]:
            try:
                prompt = grounding_input(await used(method, q), pred)
                text, cost = await client.chat(args.grounding_judge, [{"role": "system", "content": GROUNDING},
                                                                      {"role": "user", "content": prompt}], judge_tokens,
                                               {**judge_options, "response_format": {"type": "json_object"}}, pace=ground_pace)
                share, unsupported = parse_grounding(text)
            except SpendCap as exc:
                return run.halt(str(exc))
            except Exception as exc:
                run.log(f"grounding judge failed on {method}/{q['qid']}: {type(exc).__name__}: {exc}"[:300])
                return
            row.update(supported=share, unsupported=unsupported, cost=cost)
        run.add("grounding", row)
        grounded.add(key)

    async def grade(method, q):
        """Correct against the reference, regardless of length or sources; a refusal is not correct."""
        key = (method, q["qid"])
        if key in graded or run.halted:
            return
        pred = process_response(answered[key]["pred"])
        row = {"method": method, "qid": q["qid"], "correct": False, "reason": "no answer"}
        if pred != NO_ANSWER:
            try:
                text, cost = await client.chat(args.correctness_judge, [
                    {"role": "system", "content": CORRECTNESS},
                    {"role": "user", "content": correctness_input(q["question"], process_response(q["reference"]), pred)}], judge_tokens,
                    {**judge_options, "response_format": {"type": "json_object"}}, pace=grade_pace)
                row["correct"], row["reason"] = parse_correctness(text)
            except SpendCap as exc:
                return run.halt(str(exc))
            except Exception as exc:
                run.log(f"correctness judge failed on {method}/{q['qid']}: {type(exc).__name__}: {exc}"[:300])
                return
            row["cost"] = cost
        run.add("correctness", row)
        graded.add(key)

    async def solve(method, q):
        if method == "kg_context":
            await kg_ready[q["qid"]].wait()
        try:
            await answer(method, q)
        finally:
            if method == "kg_memory":
                kg_ready[q["qid"]].set()
        if (method, q["qid"]) in answered and not args.skip_judge:
            await asyncio.gather(judge(method, q), ground(method, q), grade(method, q))

    await asyncio.gather(*(solve(m, q) for q in questions for m in methods))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default="preview")
    parser.add_argument("--domains", nargs="+", default=DOMAINS, choices=DOMAINS)
    parser.add_argument("--per-domain", type=int, default=10)
    parser.add_argument("--negatives", type=int, default=4, help="BM25 hard negatives per question")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    parser.add_argument("--redo", nargs="*", default=[], choices=METHODS,
                        help="discard cached answers/judgments/grounding of these methods (kg_memory implies kg_context)")
    parser.add_argument("--judge", default="z-ai/glm-5.3-flash", help="paper: gpt-4-0125-preview; the default limits cost")
    parser.add_argument("--judge-rpm", type=float, default=18, help="judge requests per minute (OpenRouter new-account cap is 20)")
    parser.add_argument("--grounding-judge", default="z-ai/glm-5.3-flash", help="checks answer claims against the passages used")
    parser.add_argument("--correctness-judge", default="z-ai/glm-5.3-flash", help="grades each answer as correct or not against its reference")
    parser.add_argument("--judge-reasoning", default="minimal", choices=list(JUDGE_TOKENS), help="hidden reasoning effort of all three judges")
    parser.add_argument("--rejudge", action="store_true", help="discard every judge's cached verdicts and judge the cached answers again")
    parser.add_argument("--answer-standard", default="reference", choices=list(STANDARDS),
                        help="answer length for every method: each LFRQA reference's length, 50-60 words, or unbounded")
    parser.add_argument("--skip-judge", action="store_true", help="answers and retrieval metrics only; judge later")
    parser.add_argument("--reset-graph", action="store_true", help="wipe the benchmark graph and re-ingest (never the live one)")
    parser.add_argument("--passages", type=int, default=5)
    parser.add_argument("--answer-tokens", type=int, default=1024, help="cap including hidden reasoning; length is set by the prompt")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--kg-concurrency", type=int, default=4)
    parser.add_argument("--ingest-concurrency", type=int, default=6)
    parser.add_argument("--kg-setting", nargs="*", default=[], help="graph memory Settings overrides, e.g. query_timeout_seconds=600")
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7688")
    parser.add_argument("--neo4j-password", default="bench-local-only")
    parser.add_argument("--arena", type=Path, default=Path("C:/code/benchmarks/rag-qa-arena-src"))
    parser.add_argument("--references", type=Path, default=Path("C:/code/benchmarks/rag-qa-arena/data/data"))
    parser.add_argument("--lotte", type=Path, default=Path("C:/code/benchmarks/robustqa-acl23/data/lotte"))
    arguments = parser.parse_args()
    if "kg_memory" in arguments.redo and "kg_context" not in arguments.redo:
        arguments.redo.append("kg_context")  # kg_context is derived from the kg_memory result.
    sys.exit(asyncio.run(main(arguments)))
