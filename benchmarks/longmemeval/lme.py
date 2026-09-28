"""LongMemEval (Wu et al., ICLR 2025) for the associative memory against standard retrieval. Every arm ranks the same
rounds (a user turn with the assistant reply); one reader model answers every arm from the official prompt and the
same token budget, so only the memory differs.

Data: longmemeval_s_cleaned.json (HF xiaowu0162/longmemeval-cleaned) in --data. --split dev is a fixed stratified
fifth of each question type (seed 0) for tuning, held the other four fifths, all everything.

Arms:
  bm25, dense, hybrid   hybrid = reciprocal rank fusion of bm25 and dense: the standard to beat
  assoc                 hybrid's top rounds seed the associative memory, plus entities named in or similar to the
                        question and similar memories; activation spreads over round-memory-entity-category-session
  assoc-only            the same without hybrid's rounds
  oracle, full          QA only: the evidence sessions / the whole haystack, unranked and without budget
  Parameters override graph_memory.associative.Params fields or edge weights: assoc[hops=0;fan0=10;IS_A=0].

Retrieval metrics follow the official eval (recall_any/all@k, ndcg@k, session level from turn level) except that a
round is evidence when its user OR assistant turn has has_answer, so single-session-assistant questions count.
Abstention questions are excluded. budget_*: evidence inside the QA context. Judge: the official evaluate_qa.py
prompts; a reply containing "yes" is correct.

From memory/:
  .venv\\Scripts\\python benchmarks\\longmemeval\\lme.py --run dev-bm25 --arms bm25              # free
  .venv\\Scripts\\python benchmarks\\longmemeval\\lme.py --run dev --arms hybrid,assoc --dry-run  # cost estimate
  .venv\\Scripts\\python benchmarks\\longmemeval\\lme.py --run dev --arms bm25,dense,hybrid,assoc,oracle --qa --max-cost 2
Embeddings and extractions are cached in cache/ for every run; answers and verdicts in runs/<run>/, which resume.
"""
import argparse
import asyncio
import hashlib
import json
import math
import os
import random
import sqlite3
import sys
from array import array
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

from dotenv import dotenv_values

HERE = Path(__file__).resolve().parent
MEMORY = HERE.parents[1]
sys.path.insert(0, str(MEMORY / "src"))
sys.path.insert(0, str(HERE.parent / "rag_qa_arena"))
from arena import BM25, OpenRouter, SpendCap  # noqa: E402
from graph_memory.associative import (PROMPT_VERSION, AssociativeMemory, Extraction, Params, cos,  # noqa: E402
                                      extraction_messages, rounds_of)
from graph_memory.config import Settings  # noqa: E402
from graph_memory.llm import strict_schema  # noqa: E402
from graph_memory.parsing import token_count  # noqa: E402

CACHE = HERE / "cache"
# The memory's minimal reasoning, on the two cheapest endpoints only: its sort=throughput lands on providers charging
# 3.3x GLM's listed price (Fireworks, Together: $0.15/$0.50 per M), and when InferenceNet ($0.045/$0.14) throttles,
# price-sorted fallback lands on Sail Research ($0.60 per M out). Throttled calls are retried instead.
MODEL_OPTIONS = {**Settings.model_fields["openrouter_options"].default,
                 "provider": {"order": ["InferenceNet", "DeepInfra"], "allow_fallbacks": False, "require_parameters": True}}
JUDGE_OPTIONS = {"provider": {"sort": "price"}, "reasoning": {"effort": "low", "exclude": True}}
EMBED_CHARS = 4000
UNRANKED = {"oracle", "full"}
# src/generation/run_generation.py, CoT variant ("Answer step by step"), json history format
READER = ("I will give you several history chats between you and a user. Please answer the question based on the "
          "relevant chat history. Answer the question step by step: first extract all the relevant information, and "
          "then reason over the information to get the answer.\n\n\nHistory Chats:\n\n{}\n\nCurrent Date: {}\n"
          "Question: {}\nAnswer (step by step):")
# src/evaluation/evaluate_qa.py
JUDGE_BASE = ("I will give you a question, a correct answer, and a response from a model. Please answer yes if the response "
              "contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or "
              "contains all the intermediate steps to get the correct answer, you should also answer yes. If the response "
              "only contains a subset of the information required by the answer, answer no. ")
JUDGE = {
    "default": JUDGE_BASE + "\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only.",
    "temporal-reasoning": JUDGE_BASE + "In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only.",
    "knowledge-update": "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only.",
    "single-session-preference": "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only.",
    "abstention": "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only.",
}


class Round(NamedTuple):
    id: str
    session: str
    stamp: str
    turns: list
    answer: bool

    @property
    def text(self):
        return "\n".join(f"{t['role']}: {t['content']}" for t in self.turns)


def day(stamp):
    return datetime.strptime(stamp[:10], "%Y/%m/%d").date()


def sessions_of(q):
    for sid, stamp, turns in zip(q["haystack_session_ids"], q["haystack_dates"], q["haystack_sessions"]):
        yield sid, stamp, turns, [Round(f"{sid}#{i}", sid, stamp, r, any(t.get("has_answer") for t in r))
                                  for i, r in enumerate(rounds_of(turns))]


def session_key(stamp, turns):
    """Relative dates are resolved against the session date, so identical content on another date is another key."""
    return hashlib.sha1((stamp[:10] + json.dumps([[t["role"], t["content"]] for t in turns])).encode()).hexdigest()


def embed_text(r):
    return r.text[:EMBED_CHARS]


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def append(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def split(questions, name):
    if name == "all":
        return questions
    rng, by = random.Random(0), defaultdict(list)
    for q in questions:
        by[q["question_type"]].append(q["question_id"])
    dev = {i for t in sorted(by) for i in rng.sample(by[t], round(len(by[t]) / 5))}
    return [q for q in questions if (q["question_id"] in dev) == (name == "dev")]


class Budget:
    def __init__(self, cap):
        self.cap, self.spent = cap, 0.0

    def charge(self, cost):
        self.spent += cost or 0

    def check(self):
        if self.spent >= self.cap:
            raise SpendCap(f"--max-cost ${self.cap} reached")


class Vectors:
    """Unit vectors in SQLite, keyed by model, dimensions and text; loaded per question as float32 arrays."""
    def __init__(self, client, model, dims, budget):
        CACHE.mkdir(exist_ok=True)
        self.db = sqlite3.connect(CACHE / "embeddings.sqlite")
        self.db.execute("CREATE TABLE IF NOT EXISTS v (k TEXT PRIMARY KEY, v BLOB)")
        self.client, self.model, self.dims, self.budget = client, model, dims, budget

    def key(self, text):
        return hashlib.sha1(f"{self.model}|{self.dims}|{text}".encode()).hexdigest()

    def get(self, text):
        row = self.db.execute("SELECT v FROM v WHERE k=?", (self.key(text),)).fetchone()
        return array("f", row[0]) if row else None

    def missing(self, texts):
        have = set()
        keys = {self.key(t): t for t in texts}
        ks = list(keys)
        for i in range(0, len(ks), 900):
            chunk = ks[i:i+900]
            have |= {k for (k,) in self.db.execute(f"SELECT k FROM v WHERE k IN ({','.join('?' * len(chunk))})", chunk)}
        return [t for k, t in keys.items() if k not in have]

    async def ensure(self, texts, concurrency):
        todo = self.missing([t for t in texts if t.strip()])
        sem = asyncio.Semaphore(concurrency)

        async def batch(chunk):
            async with sem:
                self.budget.check()
                obj = await self.client.post("/embeddings", {"model": self.model, "input": chunk, "dimensions": self.dims})
                self.budget.charge((obj.get("usage") or {}).get("cost"))
                rows = sorted(obj["data"], key=lambda r: r["index"])
                if len(rows) != len(chunk):
                    raise RuntimeError("embedding count differs from input count")
                for text, row in zip(chunk, rows):
                    norm = math.sqrt(sum(x * x for x in row["embedding"])) or 1
                    self.db.execute("INSERT OR REPLACE INTO v VALUES (?, ?)",
                                    (self.key(text), array("f", [x / norm for x in row["embedding"]]).tobytes()))
                self.db.commit()
        await asyncio.gather(*(batch(todo[i:i+64]) for i in range(0, len(todo), 64)))
        return len(todo)


class Extractor:
    def __init__(self, client, model, budget):
        CACHE.mkdir(exist_ok=True)
        self.path = CACHE / f"extract-{model.replace('/', '_')}-p{PROMPT_VERSION}.jsonl"
        self.rows = {r["key"]: r for r in read_jsonl(self.path) if "error" not in r}  # failures are retried
        self.client, self.model, self.budget, self.failed = client, model, budget, 0

    def get(self, key):
        row = self.rows.get(key)
        return Extraction.model_validate(row["extraction"]) if row else None

    async def run(self, jobs, concurrency):
        sem = asyncio.Semaphore(concurrency)
        schema = {"type": "json_schema", "json_schema": {"name": "Extraction", "strict": True,
                                                         "schema": strict_schema(Extraction.model_json_schema())}}

        async def one(key, stamp, turns):
            async with sem:
                self.budget.check()
                messages, cost, error = extraction_messages(rounds_of(turns), day(stamp)), 0.0, None
                for _ in range(2):
                    try:  # a provider failure or invalid JSON leaves an error row, retried by the next run
                        obj = await self.client.post("/chat/completions", {
                            **MODEL_OPTIONS, "model": self.model, "messages": messages, "max_tokens": 6000,
                            "temperature": 0, "response_format": schema})
                        cost += (obj.get("usage") or {}).get("cost") or 0
                        usage = {"provider": obj.get("provider"), **{k: (obj.get("usage") or {}).get(k) for k in (
                            "prompt_tokens", "completion_tokens")}}
                        extraction = Extraction.model_validate_json(obj["choices"][0]["message"].get("content") or "")
                        break
                    except (RuntimeError, ValueError) as exc:
                        extraction, error = None, f"{type(exc).__name__}: {str(exc)[:200]}"
                self.budget.charge(cost)
                row = {"key": key, "cost": cost, **locals().get("usage", {}),
                       "extraction": (extraction or Extraction(memories=[], entities=[])).model_dump()}
                if extraction is None:
                    row["error"], self.failed = error, self.failed + 1
                else:
                    self.rows[key] = row
                append(self.path, row)
                self.done += 1
                if self.done % 200 == 0 or self.done == len(jobs):
                    print(f"extracted {self.done}/{len(jobs)}, failed {self.failed}, spent ${self.budget.spent:.3f}", flush=True)
        self.done = 0
        await asyncio.gather(*(one(k, s, t) for k, (s, t) in jobs.items()))


def arm_spec(arm):
    name, _, rest = arm.partition("[")
    p = Params()
    for kv in filter(None, rest.rstrip("]").split(";")):
        k, v = kv.split("=")
        if hasattr(p, k):
            setattr(p, k, type(getattr(p, k))(v))
        elif k in p.edge_weights or k in p.state_weights:
            (p.edge_weights if k in p.edge_weights else p.state_weights)[k] = float(v)
        else:
            raise SystemExit(f"unknown parameter {k} in {arm}")
    return name, p


def rrf(*orders, k=60):
    score = defaultdict(float)
    for order in orders:
        for rank, i in enumerate(order):
            score[i] += 1 / (k + rank + 1)
    return sorted(score, key=lambda i: -score[i])


def rank(q, arms, vecs, extractor):
    """Rounds of the question's haystack, each ranked arm's order of round IDs, and assoc arms' top activations."""
    sessions = list(sessions_of(q))
    rounds = [r for *_, rs in sessions for r in rs]
    ids = [r.id for r in rounds]
    orders, explain = {"bm25": [ids[i] for i in BM25([r.text for r in rounds]).top(q["question"], len(ids))]}, {}
    if any(a not in ("bm25", *UNRANKED) for a in arms):
        qvec, rv = vecs.get(q["question"]), {r.id: vecs.get(embed_text(r)) for r in rounds}
        orders["dense"] = sorted(ids, key=lambda i: -cos(qvec, rv[i]))
        orders["hybrid"] = rrf(orders["bm25"], orders["dense"])
    memory = None
    for arm in arms:
        name, p = arm_spec(arm)
        if not name.startswith("assoc"):
            continue
        if memory is None:
            memory = AssociativeMemory()
            for sid, stamp, turns, rs in sessions:
                for r in rs:
                    memory.add_round(r.id, r.text, rv[r.id], day(stamp), sid)
                extraction = extractor.get(session_key(stamp, turns)) or Extraction(memories=[], entities=[])
                texts = {m.text for m in extraction.memories} | {e.name for e in extraction.entities} | {
                    c for e in extraction.entities for c in e.is_a} | {n for m in extraction.memories for n in m.entities}
                memory.add_session(sid, day(stamp), [r.id for r in rs], extraction, {t: vecs.get(t) for t in texts})
        ranked = memory.activate(q["question"], qvec, day(q["question_date"]), p,
                                 orders["hybrid"] if name == "assoc" else ())
        best = {}
        for r in ranked:
            best.setdefault(r["round"], r)
        best.pop(None, None)
        orders[arm] = list(best) + [i for i in orders["hybrid"] if i not in best]
        explain[arm] = [{k: r[k] for k in ("memory", "score", "activation_path", "state", "date")}
                        for r in ranked if r["kind"] == "memory"][:10]
    return rounds, {a: orders[a] for a in arms if a not in UNRANKED}, explain


def ndcg(order, gold, k):
    dcg = sum(1 / math.log2(i + 2) for i, r in enumerate(order[:k]) if r in gold)
    return dcg / sum(1 / math.log2(i + 2) for i in range(min(k, len(gold))))


def scores(order, rounds, in_context):
    gold = {r.id for r in rounds if r.answer}
    if not gold:
        return None
    session = {r.id: r.session for r in rounds}
    gold_sessions, session_order = {session[i] for i in gold}, list(dict.fromkeys(session[i] for i in order))
    m = {}
    for k in (5, 10):
        m[f"any@{k}"], m[f"all@{k}"] = float(bool(gold & set(order[:k]))), float(gold <= set(order[:k]))
    m["ndcg@10"] = ndcg(order, gold, 10)
    for k in (3, 5):
        m[f"s_all@{k}"] = float(gold_sessions <= set(session_order[:k]))
    m["budget_all"], m["budget_frac"] = float(gold <= in_context), len(gold & in_context) / len(gold)
    return m


def clean(turn):
    return {"role": turn["role"], "content": turn["content"]}  # never the has_answer labels


def history(chosen):
    """The official json history format: sessions in date order, a session's chosen rounds in order."""
    by_session = defaultdict(list)
    for r in chosen:
        by_session[r.session].append(r)
    parts = []
    for i, rs in enumerate(sorted(by_session.values(), key=lambda rs: rs[0].stamp), 1):
        rs.sort(key=lambda r: int(r.id.rsplit("#", 1)[1]))
        parts.append(f"\n### Session {i}:\nSession Date: {rs[0].stamp}\nSession Content:\n"
                     f"{json.dumps([clean(t) for r in rs for t in r.turns])}\n")
    return "".join(parts)


def within(order, byid, budget, sizes):
    chosen, used = [], 0
    for rid in order:
        if rid not in sizes:
            sizes[rid] = token_count(json.dumps([clean(t) for t in byid[rid].turns]))
        if used + sizes[rid] <= budget:
            chosen.append(byid[rid])
            used += sizes[rid]
    return chosen


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def mcnemar(a, b):
    """Exact two-sided p of paired correctness lists."""
    x, y = sum(1 for i, j in zip(a, b) if i and not j), sum(1 for i, j in zip(a, b) if j and not i)
    n = x + y
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(x, y) + 1)) / 2 ** n)


def table(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header),
                      *("| " + " | ".join(str(c) for c in row) + " |" for row in rows)])


def retrieval_report(rows, arms):
    keys = ["any@5", "all@5", "all@10", "ndcg@10", "s_all@3", "s_all@5", "budget_all", "budget_frac"]
    ranked = [a for a in arms if a not in UNRANKED]
    full = {(r["qid"], r["arm"]): r["budget_all"] for r in rows}
    qids = sorted({r["qid"] for r in rows})

    def versus(a):  # paired exact test on budget_all against hybrid
        if a == "hybrid" or "hybrid" not in ranked:
            return ""
        return f"{mcnemar([full[(i, a)] for i in qids], [full[(i, 'hybrid')] for i in qids]):.3f}"
    out = [table(["arm", "n", *keys, "p budget_all vs hybrid"], [
        [a, len(rs := [r for r in rows if r["arm"] == a]), *(f"{mean([r[k] for r in rs]):.3f}" for k in keys), versus(a)]
        for a in ranked])]
    types = sorted({r["type"] for r in rows})
    out.append("\nbudget_all by question type\n" + table(["arm", *types], [
        [a, *(f"{mean([r['budget_all'] for r in rows if r['arm'] == a and r['type'] == t]):.3f}" for t in types)]
        for a in ranked]))
    return "\n".join(out)


def qa_report(judged, questions, arms):
    qtype = {q["question_id"]: "abstention" if q["question_id"].endswith("_abs") else q["question_type"] for q in questions}
    label = {(r["qid"], r["arm"]): r["label"] for r in judged}
    qids = [q["question_id"] for q in questions if all((q["question_id"], a) in label for a in arms)]
    types = sorted(set(qtype.values()))
    rows = []
    for a in arms:
        right = [label[(i, a)] for i in qids]
        vs = "" if a == "hybrid" or "hybrid" not in arms else f"{mcnemar(right, [label[(i, 'hybrid')] for i in qids]):.3f}"
        rows.append([a, len(qids), f"{mean(right):.3f}", vs,
                     *(f"{mean([label[(i, a)] for i in qids if qtype[i] == t]):.3f}" for t in types)])
    return table(["arm", "n", "accuracy", "p vs hybrid", *types], rows)


async def prices(client):
    response = await client.http.get("/models")
    return {m["id"]: (float(m["pricing"]["prompt"]), float(m["pricing"]["completion"])) for m in response.json()["data"]}


async def main(args):
    env = dotenv_values(MEMORY / ".env")
    key = os.environ.get("ARENA_OPENROUTER_API_KEY") or env.get("ARENA_OPENROUTER_API_KEY") or env["OPENROUTER_API_KEY"]
    client, budget = OpenRouter(key), Budget(args.max_cost)
    run = HERE / "runs" / args.run
    run.mkdir(parents=True, exist_ok=True)
    config = {k: v for k, v in vars(args).items() if k in ("reader", "judge", "budget", "embed_model", "dims",
                                                           "extract_model", "split")}
    config["prompt_version"] = PROMPT_VERSION
    saved = run / "config.json"
    if saved.exists() and (old := json.loads(saved.read_text())) != config:
        raise SystemExit(f"runs/{args.run} was made with {old}; use another --run")
    saved.write_text(json.dumps(config, indent=1))

    questions = split(json.loads((args.data / "longmemeval_s_cleaned.json").read_text(encoding="utf-8")), args.split)
    questions = questions[:args.limit or None]
    arms = args.arms.split(",")
    vecs = Vectors(client, args.embed_model, args.dims, budget)
    extractor = Extractor(client, args.extract_model, budget) if any(a.startswith("assoc") for a in arms) else None
    jobs = {session_key(s, t): (s, t) for q in questions for _, s, t, _ in sessions_of(q)}
    embed = ({q["question"] for q in questions} | {embed_text(r) for q in questions for *_, rs in sessions_of(q) for r in rs}
             if any(a not in ("bm25", *UNRANKED) for a in arms) else set())

    if args.dry_run:
        price = await prices(client)
        todo_embed = vecs.missing(embed)
        todo_extract = [v for k, v in jobs.items() if k not in extractor.rows] if extractor else []
        e_in = sum(len(m["content"]) for s, t in todo_extract for m in extraction_messages(rounds_of(t), day(s))) / 4
        e_out = 0.4 * e_in  # measured: 806 output tokens per 1,993 input
        pe, (pi, po), (ri, ro), (ji, jo) = (price.get(args.embed_model, (1e-8, 0))[0], price[args.extract_model],
                                            price[args.reader], price[args.judge])
        n = len(questions) * len(arms)
        costs = {"embeddings": sum(map(len, todo_embed)) / 4 * pe, "extraction": e_in * pi + e_out * po,
                 "answers": n * ((args.budget + 300) * ri + 600 * ro) if args.qa else 0,
                 "judge": n * (400 * ji + 300 * jo) if args.qa else 0}
        print(f"{len(questions)} questions, {len(todo_embed)} texts to embed, {len(todo_extract)} sessions to extract")
        print({k: round(v, 3) for k, v in costs.items()}, "total", round(sum(costs.values()), 2),
              "(oracle/full answers are longer than --budget)")
        return

    try:
        print(f"embedded {await vecs.ensure(sorted(embed), args.concurrency)} new texts")
        if extractor:
            await extractor.run({k: v for k, v in jobs.items() if k not in extractor.rows}, args.concurrency)
            texts = set()
            for k in jobs:
                if e := extractor.get(k):
                    texts |= {m.text for m in e.memories} | {n for m in e.memories for n in m.entities} | {
                        x.name for x in e.entities} | {c for x in e.entities for c in x.is_a}
            print(f"extraction failures {extractor.failed}; embedded {await vecs.ensure(sorted(texts), args.concurrency)} memory texts")
    except SpendCap as exc:
        raise SystemExit(f"STOPPED: {exc}; spent ${budget.spent:.3f}. Caches keep what was done; rerun to resume.")

    rows, contexts = [], {}
    (run / "activation.jsonl").unlink(missing_ok=True)
    for q in questions:
        rounds, orders, explain = rank(q, arms, vecs, extractor)
        byid, sizes = {r.id: r for r in rounds}, {}
        for arm, order in orders.items():
            chosen = within(order, byid, args.budget, sizes)
            contexts[(q["question_id"], arm)] = history(chosen)
            if not q["question_id"].endswith("_abs") and (m := scores(order, rounds, {r.id for r in chosen})):
                rows.append({"qid": q["question_id"], "type": q["question_type"], "arm": arm, **m})
        if "oracle" in arms:
            contexts[(q["question_id"], "oracle")] = history([r for r in rounds if r.session in q["answer_session_ids"]])
        if "full" in arms:
            contexts[(q["question_id"], "full")] = history(rounds)
        for arm, top in explain.items():
            append(run / "activation.jsonl", {"qid": q["question_id"], "question": q["question"], "arm": arm, "top": top})
    (run / "retrieval.json").write_text(json.dumps(rows))
    report = [f"# {args.run}: {len(questions)} questions ({args.split}), budget {args.budget} tokens\n",
              "## Retrieval\n", retrieval_report(rows, arms)]

    if args.qa:
        answers = {(r["qid"], r["arm"]): r for r in read_jsonl(run / "answers.jsonl")}
        judged = {(r["qid"], r["arm"]): r for r in read_jsonl(run / "judged.jsonl")}
        sem, byq = asyncio.Semaphore(args.concurrency), {q["question_id"]: q for q in questions}

        async def answer(qid, arm):
            async with sem:
                budget.check()
                q = byq[qid]
                prompt = READER.format(contexts[(qid, arm)], q["question_date"], q["question"])
                try:
                    text, cost = await client.chat(args.reader, [{"role": "user", "content": prompt}], 2000, MODEL_OPTIONS)
                except RuntimeError as exc:  # left unanswered; a rerun retries it
                    return print(f"answer {qid} {arm}: {exc}")
                budget.charge(cost)
                answers[(qid, arm)] = row = {"qid": qid, "arm": arm, "hypothesis": text, "cost": cost,
                                             "context_tokens": token_count(prompt)}
                append(run / "answers.jsonl", row)

        async def judge(qid, arm):
            async with sem:
                budget.check()
                q = byq[qid]
                task = "abstention" if qid.endswith("_abs") else q["question_type"]
                prompt = JUDGE.get(task, JUDGE["default"]).format(q["question"], q["answer"], answers[(qid, arm)]["hypothesis"])
                try:
                    text, cost = await client.chat(args.judge, [{"role": "user", "content": prompt}], 1000, JUDGE_OPTIONS)
                except RuntimeError as exc:
                    return print(f"judge {qid} {arm}: {exc}")
                budget.charge(cost)
                judged[(qid, arm)] = row = {"qid": qid, "arm": arm, "label": "yes" in text.lower(), "reply": text[:50], "cost": cost}
                append(run / "judged.jsonl", row)
        try:
            await asyncio.gather(*(answer(i, a) for i in byq for a in arms if (i, a) not in answers))
            await asyncio.gather(*(judge(i, a) for i in byq for a in arms if (i, a) in answers and (i, a) not in judged))
        except SpendCap as exc:
            print(f"STOPPED: {exc}; rerun to resume")
        report += ["\n## QA (judge: " + args.judge + ")\n", qa_report(list(judged.values()), questions, arms)]
        cost = sum(r["cost"] for r in answers.values()) + sum(r["cost"] for r in judged.values())
        report.append(f"\nanswers + judge cost so far: ${cost:.3f}")
    report.append(f"\nspent this invocation: ${budget.spent:.3f}")
    (run / "report.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--data", type=Path, default=Path("C:/code/benchmarks/longmemeval"))
    parser.add_argument("--split", choices=["dev", "held", "all"], default="dev")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--arms", default="bm25,dense,hybrid,assoc")
    parser.add_argument("--qa", action="store_true", help="answer and judge, not only retrieval metrics")
    parser.add_argument("--budget", type=int, default=8000, help="tokens of history the reader gets")
    parser.add_argument("--reader", default="z-ai/glm-5.3-flash")
    parser.add_argument("--judge", default="deepseek/deepseek-v4.1-flash")
    parser.add_argument("--extract-model", default="z-ai/glm-5.3-flash")
    parser.add_argument("--embed-model", default="qwen/qwen3-embedding-8b")
    parser.add_argument("--dims", type=int, default=1024)
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--max-cost", type=float, default=1.0, help="USD this invocation may spend")
    parser.add_argument("--dry-run", action="store_true", help="estimate the cost of what is not cached, spend nothing")
    asyncio.run(main(parser.parse_args()))
