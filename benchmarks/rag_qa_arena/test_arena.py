"""Offline checks of the ported benchmark logic: python benchmarks/rag_qa_arena/test_arena.py"""
import asyncio
import tempfile
from pathlib import Path

import httpx

import json

from arena import (BM25, NO_ANSWER, OpenRouter, Run, SpendCap, context_texts, kg_passages, mine, mixed_rows, parse_correctness, parse_grounding,
                   parse_vote, passages, process_response, reset_refusal, watch_spend_cap, without_citations)

assert process_response("<thinking>short</thinking>\nThe answer.") == "The answer."
assert process_response("Thought: x Answer: The answer.") == "The answer."
assert process_response("FAIL TO GENERATE ANS.") == NO_ANSWER
assert parse_vote("<thinking>Answer 1 lacks detail.</thinking><rating>2</rating>") == 2
assert parse_vote("<rating>0</rating>") == 0
assert parse_vote("no rating at all") == 0
assert without_citations("Retries need idempotency [c:1][c:2]. See [note].", {"c:1", "c:2"}) == "Retries need idempotency. See [note]."
assert without_citations("Both [a, b].", {"a", "b"}) == "Both."
assert [p["text"] for p in passages([{"id": "d", "text": " ".join(map(str, range(150)))}])][1].startswith("100 ")

with tempfile.TemporaryDirectory() as root:
    base = Path(root) / "science" / "test"
    base.mkdir(parents=True)
    docs = ["photosynthesis converts light energy", "light bulbs emit light energy", "cats sleep a lot", "energy of light photons"]
    (base / "collection.tsv").write_text("".join(f"{i}\t{t}\n" for i, t in enumerate(docs)), encoding="utf-8")
    gold, heaps = mine(Path(root), "science", [{"question": "how does light energy work?", "gold": ["0"]}], 2)
    assert gold == {"0": docs[0]}
    assert {pid for _, pid, _ in heaps[0]} == {"1", "3"}  # Gold excluded, unrelated doc never mined.

assert BM25(docs).top("photosynthesis light", 1) == [0]

# --reset-graph never wipes the live graph: its .env URI (any spelling of localhost) or port 7687, bolt's default.
LIVE = "bolt://localhost:7687"
assert reset_refusal("bolt://localhost:7688", LIVE) is None
for uri in ("bolt://127.0.0.1:7687", "neo4j://localhost", "bolt://db.example:7687"):
    assert reset_refusal(uri, LIVE), uri
assert reset_refusal("bolt://127.0.0.1:7690", "bolt://localhost:7690")

# kg_context: the final context split into 100-word passages, first k; recall only counts documents behind them.
context = [{"text": " ".join(["a"] * 150), "provenance": {"sources": [{"metadata": {}}, {"metadata": {"bench_doc_id": "d1"}}]}},
           {"text": "b c", "provenance": {"sources": [{"metadata": {"bench_doc_id": "d2"}}]}},
           {"text": "web page", "provenance": {"sources": [{"uri": "https://x"}]}}]
cut = kg_passages(context_texts(context), 2)
assert [(p["doc"], len(p["text"].split())) for p in cut] == [("d1", 100), ("d1", 50)]
assert [p["doc"] for p in kg_passages(context_texts(context), 5)] == ["d1", "d1", "d2", None]

share, unsupported = parse_grounding('```json\n{"claims": [{"claim": "A", "supported": true}, {"claim": "B", "supported": false},'
                                     ' {"claim": "C", "supported": "yes"}, {"claim": " ", "supported": false}]}\n```')
assert (share, unsupported) == (1 / 3, ["B", "C"])  # Only a literal true counts; blank claims are ignored.
assert parse_grounding('{"claims": []}') == (None, [])
assert parse_correctness('```json\n{"correct": true, "reason": "Agrees with the reference."}\n```') == (True, "Agrees with the reference.")
assert parse_correctness('{"correct": "yes"}') == (False, "")  # only a literal true is correct
try:
    parse_grounding("I cannot judge this.")
    raise AssertionError("malformed verdict must raise so a rerun retries it")
except ValueError:
    pass


async def spend_cap_stops_the_run():
    calls = []

    def respond(status, body):
        def handler(request):
            calls.append(request)
            return httpx.Response(status, json=body)
        return httpx.MockTransport(handler)
    # Baselines and judges: a spend-cap refusal raises at once instead of retrying or logging per item.
    client = OpenRouter("key", respond(403, {"error": {"message": "Key limit exceeded", "code": 403}}))
    try:
        await client.chat("m", [], 5)
        raise AssertionError("spend cap must be fatal")
    except SpendCap:
        assert len(calls) == 1
    # A moderation 403 is an ordinary per-item failure.
    client = OpenRouter("key", respond(403, {"error": {"message": "Input flagged by moderation", "code": 403}}))
    try:
        await client.chat("m", [], 5)
    except SpendCap:
        raise AssertionError("moderation refusal is not a spend cap")
    except RuntimeError:
        pass
    # The graph memory's own client halts the run through a response hook, logging once.
    with tempfile.TemporaryDirectory() as root:
        run = Run(Path(root) / "r", {})
        memory_client = httpx.AsyncClient(transport=respond(402, {"error": {"message": "Insufficient credits"}}))
        watch_spend_cap(memory_client, run)
        for _ in range(2):
            await memory_client.post("https://openrouter.ai/api/v1/chat/completions", json={})
        assert run.halted and run.state["phase"] == "stopped"
        assert sum("STOPPED" in m for _, m in run.state["log"]) == 1
        await memory_client.aclose()

asyncio.run(spend_cap_stops_the_run())

# Answer standards: under "reference" every method gets the reference's own length as its limit, in the v2 sentence.
from arena import CLOSED_V2, V2_LIMIT, bounded, word_limit  # noqa: E402
reference = "one two three four five six seven"
assert word_limit("reference", reference) == 7 and word_limit("50-60", reference) == 60 and word_limit("unbounded", reference) is None
limited = bounded(CLOSED_V2.format(q="why?"), "reference", reference)
assert limited.endswith("Your answer should not be longer than 7 words.") and V2_LIMIT not in limited
assert bounded(CLOSED_V2.format(q="why?"), "50-60", reference).endswith(V2_LIMIT)
try:
    bounded("a v1 prompt without the sentence", "reference", reference)
    raise AssertionError("a template without the v2 length sentence must not silently go unbounded")
except ValueError:
    pass

# A run never mixes cached rows made with another answer model, answer standard or judge into one leaderboard.
with tempfile.TemporaryDirectory() as root:
    old = Path(root) / "old"
    old.mkdir()
    (old / "state.json").write_text(json.dumps({"config": {"answer_model": "deepseek", "judge": "gpt-4-turbo"}}), encoding="utf-8")
    (old / "answers.jsonl").write_text(json.dumps({"method": "bm25_rag", "qid": "q"}) + "\n", encoding="utf-8")
    now = {"answer_model": "glm", "answer_standard": "v2", "judge": "gpt-4-turbo"}
    assert "deepseek -> glm" in mixed_rows(old, now, [])
    assert mixed_rows(old, now, ["bm25_rag"]) is None  # redone rows are discarded, so nothing is mixed
    assert mixed_rows(old, {**now, "answer_model": "deepseek"}, []) is None  # unrecorded keys (answer_standard) never block
    assert mixed_rows(Path(root) / "new", now, []) is None
    # A changed judge only invalidates judge rows: answers can be judged again with --rejudge.
    (old / "state.json").write_text(json.dumps({"config": {"answer_model": "glm", "judge": "glm"}}), encoding="utf-8")
    (old / "judgments.jsonl").write_text(json.dumps({"method": "bm25_rag", "qid": "q"}) + "\n", encoding="utf-8")
    stronger = {"answer_model": "glm", "judge": "deepseek"}
    assert "judgments rows" in mixed_rows(old, stronger, [])
    assert mixed_rows(old, stronger, [], rejudge=True) is None

# Threshold calibration: the lowest threshold whose accepted links are >= 90% correct on 5+ labeled links.
import contextlib, io  # noqa: E401,E402
from replay_mapping import calibrate  # noqa: E402
with tempfile.TemporaryDirectory() as root:
    sheet = Path(root) / "review.csv"
    rows = [(.95, "y")] * 4 + [(.85, "y"), (.82, "y"), (.72, "n"), (.7, "y"), (.62, "n"), (.6, "n"), (.9, "")]
    sheet.write_text("concept,description,domain,proposal,confidence,outcome,accept\n" +
                     "".join(f"c{i},,d,parents X,{c},o,{a}\n" for i, (c, a) in enumerate(rows)), encoding="utf-8")
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        calibrate(sheet)
    assert "10 labeled proposals" in out.getvalue() and "proposed TAXONOMY_MATCH_THRESHOLD=0.75" in out.getvalue(), out.getvalue()
print("ok")
