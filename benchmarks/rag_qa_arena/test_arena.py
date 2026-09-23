"""Offline checks of the ported benchmark logic: python benchmarks/rag_qa_arena/test_arena.py"""
import tempfile
from pathlib import Path

from arena import BM25, NO_ANSWER, mine, parse_vote, passages, process_response, without_citations

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
print("ok")
