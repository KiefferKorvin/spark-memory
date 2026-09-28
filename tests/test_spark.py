import asyncio
import hashlib
import math

from fastapi.testclient import TestClient

from graph_memory import spark
from graph_memory.api import create_app
from graph_memory.associative import norm
from graph_memory.config import Settings
from graph_memory.graph import InMemoryGraph
from graph_memory.service import Memory
from graph_memory.spark import RecallRequest, SessionRequest, SparkMemory

EXTRACTION = {
    "memories": [
        {"text": "L'apprenant trouve les paradiddles à 90 BPM trop difficiles", "state": "stated", "round": 1, "date": "",
         "entities": ["paradiddle"], "confidence": 0.9},
        {"text": "L'apprenant veut travailler les ghost notes la semaine prochaine", "state": "intended", "round": 1,
         "date": "", "entities": ["ghost notes"], "confidence": 0.9},
        {"text": "Le coach a conseillé de redescendre à 70 BPM", "state": "assistant_said", "round": 1, "date": "",
         "entities": ["paradiddle"], "confidence": 0.9},
    ],
    "entities": [{"name": "paradiddle", "is_a": ["rudiment de batterie"]},
                 {"name": "ghost notes", "is_a": ["technique de batterie"]}]}
TURNS = [{"role": "assistant", "content": "Leçon « Rudiments simples » : paradiddles et ghost notes."},
         {"role": "user", "content": "Trop dur à 90, et je veux bosser les ghost notes la semaine prochaine."},
         {"role": "assistant", "content": "Redescends à 70 BPM."}]


def vector(text):
    v = [0.0] * 64
    for w in norm(text).split():
        v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1
    length = math.sqrt(sum(x * x for x in v)) or 1
    return [x / length for x in v]


class Models:
    def __init__(self, slow=False):
        self.slow, self.calls = slow, []

    async def structured(self, operation, payload, schema, query_id=None):
        self.calls.append(operation)
        assert operation == "spark_extract" and "Trop dur à 90" in payload["session"]
        return schema.model_validate(EXTRACTION)

    async def embed(self, text, query_id=None):
        if self.slow:
            await asyncio.sleep(5)
        return vector(text)

    async def embed_batch(self, texts, query_id=None):
        return [vector(t) for t in texts]

    async def close(self):
        pass


def session(**kw):
    return SessionRequest(scope="user:7", session="lesson-3-12", title="Rudiments simples", kind="lesson", turns=TURNS, **kw)


async def test_a_lesson_is_recalled_by_what_the_learner_mentions():
    memory = SparkMemory(None, InMemoryGraph(), Models())
    assert (await memory.remember(session()))["memories"] == 3
    found = await memory.recall(RecallRequest(scope="user:7", text="On reprend les paradiddles aujourd'hui ?"))
    top = found["memories"][0]
    assert "paradiddle" in top["text"] and "paradiddle" in found["cues"]          # named in the message
    assert top["session"] == "Rudiments simples" and top["kind"] == "lesson"
    assert (await memory.recall(RecallRequest(scope="user:8", text="paradiddles")))["memories"] == []  # scopes are private


async def test_a_slow_embedding_never_stalls_recall_and_the_encounter_carries_cues(monkeypatch):
    monkeypatch.setattr(spark, "EMBED_SECONDS", 0.05)
    memory = SparkMemory(None, InMemoryGraph(), Models(slow=True))
    await memory.remember(session(encounter="goal:2"))
    vague = RecallRequest(scope="user:7", text="Et maintenant, on fait quoi ?", encounter="goal:2")
    carried = await memory.recall(vague)                                              # no names, no vector: the lesson just ended
    assert {m["text"] for m in carried["memories"]} >= {EXTRACTION["memories"][0]["text"], EXTRACTION["memories"][1]["text"]}
    assert (await memory.recall(RecallRequest(scope="user:7", text=vague.text)))["memories"] == []  # another conversation


def test_api_requires_a_private_scope_and_erasing_the_scope_forgets(tmp_path):
    memory = Memory(Settings(memory_mode="demo", _env_file=None), InMemoryGraph(), Models())
    with TestClient(create_app(memory)) as client:
        body = session().model_dump(mode="json")
        assert client.post("/memory/spark/sessions", json={**body, "scope": "shared"}).status_code == 422
        assert client.post("/memory/spark/sessions", json=body).json()["memories"] == 3
        recall = {"scope": "user:7", "text": "paradiddle"}
        assert client.post("/memory/spark/recall", json=recall).json()["memories"]
        assert client.delete("/memory/scopes/user:7").status_code == 200
        assert client.post("/memory/spark/recall", json=recall).json()["memories"] == []


async def test_recall_for_a_subject_damps_the_others():
    memory = SparkMemory(None, InMemoryGraph(), Models())
    await memory.remember(session(subject="Batterie"))
    await memory.remember(SessionRequest(scope="user:7", session="lesson-9-1", title="Rudiments au clavier", kind="lesson",
                                         subject="Piano", turns=TURNS))
    found = await memory.recall(RecallRequest(scope="user:7", text="paradiddle", subject="Batterie", limit=6))
    rank = {(m["session"], m["text"]): i for i, m in enumerate(found["memories"])}
    hard = EXTRACTION["memories"][0]["text"]  # the same memory, stored under both subjects
    assert rank[("Rudiments simples", hard)] == 0 and rank[("Rudiments simples", hard)] < rank[("Rudiments au clavier", hard)]
