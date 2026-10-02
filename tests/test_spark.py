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
from graph_memory.spark import ForgetRequest, RecallRequest, SessionRequest, SparkMemory, settle

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
EPISODE = {"title": "Paradiddles trop rapides à 90 BPM",
           "text": "L'apprenant a travaillé les paradiddles et les a trouvés trop rapides à 90 BPM. Le coach a conseillé de redescendre à 70 BPM."}
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
        assert operation in ("spark_extract", "spark_episode") and "Trop dur à 90" in payload["session"]
        return schema.model_validate({**EXTRACTION, **({"episode": EPISODE} if operation == "spark_episode" else {})})

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


def test_corrupted_extractions_are_dropped():
    from graph_memory.spark import corrupted
    session = "Leçon « Binaire ↔ ternaire » : shuffle"
    assert corrupted('Le [date] le suit - ~~A~ret">ass.: "setFiresVial(YoctoTestRunner.runAll())"*flag]]', session)
    assert corrupted("[s3] L'utilisateur étudie les marques diacritiques grecques chez 古希腊语", session)
    assert not corrupted("L'apprenant confond binaire et ternaire dans le shuffle (croches inégales).", session)
    assert not corrupted("学生觉得这个很难", "学生说：这个很难")  # a script the session itself uses is fine
    french = "L'apprenant a trouvé la leçon de batterie difficile et le coach a conseillé de ralentir le tempo de la grosse caisse."
    assert corrupted("Le pied droit (grosse caisse)... ", french)                       # cut off
    assert corrupted("When asked how the lesson went, the learner answered that it was fine.", french)
    assert not corrupted("L'apprenant a trouvé la leçon difficile.", french)


async def test_a_session_leaves_an_episode_recalled_with_its_facts_and_listed_newest_first():
    memory = SparkMemory(None, InMemoryGraph(), Models())
    stored = await memory.remember(session(date="2026-09-20"))
    assert stored["episode"] == EPISODE and stored["date"] == "2026-09-20"
    found = await memory.recall(RecallRequest(scope="user:7", text="on reprend les paradiddles ?"))
    assert found["episodes"][0]["title"] == EPISODE["title"] and found["episodes"][0]["date"] == "2026-09-20"
    assert all(m["text"] != EPISODE["text"] for m in found["memories"])             # an episode is not a fact
    await memory.remember(SessionRequest(scope="user:7", session="chat-1-9", title="Conversation", kind="chat", date="2026-09-25",
                                         turns=TURNS, episode=False))                # a Drifts feed has no episode
    listed = await memory.listing("user:7")
    assert [e["session"] for e in listed["episodes"]] == ["Rudiments simples"] and len(listed["memories"]) == 3


async def test_an_unrelated_message_recalls_nothing_but_a_named_concept_or_a_close_meaning_does():
    memory = SparkMemory(None, InMemoryGraph(), Models())
    await memory.remember(session())
    for text in ("recette de crêpes", "météo demain"):  # the fake 64-bucket embedding collides on short words like "salut"
        found = await memory.recall(RecallRequest(scope="user:7", text=text))
        assert found["memories"] == [] and found["episodes"] == []
    assert (await memory.recall(RecallRequest(scope="user:7", text="ghost notes")))["memories"]


def item(i, text, date, vec, group=("batterie", False), score=1.0):
    return dict(id=str(i), text=text, date=date, vec=vec, group=group, score=score)


def test_what_was_said_again_merges_and_what_changed_keeps_the_old_value_behind_it():
    near, other = vector("entraîne minutes jour"), vector("piano clavier")
    twenty, again, forty_five = (item(1, "20 minutes par jour", "2026-09-20", near), item(2, "20 minutes par jour pour la batterie", "2026-09-25", near),
                                 item(3, "maintenant 45 minutes par jour", "2026-10-01", near, score=.5))
    [merged] = settle([twenty, again, forty_five])
    assert merged["text"] == "maintenant 45 minutes par jour" and merged["before"] == ["20 minutes par jour pour la batterie"] and merged["seen"] == 1
    assert merged["ids"] == ["3", "2", "1"] and merged["score"] == 1.0
    [repeat] = settle([twenty, again])
    assert repeat["text"] == again["text"] and repeat["seen"] == 2 and repeat["before"] == []
    assert len(settle([twenty, item(4, "20 minutes par jour", "2026-10-02", near, group=("piano", False))])) == 2  # another subject
    assert len(settle([twenty, item(5, "clavier piano", "2026-10-02", other)])) == 2


async def test_forgetting_removes_a_memory_its_entities_and_an_episode():
    memory = SparkMemory(None, InMemoryGraph(), Models())
    await memory.remember(session())
    listed = await memory.listing("user:7")
    ghost = next(m for m in listed["memories"] if "ghost notes" in m["text"])
    assert (await memory.forget_memories("user:7", [ghost["id"], listed["episodes"][0]["id"]])) == {"forgotten": 2}
    after = await memory.listing("user:7")
    assert all("ghost notes" not in m["text"] for m in after["memories"]) and after["episodes"] == []
    assert "ghost notes" not in (await memory.recall(RecallRequest(scope="user:7", text="ghost notes")))["cues"]


async def test_a_deep_recall_goes_further_than_the_automatic_one():
    memory = SparkMemory(None, InMemoryGraph(), Models())
    await memory.remember(session())
    plain = await memory.recall(RecallRequest(scope="user:7", text="rudiment de batterie"))
    deep = await memory.recall(RecallRequest(scope="user:7", text="rudiment de batterie", deep=True))
    assert len(deep["memories"]) >= len(plain["memories"])


def test_the_api_lists_forgets_and_serves_episodes():
    memory = Memory(Settings(memory_mode="demo", _env_file=None), InMemoryGraph(), Models())
    with TestClient(create_app(memory)) as client:
        client.post("/memory/spark/sessions", json=session(subject="Batterie").model_dump(mode="json"))
        assert client.get("/memory/spark/episodes", params={"scope": "shared"}).status_code == 422
        episodes = client.get("/memory/spark/episodes", params={"scope": "user:7", "subject": "Batterie"}).json()["episodes"]
        assert [e["title"] for e in episodes] == [EPISODE["title"]]
        assert client.get("/memory/spark/episodes", params={"scope": "user:7", "subject": "Piano"}).json()["episodes"] == []
        listed = client.get("/memory/spark/memories", params={"scope": "user:7"}).json()
        done = client.post("/memory/spark/forget", json={"scope": "user:7", "ids": [m["id"] for m in listed["memories"]]}).json()
        assert done == {"forgotten": 3}
        assert client.get("/memory/spark/memories", params={"scope": "user:7"}).json()["memories"] == []
