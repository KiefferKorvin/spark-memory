"""Explicit offline fixtures, not a substitute for semantic inference or Jev."""
import asyncio
import hashlib
import math
import re

from .external import RetrievedSource
from .graph import STOP, words
from .models import IngestRequest, NavigationDecision

DOCUMENTS = [
    {"title": "Jazz piano handbook", "mime_type": "text/markdown", "text": "# Music\n## Piano\n### Rootless voicings\nRootless voicings omit the root. Their purpose is to leave room for a bassist and make smooth voice leading easier.\n### Harmony\nJazz harmony uses thirds and sevenths to establish chord quality. Rootless voicings connect piano and harmony.\n## Rhythm\nSwing divides a beat unevenly and depends on ensemble timing."},
    {"title": "Acoustics note", "mime_type": "application/json", "text": '{"topic":"Acoustics", "observation":"The harmonic series describes integer multiples of a fundamental frequency."}'},
    {"title": "Keyboard observation", "mime_type": "text/plain", "text": "Piano: keeping the wrist relaxed reduces unnecessary tension. This observation does not prescribe medical treatment."},
]
QUESTIONS = [
    "What is the purpose of rootless voicings?",
    "How do piano and harmony connect?",
    "What does acoustics say about the harmonic series?",
    "Why are rootless voicings useful and how should I practice them?",
    "What is the orbital period of an unknown exoplanet?",
]
CONCEPTS = {
    "Music": ([], []), "Piano": (["Music"], ["Harmony"]),
    "Harmony": (["Music"], []), "Jazz": (["Music"], []),
    "Rootless voicings": (["Piano", "Harmony"], ["Jazz"]),
    "Rhythm": (["Music"], []), "Acoustics": (["Science"], []),
    "Practice": (["Education"], ["Piano"]),
}


class DemoModels:
    """Deterministic lexical mock. No model quality claims apply to demo results."""
    async def close(self):
        pass

    async def embed(self, text, query_id=None):
        vector = [0.0] * 64
        for word in words(text):
            digest = hashlib.sha256(word.encode()).digest()
            vector[int.from_bytes(digest[:2], "big") % len(vector)] += 1
        norm = math.sqrt(sum(v*v for v in vector)) or 1
        return [v / norm for v in vector]

    async def structured(self, operation, payload, schema, query_id=None):
        await asyncio.sleep(0)
        if operation == "source_metadata":
            data = {"title": payload.get("title") or payload.get("filename") or "Untitled source", "author": payload.get("author"), "published_at": payload.get("published_at"), "publisher": None, "language": None, "description": payload["text"][:500], "keywords": []}
        elif operation == "understanding":
            text = payload["text"]
            concepts = [{"label": label, "aliases": [], "description": f"Information about {label}.",
                         "broader": broader, "related": related} for label, (broader, related) in CONCEPTS.items()
                        if label.casefold() in (payload["title"] + " " + text).casefold()]
            data = {"document_type": "demo_note", "language": "en", "summary": text[:1400],
                    "routing_summary": f"Explore for {payload['title']}: {text[:800]}", "concepts": concepts}
        elif operation == "resolution":
            data = {"reuse_id": None}
        elif operation == "taxonomy_resolution":
            # Explicit lexical fixture only: no general semantic classification in demo mode.
            broader = {label.casefold() for label in payload["candidate"]["broader"]}
            parents = [n['id'] for n in payload['existing'] if broader & {n['label'].casefold(), *(a.casefold() for a in n['aliases'])}]
            data = {"reuse_id": None, "parent_ids": parents[:6], "confidence": .9 if parents else 0}
        elif operation == "decomposition":
            query = payload["query"]
            if "practice" in query.lower() and " and " in query:
                descriptions = ["purpose of rootless voicings", "practice exercises"]
            else:
                descriptions = [part.strip() for part in re.split(r"\s+and\s+", query) if part.strip()]
            data = {"information_needs": [{"id": f"N{i+1}", "description": d} for i, d in enumerate(descriptions)]}
        elif operation == "relevance":
            need = words(payload["information_need"]["description"]) - STOP - {"about"}
            text = words(payload["text"])
            relevant = len(need & text) >= min(2, len(need)) and bool(need)
            data = {"relevant": relevant, "confidence": 0.8 if relevant else 0.1}
        elif operation == "coverage":
            coverage = []
            for need in payload["information_needs"]:
                ids = [e["id"] for e in payload["evidence"] if need["id"] in e["information_need_ids"]]
                coverage.append({"information_need_id": need["id"], "status": "COVERED" if ids else "MISSING",
                                 "evidence_ids": ids, "missing": "" if ids else need["description"]})
            data = {"coverage": coverage, "overall_status": "SUFFICIENT" if all(c["evidence_ids"] for c in coverage) else "INSUFFICIENT"}
        elif operation == "synthesis":
            evidence = payload["evidence"]
            prefix = "Offline demo: original evidence excerpts.\n"
            if payload["coverage"]["overall_status"] != "SUFFICIENT":
                prefix += "Some information needs remain unanswered.\n"
            data = {"answer": prefix + "\n".join(f"{e['text']} [{e['id']}]" for e in evidence),
                    "evidence_ids": [e["id"] for e in evidence]}
        elif operation == "enrichment":
            text = payload["text"].split(".")[0].strip()
            data = {"assertions": [{"proposition": text, "assertion_type": "claim", "confidence": 0.7,
                                    "supporting_quote": text, "contradicts_ids": []}]}
        else:
            raise ValueError(operation)
        return schema.model_validate(data)

    async def navigate(self, payload, query_id):
        await asyncio.sleep(0.002)
        candidates = payload["candidate_children"]
        decisions = [{"node_id": c["node_id"], "action": ("SELECT" if c["retrievable"] else "EXPAND")
                      if c["score"] >= 0.16 else "PRUNE"} for c in candidates]
        return NavigationDecision(decisions=decisions, current_node_action="CONTINUE")


class OnlineDemoModels(DemoModels):
    """Real semantic inference and synthesis; explicit lexical demo retrieval/embeddings."""
    def __init__(self, online):
        self.online = online

    async def structured(self, operation, payload, schema, query_id=None):
        return await self.online.structured(operation, payload, schema, query_id)

    async def close(self):
        await self.online.close()


class DemoExternal:
    async def search(self, query, limit):
        if "practice" not in query.lower() or limit <= 0:
            return []
        return [RetrievedSource(title="Demo practice source", url="https://example.org/demo/practice",
            text="Practice exercises for rootless voicings: alternate third-seventh and seventh-third shapes through ii–V–I progressions. Practice slowly in several keys, with a metronome, then add a bass recording.",
            metadata={"synthetic": True})]


async def seed(memory):
    return [await memory.ingest(IngestRequest(**document)) for document in DOCUMENTS]


async def main():
    from .config import Settings
    from .service import Memory
    memory = Memory.from_settings(Settings(memory_mode="demo", _env_file=None))
    await memory.initialize()
    try:
        await seed(memory)
        result = await memory.query(QUESTIONS[3])
        print(result["answer"])
        print("Coverage:", result["coverage"]["overall_status"])
        print("Events:", len(await memory.events(result["query_id"])))
    finally:
        await memory.close()


if __name__ == "__main__":
    asyncio.run(main())
