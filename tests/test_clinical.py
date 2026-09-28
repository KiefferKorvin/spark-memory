import hashlib
import json
import math
import sqlite3
from datetime import date

import pytest
from fastapi.testclient import TestClient

from graph_memory.associative import norm
from graph_memory.clinical import Clinical, ClinicalSettings, Terminology
from graph_memory.clinical_api import create_app
from graph_memory.graph import InMemoryGraph

IS_A = "116680003"
# Real SNOMED CT codes (FR edition 2026-06), with depths below the root and is-a parents.
CONCEPTS = {
    "56819008": ("Endocarditis (disorder)", 5, []),
    "233850007": ("Infective endocarditis (disorder)", 6, ["56819008"]),
    "459067002": ("Infective endocarditis of cardiac valve (disorder)", 7, ["233850007"]),
    "459152004": ("Infective endocarditis of mitral valve (disorder)", 8, ["459067002"]),
    "3092008": ("Staphylococcus aureus (organism)", 6, []),
    "115329001": ("Methicillin resistant Staphylococcus aureus (organism)", 7, ["3092008"]),
    "372735009": ("Vancomycin (substance)", 6, []),
    "406439009": ("Daptomycin (substance)", 6, []),
    "387159009": ("Rifampicin (substance)", 6, []),
    "387321007": ("Gentamicin (substance)", 6, []),
    "25510005": ("Cardiac valve prosthesis (physical object)", 5, []),
    "5758002": ("Bacteremia (finding)", 5, []),
    "709044004": ("Chronic kidney disease (disorder)", 6, []),
    "433144002": ("Chronic kidney disease stage 3 (disorder)", 7, ["709044004"]),
    "609328004": ("Allergic disposition (finding)", 3, []),
    "91936005": ("Allergy to penicillin (finding)", 5, ["609328004"]),
}
FRENCH = {"233850007": "endocardite infectieuse"}


def terminology(path):
    db = sqlite3.connect(path)
    db.executescript("""CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE concept(id TEXT PRIMARY KEY, fsn TEXT, tag TEXT, depth INTEGER);
        CREATE TABLE term(concept TEXT, lang TEXT, fsn INTEGER, term TEXT, norm TEXT);
        CREATE TABLE relation(source TEXT, type TEXT, target TEXT);
        CREATE VIRTUAL TABLE term_fts USING fts5(norm, concept UNINDEXED);""")
    db.execute("INSERT INTO meta VALUES ('manifest', ?)", (json.dumps({"source_file": "fixture"}),))
    for code, (fsn, depth, parents) in CONCEPTS.items():
        name, tag = fsn.rsplit(" (", 1)
        db.execute("INSERT INTO concept VALUES (?,?,?,?)", (code, fsn, tag[:-1], depth))
        for lang, term, is_fsn in [("en", name, 1)] + ([("fr", FRENCH[code], 0)] if code in FRENCH else []):
            db.execute("INSERT INTO term VALUES (?,?,?,?,?)", (code, lang, is_fsn, term, norm(term)))
            db.execute("INSERT INTO term_fts VALUES (?,?)", (norm(term), code))
        for parent in parents:
            db.execute("INSERT INTO relation VALUES (?,?,?)", (code, IS_A, parent))
    db.commit()
    db.close()
    return Terminology(path)


def term(t, tag):
    return {"term": t, "tag": tag}


MRSA, IE = term("Methicillin resistant Staphylococcus aureus", "organism"), term("Infective endocarditis", "disorder")
VANCO, DAPTO = term("Vancomycin", "substance"), term("Daptomycin", "substance")
GUIDELINE = {"recommendations": [
    {"text": "Native-valve MRSA endocarditis: vancomycin or daptomycin for 6 weeks.", "conditions": [MRSA, IE, term("native heart valve", "body structure")],
     "exclusions": [term("Cardiac valve prosthesis", "physical object")], "actions": [VANCO, DAPTO], "strength": "Class I"},
    {"text": "Prosthetic-valve MRSA endocarditis: vancomycin with rifampicin for 6 weeks and gentamicin for 2 weeks.",
     "conditions": [MRSA, IE, term("Cardiac valve prosthesis", "physical object")], "exclusions": [],
     "actions": [VANCO, term("Rifampicin", "substance"), term("Gentamicin", "substance")], "strength": "Class I"},
    {"text": "Uncomplicated MRSA bacteremia: vancomycin or daptomycin for at least 2 weeks.", "conditions": [MRSA, term("Bacteremia", "finding")],
     "exclusions": [IE], "actions": [VANCO, DAPTO], "strength": "Class IIa"},
    {"text": "Adjust the vancomycin dose to renal function and monitor levels.", "conditions": [term("Chronic kidney disease", "disorder")],
     "exclusions": [], "actions": [VANCO], "strength": ""},
]}


def finding(text, concepts, subject="patient", state="present"):
    return {"text": text, "state": state, "subject": subject, "date": "", "concepts": concepts, "confidence": 0.95}


NOTE = {"findings": [
    finding("Infective endocarditis of the native mitral valve (TEE vegetation)", [term("Infective endocarditis of mitral valve", "disorder")]),
    finding("Blood cultures growing Staphylococcus aureus (2/2 sets)", [term("Staphylococcus aureus", "organism")]),
    finding("Chronic kidney disease stage 3, creatinine 1.8 mg/dL", [term("Chronic kidney disease stage 3", "disorder")]),
    finding("Penicillin allergy (rash)", [term("Allergy to penicillin", "finding")]),
    finding("Mother had endocarditis", [term("Endocarditis", "disorder")], subject="family"),
]}
LAB = [
    {"resourceType": "Observation", "effectiveDateTime": "2026-09-20", "code": {"coding": [
        {"system": "http://loinc.org", "code": "600-7", "display": "Bacteria identified in Blood by Culture"}]},
     "valueCodeableConcept": {"coding": [{"system": "http://snomed.info/sct", "code": "115329001",
                                          "display": "Methicillin resistant Staphylococcus aureus"}]}},
    {"resourceType": "Observation", "effectiveDateTime": "2026-09-20", "code": {"coding": [
        {"system": "http://loinc.org", "code": "524-9", "display": "Vancomycin [Susceptibility] by Minimum inhibitory concentration (MIC)"}]},
     "valueQuantity": {"value": 1, "unit": "ug/mL"}, "interpretation": [{"coding": [{"code": "S"}]}]},
    {"resourceType": "Observation", "effectiveDateTime": "2026-09-20", "code": {"coding": [
        {"system": "http://loinc.org", "code": "2160-0", "display": "Creatinine [Mass/volume] in Serum or Plasma"}]},
     "valueQuantity": {"value": 1.8, "unit": "mg/dL"}},
]


class FakeModels:
    def __init__(self):
        self.calls = []
        self.replies = {"GuidelineExtraction": GUIDELINE, "NoteExtraction": NOTE,
                        "ClinicalAnswer": {"answer": "Start vancomycin or daptomycin for 6 weeks [G1][P1]; adjust to renal function [H2].", "cited": []}}

    async def structured(self, model, system, user, schema):
        self.calls.append(schema.__name__)
        return schema.model_validate(self.replies[schema.__name__])

    async def embed(self, texts):  # bag of hashed words: deterministic, and similar texts are close
        vectors = []
        for t in texts:
            v = [0.0] * 64
            for w in norm(t).split():
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1
            length = math.sqrt(sum(x * x for x in v)) or 1
            vectors.append([x / length for x in v])
        return vectors

    async def close(self):
        pass


def settings(**overrides):
    return ClinicalSettings(_env_file=None, memory_api_token="secret", model_api_key="k", **overrides)


@pytest.fixture
def clinical(tmp_path):
    return Clinical(settings(), InMemoryGraph(), FakeModels(), terminology(str(tmp_path / "snomed.sqlite")))


async def test_endocarditis_encounter(clinical):
    await clinical.add_guideline("Management of endocarditis.", "Endocarditis guideline", "ESC", "2023")
    await clinical.add_note("p1", "Admission note ...", date(2026, 9, 18), encounter="e1")
    lab = await clinical.add_fhir("p1", LAB, encounter="e1")
    assert clinical.models.calls == ["GuidelineExtraction", "NoteExtraction"]  # FHIR needs no model call
    assert {c["key"] for m in lab["memories"] for c in m["concepts"]} >= {"sct:115329001", "sct:372735009", "loinc:2160-0"}

    result = await clinical.ask("p1", "What treatment should we start?", encounter="e1", day=date(2026, 9, 21))
    recs = [r["text"] for r in result["recommendations"]]
    assert recs.index(GUIDELINE["recommendations"][0]["text"]) < recs.index(GUIDELINE["recommendations"][1]["text"])
    assert GUIDELINE["recommendations"][2]["text"] not in recs          # excluded: the patient has endocarditis
    assert GUIDELINE["recommendations"][3]["text"] in recs              # renal dosing applies: CKD 3 is a CKD
    native = next(r for r in result["recommendations"] if r["text"].startswith("Native"))
    assert set(native["met"]) == {MRSA["term"], IE["term"]} and native["unverified"] == ["native heart valve"]

    header = result["header"]
    assert [m["text"] for m in header["allergies"]] == ["Penicillin allergy (rash)"]
    assert "Mother had endocarditis" not in [m["text"] for m in header["problems"]]
    assert [m["text"] for m in header["results"]] == ["Creatinine [Mass/volume] in Serum or Plasma: 1.8 mg/dL"]
    assert result["cited"] and clinical.models.calls[-1] == "ClinicalAnswer"

    records = await clinical.repo.records("clinical", "patient:p1|")
    ai = [m for r in records for m in r["memories"] if m["source"] == "ai"]
    assert len(ai) == 1 and ai[0]["state"] == "hypothetical"            # the answer is never stored as a fact
    header = clinical.header(records)
    assert all(m["source"] != "ai" for rows in header.values() for m in rows)


async def test_french_question_names_the_concept(clinical):
    await clinical.add_guideline("Management of endocarditis.", "Endocarditis guideline")
    await clinical.add_note("p2", "Note", date(2026, 9, 18))
    result = await clinical.ask("p2", "Quel traitement pour l'endocardite infectieuse ?", answer=False, day=date(2026, 9, 21))
    native = next(r for r in result["recommendations"] if r["text"].startswith("Native"))
    assert native["activation_path"][0] == "Infective endocarditis"   # seeded by the French synonym
    assert native["not_met"] == [MRSA["term"]]                          # this patient has S. aureus, no MRSA result yet
    assert result["recommendations"][0]["text"].startswith("Adjust")   # so the CKD dosing advice fits best


def test_governance_guard():
    with pytest.raises(ValueError, match="MEMORY_API_TOKEN"):
        ClinicalSettings(_env_file=None, model_api_key="k").check()
    with pytest.raises(ValueError, match="openrouter.ai"):
        settings(clinical_data_mode="real", clinical_compliant_hosts="openrouter.ai").check()
    with pytest.raises(ValueError, match="refuses"):
        settings(clinical_data_mode="real", model_base_url="https://llm.hospital.example/v1").check()
    settings(clinical_data_mode="real", model_base_url="https://llm.hospital.example/v1",
             clinical_compliant_hosts="llm.hospital.example").check()


def test_api_requires_the_token(clinical):
    with TestClient(create_app(clinical)) as client:
        assert client.get("/clinical/patients/p1/header").status_code == 401
        ok = client.get("/clinical/patients/p1/header", headers={"Authorization": "Bearer secret"})
        assert ok.status_code == 200 and ok.json()["allergies"] == []
        assert client.get("/clinical/patients/..%2F/header", headers={"Authorization": "Bearer secret"}).status_code in (404, 422)
