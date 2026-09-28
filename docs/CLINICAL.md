# Clinical SPARK

SPARK (Semantic Priming & Associative Retrieval Kernel, `graph_memory/associative.py`) recalls a memory because the
current input activates what it is linked to. The clinical deployment applies it to patient records: notes, coded
results and clinical guidelines, linked through SNOMED CT. It runs apart from PAKT's memory, with its own database,
token, model hosts and settings, on the same core code (`clinical.py`, `clinical_api.py`).

## Deployment

| | PAKT memory | Clinical SPARK |
|---|---|---|
| Compose | `compose.yaml`, project `memory` | `compose.clinical.yaml`, project `memory-clinical` |
| Settings | `.env` | `.env.clinical` (template `.env.clinical.example`) |
| API | `graph_memory.api`, 127.0.0.1:8010 | `graph_memory.clinical_api`, 127.0.0.1:8020 |
| Neo4j | volume `memory-data`, 7474/7687 | volume `clinical-data`, 127.0.0.1:7476/7690 |
| Online search, recipes, images, episodes | yes | none of these endpoints exist |

```
python -m graph_memory.clinical build-terminology <SNOMED RF2 snapshot ZIP> <snomed.sqlite>   # once per release, ~2 min
docker compose --env-file .env.clinical -f compose.clinical.yaml up -d --build
```

The terminology SQLite (~500 MB for the French edition, June 2026: 381,253 active concepts, 1.34M inferred
relationships, English and French terms) is mounted read-only. It is licensed SNOMED CT content: keep it out of the
repository and within the licence's territory.

## Governance

- Every route needs `Authorization: Bearer <MEMORY_API_TOKEN>`; the app refuses to start without a token.
- `CLINICAL_DATA_MODE=synthetic` (default) is for fictional or de-identified data. `real` refuses to start unless
  every model host (chat and embeddings) is listed in `CLINICAL_COMPLIANT_HOSTS`, and never accepts `openrouter.ai`:
  identifiable health data may only reach hosts under a data processing agreement for it (for MIMIC-derived data,
  PhysioNet names Azure OpenAI, Amazon Bedrock, Google Vertex AI and Anthropic).
- A model's output is never a fact. Answers are stored as `hypothetical` memories with source `ai` (weight 0.3), and are
  left out of the header and of recommendation applicability. Only the clinician's own record (a note, a coded result or
  order) establishes a fact.
- Erasure: `DELETE /clinical/patients/{id}` erases the patient's records and working memory; a document or guideline
  can be deleted alone.
- Not built yet, and needed before real patient data: an access audit log, per-user access control (one token opens
  every patient), encryption at rest (Neo4j Community has none), backups and a retention policy.

## Write paths

| Input | Model calls | Becomes |
|---|---|---|
| Note (`POST /clinical/patients/{id}/notes`) | 1 | findings: a self-contained statement, `state` (present, absent, possible, conditional, hypothetical), `subject` (patient, family, other), date, confidence, concepts |
| FHIR R4 (`POST .../fhir`): Observation, Condition, AllergyIntolerance, MedicationStatement | 0 | one memory per resource from its codes, value, interpretation, status and date |
| Guideline (`POST /clinical/guidelines`, text or PDF/DOCX) | 1 per ~2,000-token section | recommendations with `conditions`, `exclusions`, `actions` and `strength`, in the shared scope |

Concepts are resolved locally against SNOMED CT: an exact accent-folded synonym in English or French, else a full-text
match sharing at least 60% of the words. The semantic tag chooses among exact synonyms and filters fuzzy matches (disorder
and finding are compatible, substance and product too); an exact synonym wins over a disagreeing tag, since the
extractor's tag is the weaker evidence. An unresolved term stays free text (`txt:`) and is resolved again whenever the
graph loads, so a newer release or better matching applies without re-ingesting. FHIR SNOMED codings are used as they are; LOINC codes become
`loinc:` concepts, and a display like "Vancomycin [Susceptibility] by ..." also links the drug. Each document is one
record (category `clinical`, key `<scope>|doc|<id>`) with its memories and float32 vectors.

## Read path (`POST /clinical/patients/{id}/ask`)

1. **Graph.** The patient's documents plus the guidelines, each concept linked to its SNOMED neighbourhood: is-a
   ancestors up to `SNOMED_LEVELS` (3) and its own attributes, leaving out concepts within `SNOMED_MIN_DEPTH` (4)
   of the root. Cached per patient until a write.
2. **Seeds.** Concepts the question names (any English or French synonym, up to six words), the concepts and memories
   closest to its embedding, and the encounter's working memory: the concepts its documents and earlier questions
   activated (`encounter`, kept `ENCOUNTER_HOURS`, carried at 0.8).
3. **Spreading.** Three hops, decay 0.6 per hop, each seed separately (`combine="distinct"`): a node adds its best
   path from each seed, so separate cues (endocarditis, MRSA) converge on the recommendation that needs both, while
   echoes through shared nodes (vancomycin, daptomycin) do not inflate the passages that name the most drugs.
4. **Scoring.** activation × temporal × confidence × provenance × relevance, then for recommendations × applicability.

| Edge | Weight | | State | Weight | | Source | Weight |
|---|---|---|---|---|---|---|---|
| mentions | 1.0 | | present | 1.0 | | lab, structured, guideline | 1.0 |
| causative agent, has active ingredient | 0.9 | | absent | 0.8 | | note | 0.95 |
| is a, finding site, interprets | 0.5 | | possible, conditional | 0.7 | | ai | 0.3 |
| same document | 0.2 | | hypothetical | 0.5 | | | |
| has disposition | 0.2 | | | | | | |

**Applicability.** A recommendation is dropped when one of its exclusions holds for the patient. Otherwise its score is
multiplied, per condition, by 1 when the patient has the concept or a descendant (SNOMED subsumption: an "infective
endocarditis of mitral valve" meets "infective endocarditis", MRSA meets *S. aureus*), 0.9 when a free-text condition's
words all appear in one of the patient's findings, 0.6 when it is a working diagnosis (stated as possible), 0.35 when a
free-text condition cannot be checked, 0.2 when the record does not show it and 0.05 when it is recorded as absent.
Unverifiable stays below a working diagnosis, so a recommendation naming only things the record cannot check never
outranks one the patient demonstrably fits. The patient's facts come from their own record (family history and AI
suggestions excluded).

**Header.** Always returned and given to the answer, never left to association: allergies, problems (working diagnoses
marked possible), current medications and the latest `HEADER_CODES` results, LOINC or SNOMED keys from coded results or
notes (creatinine, eGFR, weight, potassium, body weight and GFR by default).

**Answer.** The model receives the header [H], the patient memories [P] and the best-fitting recommendations [G] with
what each condition's check found, and must cite them, name what the record cannot confirm and raise the header's
safety checks. `answer: false` returns the working memory without a model call.

## API

| Route | Use |
|---|---|
| `POST /clinical/patients/{id}/notes` `{text, date, title?, author?, encounter?}` | Ingest a note |
| `POST /clinical/patients/{id}/fhir` `{resources: [...], encounter?}` | Ingest coded resources (≤ 500) |
| `GET /clinical/patients/{id}/header` | The header |
| `POST /clinical/patients/{id}/ask` `{question, encounter?, answer?, date?}` | Header, memories, recommendations and answer |
| `DELETE /clinical/patients/{id}`, `DELETE .../documents/{doc}` | Erase |
| `POST /clinical/guidelines` `{title, issuer?, version?, published?, text? or content_base64 + mime}` | Ingest a guideline |
| `GET /clinical/guidelines`, `DELETE /clinical/guidelines/{doc}` | List, delete |

## Evidence so far

- `tests/test_clinical.py`: the endocarditis case (note, FHIR culture and susceptibility, four recommendations) ranks
  the native-valve MRSA recommendation above the prosthetic-valve one, drops the uncomplicated-bacteremia one (the
  patient has endocarditis), keeps the renal dosing advice (CKD stage 3 is a CKD), puts the penicillin allergy and the
  latest creatinine in the header, never the mother's endocarditis, and stores the answer as an AI hypothesis.
- On LongMemEval (chat memory, `benchmarks/longmemeval`), SPARK beat hybrid retrieval, but distinct cues were no better
  than summing there (0.948 vs 0.969 of questions with all evidence in context, within noise), so chat memory keeps
  `combine="sum"`.
- ESC 2023 endocarditis guidelines (95-page PDF): 114 sections, 0 failures, 506 recommendations in 7.7 min for about
  $0.02. A fictional French admission note (14 findings: "pas de prothèse" absent, "Mère : endocardite" family,
  "Hypothèse : endocardite infectieuse" possible), an MRSA culture with susceptibilities (FHIR, no model call) and
  "Quel traitement antibiotique débuter ?" in the encounter: the top recommendation is ESC's MRSA native-valve daptomycin
  regimen, then vancomycin with AUC/MIC targets; the answer (16 citations) weighs the penicillin allergy, the eGFR of 38
  and the weight from the note, and leaves the decision to the clinician. The first run ranked device and right-sided
  recommendations first: a working diagnosis counted as unrecorded and unverifiable conditions scored above unmet ones,
  hence the factors above.
- Not yet measured: a clinical case set with known answers (which recommendation applies, which patient facts must be
  cited), and LongHealth for patient-fact retrieval.
