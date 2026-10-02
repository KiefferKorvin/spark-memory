# Memory roles

The graph memory is first a **semantic memory**: what the stored sources say, used to answer before searching
anywhere else. Around it sit an episodic memory (what was asked recently), a user memory (facts about the subject of
a private scope), a procedural memory (retrieval know-how) and forgetting. Each is stored as graph nodes or
`MemoryRecord`s, so one Neo4j database holds all of them.

## Semantic memory

Retrieval, navigation and synthesis are described in [retrieval](RETRIEVAL.md). Three parts serve "what do I
already know":

- **Dossiers** (`mode: "dossier"`) report what the sources say about a topic: main points with every supporting
  source, agreements, disagreements, dates and gaps.
- **Search memory** keeps a need that was searched online within `SEARCH_MEMORY_DAYS` from being searched again.
- **Freshness**: sources carry `retrieved_at`, and a publication date once known; web knowledge older than the
  search-memory window is searched again when coverage finds it incomplete.

## Scopes

`shared` is the common memory. A private scope is `<kind>:<opaque id>` (for example `user:42`), chosen by the client;
the module never resolves it to an account. `IngestRequest.scope` writes a private copy: identical content in another
scope never merges with it, and private documents stay out of the shared concept graph, so none of their labels can
leak into it. `QueryRequest.scope` reads `shared` plus that one scope. Sources found online during a query are public
and go to `shared`. Nodes stored before scopes existed count as shared. Scopes isolate what one subject's queries read;
access to the API is still the bearer token (see [security](SECURITY.md)).

## Forgetting

| Action | Effect |
|---|---|
| `POST /memory/documents/{id}/forget` `{"reason": ...}` | The document, its sections and chunks are excluded from retrieval (reversible) |
| `... {"reason": ..., "hard": true}` | Erased: nodes, assertions extracted from them, sources providing only them; shared concepts stay |
| `POST /memory/documents/{id}/restore` | Undoes an exclusion |
| `POST /memory/forget` `{"url": ..., "reason": ...}` | Forgets every document from that address |
| `DELETE /memory/scopes/{scope}` | Erases a private scope: its documents, episodes with their query records and traces, searches and facts |

A forgotten address is recorded (`forgotten`) and never fetched again by a query's online search, and the host's
`forgotten` count feeds the retrieval know-how. Query records and traces older than `TRACE_RETENTION_DAYS` (default 30)
are swept at startup and then at most daily; usage records stay for cost accounting.

## Episodic memory

Every completed query is an **episode**: one per distinct question and options in a scope (same content words; a
repeat updates it), with when it was asked, how often, its information needs, the start of its answer, the sources it
used, whether it searched online and its query IDs. `GET /memory/episodes?scope=...` lists what was asked recently;
with `q=...` it returns the episodes closest to a question, with their similarity. A client can use it to avoid
repeating itself, for example before preparing a lesson on a topic it covered last week.

A question asked again within `EPISODE_REUSE_HOURS` (default 24) in the same scope and with the same options gets the
earlier result (`reused_from`, event `EPISODE_REUSED`): an exact repeat without any model call, a paraphrase from
`EPISODE_SIMILARITY` (default 0.94, the threshold calibrated for search memory). Nothing is reused over a change to what
the scope reads: an ingestion (including a duplicate, which adds a source), a forget or restore, or a change to the
user's facts. `reuse: false` asks for a fresh answer; the benchmark harness always sends it. Episodes not asked for
`EPISODE_RETENTION_DAYS` (default 365) are swept.

## User memory

A private scope has **facts** about its user: `preference`, `constraint`, `objective` or `profile`, each a short
third-person statement.

| Endpoint | Use |
|---|---|
| `GET /memory/scopes/{scope}/facts` | All facts, with origin (`stated`, `inferred`), status and evidence |
| `POST /memory/scopes/{scope}/facts` `{"kind": ..., "text": ...}` | A stated fact, active at once |
| `PATCH /memory/scopes/{scope}/facts/{id}` | Edit the text or kind, confirm (`status: active`) or reject a proposal |
| `DELETE /memory/scopes/{scope}/facts/{id}` | Remove it |
| `POST /memory/scopes/{scope}/facts/infer` | Propose facts from the scope's recent questions (one model call) |

Inferred facts are proposals: they must cite the questions they come from, are never sensitive attributes the user
did not state, and apply only once confirmed. The PAKT client owns profiles and goals; this is where facts the memory
learns live, visible and editable. Active facts reach the relevance check and the answer as `user_context`, never the
search text: live recipe queries had exclusions glued into it ("courgette feta sans Crevettes Caramel Viande rouge à
éviter"), which pollutes lexical and vector search. Relevance judges for the user (a recipe built on an excluded
ingredient is not relevant; information about the constraint itself still is), and the answer respects the facts
without citing them as evidence. The instruction is added to the prompt only when a payload carries user context.

## Procedural memory: retrieval know-how

For every web host and retriever (`wikipedia`, `pubmed`, `web`), the memory counts sources found, accepted by the
relevance check, rejected, failed, used in an answer's context and forgotten. A query's online search skips blocked
hosts and hosts whose found sources were never accepted nor used after `HOST_MIN_TRIALS` (default 5), each saving a
relevance call, and ranks the remaining sources by usefulness, `(accepted + used + 1) / (found + 2)` divided by
`1 + forgotten`. `GET /memory/procedures` lists the counts and scores; `POST /memory/procedures/hosts/{host}/block`
`{"reason": ...}` and `DELETE` on the same path block and unblock a host. Counters are updated read-modify-write, so
concurrent queries can lose an increment: they are a trust signal, not accounting.

## Associative recall (SPARK)

A private scope also has an associative memory of what happened with its subject (`spark.py`, on the core in
`associative.py`). PAKT sends each completed lesson there, with the chat about it and the learner's end-of-lesson
report (`POST /memory/spark/sessions`). One extraction call turns the session into memories: short third-person
statements with an assertion state (`stated`, `intended`, `negated`, `hypothetical`, `assistant_said`), a date, a
confidence and the entities they mention, each with general categories. On a French drum lesson it kept, for $0.0004,
that the left hand tenses at 90 BPM, that ghost notes were unknown and failed, that the learner wants rock rather than
jazz and finds 25 minutes too long, and the coach's advice as `assistant_said`. A session is one `spark` record with
float32 vectors truncated to 1,024 dimensions (the embedding model is Matryoshka-trained); storing the same session ID
replaces it, and erasing the scope erases them. Memories state only what was said, done or measured: an inference is
a memory of its own, `hypothetical`, and only when the learner or the coach made it. A memory with code or markup debris,
a cut-off sentence, a script the session never uses or English written about a French session is dropped: on real
lessons about 4% of extractions degenerated that way, whatever the provider, before the prompt asked for complete
sentences and no interpretation (0 of 225 after, reasoning effort minimal or low alike).

Recall (`POST /memory/spark/recall`) makes no model call: entities named in the text (1-6-word phrases, accents
ignored), the entities and memories closest to its embedding (asked of two providers at once, the first answer wins,
and skipped after 3 s, so a slow provider never stalls a chat) and the encounter's working memory seed activation, which spreads one hop over memory-entity-category links
before memories are ranked by activation × temporal × confidence × state × relevance. An `encounter` (PAKT: the coach
thread, `coach:<goal>`) carries what one message activated into the next for three hours, so "and now?" right after a
lesson recalls that lesson. A session may carry a `subject` (PAKT: the learning goal); a recall naming one multiplies
other subjects' memories by 0.3, so a history question no longer brings back an AI lesson when history has nothing close. Recall runs alone here, so it spreads one hop: on LongMemEval dev that put all the evidence
in an 8k-token context for 0.938 of questions against 0.906 with two hops and 0.896 with three, the best-connected
memories otherwise outranking the directly relevant ones. Where another retriever seeds the graph (hybrid search on
LongMemEval), three hops did best (0.969).

A session also leaves one **episode**: a title (3 to 8 words) and 2 to 4 past-tense sentences saying what the learner worked
on or asked, how it went, what the coach advised and what was left open, under the same rules as the memories (nothing the
session does not show, no interpretation, no names). Facts are the semantic memory and episodes the episodic one: an
episode is linked to every entity of its session, so any of them recalls what happened as a unit, and recall returns
episodes beside the facts (two, five for a deep recall). A feed that is not a conversation asks for no episode.

Recall answers only when the text is about something remembered. A memory counts when the text names one of its entities
or when its embedding is at least 0.47 cosine close (0.40 for a deep recall): on real chats unrelated messages ("recette de
crêpes", "salut", "explique-moi les intégrales") reached 0.31-0.46 of their best memory and real follow-ups 0.44-0.70, and
the 3 s embedding cap still falls back to names and the encounter. Next to the best memory, one scoring under a quarter of
it is dropped. What the learner said again or changed is then **settled**, within a subject and a speaker: memories at
least 0.85 close with the same numbers are one fact said again (`seen` counts it), and with other numbers the newest wins
and the old value comes back as `before` ("45 minutes par jour", before: "20 minutes par jour"): cosine alone cannot tell
the two apart (both pairs measured 0.88), the numbers can. `deep` is the deliberate pass, with a second hop, a lower floor
and more results, still without a model call. A learner can read what is remembered (`/memory/spark/memories`) and forget
any memory or episode by its ID.

PAKT uses it three ways. Every coach message recalls first and gives the memories to the coach, marked as possibly
outdated and never as sources for facts about the subject. When they, the lesson and the notebook are not enough for a
factual question, the coach asks for a look-up: a follow-up message answered in the background from `/memory/query`,
which searches the memory deliberately, then Wikipedia and the web for what it lacks. Lesson authors recall what the
learner's past sessions left about the unit they write.

## Images

Images fetched from the web or generated by a model are kept so a client reuses them instead of fetching or paying for
them again. `POST /memory/images` takes base64 `data`, `origin` (`fetched`, with the `url` it came from, or
`generated`, with its `prompt` and `model`), and optional `title`, `description`, `metadata` (author, licence) and
`scope`. The type is read from the bytes (PNG, JPEG, WebP or GIF; no SVG, which can carry scripts) and the decoded size
is capped by `MAX_SOURCE_BYTES`. A fetched image's ID is the digest of its address, so storing it again replaces it. A
generated image's ID is the digest of its content. Metadata lives in `image` records, with a vector of the prompt (or of
the title and description), and the bytes in `image-data` records, so listings and searches never carry images.

Lookups read the scope and the shared memory. `url=` finds the image fetched from that address. `q=` returns the
closest candidates with their `similarity`, and never decides reuse. On tutor-style prompts, rewordings of one
illustration scored 0.75-0.88, while sibling illustrations (violin vs cello bow hold, C vs G chord) reached 0.93, so no
threshold separates the right image from a wrong one. The client, or its model, judges the candidates. Erasing a scope
erases its images, and `DELETE /memory/images/{id}?scope=` forgets one.

PAKT keeps recipe photos (as 900x600 JPEG thumbnails) in the shared memory, so another learner's view of the same
recipe fetches nothing. It keeps tutor illustrations in the learner's scope, since their prompts come from the
learner's questions. The tutor model is shown the closest remembered illustrations and reuses one only when it shows
exactly what is asked. Lesson illustrations from Wikimedia Commons are not kept: the learner's browser loads them from
Commons, so the server never fetches them.

## Not yet

- Episodes are not consolidated into summaries of what a user worked on over weeks.
- Associative memories are not consolidated or superseded: a changed preference stays beside the old one, and the reader
  weighs them by date. The coach's general chat outside lessons is not remembered yet, and co-activation learning is not built.
- Procedural memory does not yet learn which query reformulations found an answer, or task playbooks with outcomes.
- Stale content is refreshed through search memory's window, not detected from its content (for example old prices).
