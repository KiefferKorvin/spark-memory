# Jev navigation

Navigation is exclusively `typesafe/jev-1.13` by default, configurable through `NAVIGATION_MODEL`. OpenRouter exposes its typed decisions separately from chat completions. The adapter submits a bounded state object, which carries the navigation policy once (`navigation_policy`), and one short choice question per candidate that refers to it, plus a branch-status question, to `/api/alpha/decisions`. Answers map back to candidates by index, so candidates travel without node IDs or empty fields and with scores rounded to two decimals. Replaying 40 recorded root decisions, this took requests from 10.7k to 6.2k input tokens and 1.4 s to 0.9 s, with as many gold documents selected as the earlier format (115 against 114), which repeated the policy in every question. Responses are converted to the internal validated `NavigationDecision` schema.

| Choice | Meaning |
|---|---|
| EXPAND | Schedule a promising container/concept branch |
| SELECT | Retrieve evidence from the selected node |
| PRUNE | Do not explore this candidate |
| CONTINUE | Candidate routes remain |
| FOUND_EVIDENCE | Current node is indicated as evidence; collector still verifies original text |
| DEAD_END | Stop this branch and backtrack to remaining frontier work |

Payloads contain the information need, current path/node, candidate routing metadata, existing evidence identifiers and remaining budgets. Retrievable leaves (leaf documents, chunks, assertions) also carry an `excerpt`: the opening `NAVIGATION_EXCERPT_TOKENS` (default 80) of their original text, since short leaves such as forum answers rarely reveal in a routing summary whether they answer the need; containers and concepts carry routing summaries only. If a request would exceed the default 24k `MODEL_INPUT_TOKEN_BUDGET`, excerpts are halved until it fits instead of failing. Persisted events keep at most 300 characters of each excerpt. Unknown node IDs, duplicate choices, malformed JSON and invalid actions are rejected. A decision never constitutes evidence, grants access, or generates a final answer. Branch limits are enforced in code even if the policy selects too many nodes: `MAX_ROOT_CHILDREN` for the root decision of each need, `MAX_CHILDREN_PER_DECISION` below it. A root decision that keeps nothing is overridden by the `MIN_ROOT_CHILDREN` top-ranked candidates (`NAVIGATION_FLOOR` event).

The separate semantic model handles understanding, concept resolution, decomposition, evidence relevance/coverage, enrichment and synthesis. Embeddings have their own model. All machine outputs are validated by Pydantic. Transient requests and malformed output get bounded exponential-backoff retries; permanent provider HTTP errors fail immediately. API keys never reach the frontend.

The central [prompt registry](../src/graph_memory/prompts.py) includes an explicit version. The bounded LRU cache keys include model, prompt version, operation, schema and payload; embeddings include model, dimensions and text hash. Cached routing/understanding results are reusable; navigation and coverage are not cached. Durable usage records include operation, model, query ID, input/output tokens, latency, provider-reported cost when available and attempt status. Missing costs remain unknown, not estimates fabricated from a price table.

Provider reference: [OpenRouter Jev typed questions example](https://openrouter.ai/labs/jev/compile), [structured output documentation](https://openrouter.ai/docs/guides/features/structured-outputs). No live provider call is required by automated tests.
