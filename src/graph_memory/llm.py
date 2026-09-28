import asyncio
import hashlib
import json
import math
import re
import time
from collections import OrderedDict
from typing import Protocol, TypeVar
from uuid import uuid4

import httpx
from pydantic import BaseModel, ValidationError

from .models import Decision, NavigationDecision
from .prompts import VERSION, prompt, synthesis_rules

T = TypeVar("T", bound=BaseModel)


class ModelProvider(Protocol):
    async def structured(self, operation: str, payload: dict, schema: type[T], query_id: str | None = None) -> T: ...
    async def embed(self, text: str, query_id: str | None = None) -> list[float]: ...
    async def embed_batch(self, texts: list[str], query_id: str | None = None) -> list[list[float]]: ...
    async def navigate(self, payload: dict, query_id: str) -> NavigationDecision: ...
    async def close(self): ...


# Operations that write a cited answer from evidence: same model, citation check and closing rules.
ANSWERS = {"synthesis", "dossier"}


# Operations nobody waits on: they take the cheap routing (Settings.openrouter_background_options).
BACKGROUND = {"understanding", "source_metadata", "resolution", "taxonomy_resolution", "enrichment", "spark_extract", "user_facts"}


class ProviderError(RuntimeError):
    pass


class AccountError(RuntimeError):
    """The provider refused the key or its credits (HTTP 401, 403, or 402 other than in-flight reservations). Not a
    ProviderError on purpose: every later call would fail too, so no fallback may turn it into an empty answer."""


class OpenRouter:
    def __init__(self, settings, repository, client=None):
        self.settings = settings
        self.repository = repository
        self.client = client or httpx.AsyncClient(timeout=settings.provider_timeout_seconds,
                                                  follow_redirects=False, trust_env=False)
        self.cache = OrderedDict()

    async def close(self):
        await self.client.aclose()

    def cached(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]

    def save_cache(self, key, value):
        self.cache[key] = value
        self.cache.move_to_end(key)
        while len(self.cache) > self.settings.cache_max_entries:
            self.cache.popitem(last=False)

    async def request(self, path, body, operation, query_id, validate):
        from .parsing import token_count
        # Count what the model reads; dumping chat bodies double-escapes the embedded JSON payload and overcounts.
        text = "".join(m["content"] for m in body["messages"]) if "messages" in body else json.dumps(body, ensure_ascii=False)
        if token_count(text) > self.settings.model_input_token_budget:
            raise ProviderError("Model input exceeds configured token budget")
        for attempt in range(self.settings.provider_retries + 1):
            started = time.monotonic()
            usage, status, retryable, reason, wait, code = {}, "error", True, None, None, None
            try:
                response = await self.client.post("https://openrouter.ai" + path, json=body,
                    headers={"Authorization": "Bearer " + self.settings.openrouter_api_key.get_secret_value(),
                             "X-Title": "Progressive Graph Memory"})
                if not 200 <= response.status_code < 300:
                    code = response.status_code
                    # 402 "retry after in-flight requests settle": credit is reserved by concurrent requests, not spent.
                    reserved = response.status_code == 402 and "in-flight" in response.text.lower()
                    retryable = response.status_code in (408, 429) or response.status_code >= 500 or reserved
                    if reserved:  # wait as long as OpenRouter asks (Retry-After), within reason
                        wait = min(30.0, float(response.headers.get("retry-after") or 5))
                    raise ProviderError(f"Model provider HTTP {response.status_code}")
                obj = response.json()
                usage = obj.get("usage") or {}
                result = validate(obj)
                status = "ok"
                return result
            except (httpx.HTTPError, ValidationError, ValueError, KeyError, TypeError, ProviderError) as exc:
                reason = f"{type(exc).__name__}: {exc}"[:200]
                if attempt >= self.settings.provider_retries or not retryable:
                    error = AccountError if code in (401, 402, 403) and not retryable else ProviderError
                    raise error(f"{operation} failed after {attempt+1} attempt(s): {reason}") from exc
                # Repair a response that failed validation by asking for the original schema; never salvage unvalidated
                # JSON. A network error or HTTP status leaves no response to repair, so the request is sent unchanged.
                if "messages" in body and code is None and not isinstance(exc, httpx.HTTPError):
                    body = {**body, "messages": [*body["messages"], {
                        "role": "user", "content": f"The preceding response was invalid ({reason}). Return only valid JSON matching the supplied schema. For synthesis, copy evidence IDs exactly, without extra characters or suffixes, into both inline [id] citations and evidence_ids. Use only IDs present in the supplied evidence."}]}
            finally:
                await self.repository.record("usage", str(uuid4()), {
                    "model": body["model"], "operation": operation, "query_id": query_id,
                    "input_tokens": usage.get("prompt_tokens", usage.get("input_tokens", 0)),
                    "output_tokens": usage.get("completion_tokens", usage.get("output_tokens", 0)),
                    "estimated_cost": usage.get("cost"), "latency": time.monotonic()-started,
                    "status": status, "attempt": attempt+1, **({"error": reason} if reason else {}),
                })
            await asyncio.sleep(wait if wait is not None else min(0.5 * 2**attempt, 4))

    async def structured(self, operation, payload, schema, query_id=None):
        options = self.settings.openrouter_background_options if operation in BACKGROUND else self.settings.openrouter_options
        model = self.settings.synthesis_model if operation in ANSWERS else self.settings.semantic_model
        # A synthesis length cap is an instruction for the system prompt, not data in the user message.
        limit = payload.get("answer_max_words", 0) if operation in ANSWERS else 0
        payload = {k: v for k, v in payload.items() if k != "answer_max_words"}
        serial = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        key = hashlib.sha256(f"{VERSION}:{model}:{operation}:{schema.model_json_schema()}:{serial}".encode()).hexdigest()
        reusable = operation in {"understanding", "resolution"}
        cached = self.cached(key) if reusable else None
        if cached is not None:
            return schema.model_validate(cached)
        def validate(obj):
            value = schema.model_validate_json(obj["choices"][0]["message"]["content"])
            if operation in ANSWERS and payload["evidence"] and not cited(value.answer, payload["evidence"]):
                # Other bracketed text is tolerated; cited IDs are derived from the answer by the caller.
                raise ValueError("Synthesis must cite supplied evidence IDs inline")
            return value
        result = await self.request("/api/v1/chat/completions", {
            **options, "model": model,
            "messages": [{"role": "system", "content": prompt(operation, limit, personal=bool(payload.get("user_context")))},
                         {"role": "user", "content": serial},
                         *([{"role": "user", "content": synthesis_rules(limit)}] if operation in ANSWERS else [])],
            "response_format": {"type": "json_schema", "json_schema": {
                "name": schema.__name__, "strict": True, "schema": strict_schema(schema.model_json_schema())}},
            "max_tokens": 5000,
        }, operation, query_id, validate)
        if reusable:
            self.save_cache(key, result.model_dump())
        return result

    async def embed(self, text, query_id=None):
        key = hashlib.sha256(f"embedding:{VERSION}:{self.settings.embedding_model}:{self.settings.embedding_dimensions}:{text}".encode()).hexdigest()
        cached = self.cached(key)
        if cached is not None:
            return list(cached)
        vector = await self.request("/api/v1/embeddings", {
            "model": self.settings.embedding_model, "input": text,
            "dimensions": self.settings.embedding_dimensions,
        }, "embedding", query_id, lambda obj: self.vector(obj["data"][0]["embedding"]))
        self.save_cache(key, vector)
        return vector

    async def embed_batch(self, texts, query_id=None):
        """One request for many texts (bulk jobs, a query's information needs); not cached."""
        def validate(obj):
            rows = sorted(obj["data"], key=lambda r: r["index"])
            if len(rows) != len(texts):
                raise ValueError("Embedding count differs from input count")
            return [self.vector(r["embedding"]) for r in rows]
        return await self.request("/api/v1/embeddings", {
            "model": self.settings.embedding_model, "input": texts,
            "dimensions": self.settings.embedding_dimensions,
        }, "embedding", query_id, validate)

    def vector(self, value):
        if len(value) != self.settings.embedding_dimensions or not all(
            type(x) in (int, float) and math.isfinite(x) for x in value
        ) or not any(value):
            raise ValueError("Invalid embedding dimensions or values")
        return value

    async def navigate(self, payload, query_id):
        """The policy travels once, in the state; each candidate's question only points at it. Repeating the policy in
        every question, plus node UUIDs and empty fields, made up ~40% of the request: replaying 40 recorded root
        decisions took 10.7k input tokens down to 6.2k and 1.4 s to 0.9 s, with as many gold documents selected."""
        ids = [c["node_id"] for c in payload["candidate_children"][:self.settings.candidate_limit]]
        if not ids:
            return NavigationDecision(decisions=[], current_node_action="DEAD_END")
        # Answers come back by index, so Jev does not need the node IDs.
        candidates = [{k: round(v, 2) if k == "score" else v for k, v in c.items() if k != "node_id" and v not in (None, [], "")}
                      for c in payload["candidate_children"][:len(ids)]]
        questions = {f"node_{i}": {
            "type": "choice",
            "instructions": f"Apply state.navigation_policy to candidate_children[{i}] for state.information_need.",
            "criteria": {"EXPAND": "A container or concept likely to lead to evidence for the need",
                         "SELECT": "Its excerpt (or routing summary when there is none) addresses the need, even partially or anecdotally",
                         "PRUNE": "Off-topic, or shares only keywords with the need"},
        } for i in range(len(ids))}
        questions["branch_status"] = {"type": "choice", "instructions": "Apply state.navigation_policy: decide the current branch status.",
                                      "criteria": {"CONTINUE": "Useful candidate branches remain", "FOUND_EVIDENCE": "Current node contains directly useful evidence", "DEAD_END": "No useful current node or candidates"}}

        def validate(obj):
            answers = obj["answers"]
            for key, question in questions.items():
                if answers[key]["type"] != "choice" or answers[key]["choice"] not in question["criteria"]:
                    raise ValueError("Invalid decision choice")
            return NavigationDecision(decisions=[Decision(node_id=node_id, action=answers[f"node_{i}"]["choice"])
                                                  for i, node_id in enumerate(ids)],
                                      current_node_action=answers["branch_status"]["choice"])
        from .parsing import token_count, truncate
        body = {"model": self.settings.navigation_model, "questions": questions,
                "state": {"navigation_policy": prompt("navigation"), **payload, "candidate_children": candidates}}
        # Excerpts are the elastic part of the state: halve them until the request fits rather than failing it.
        while token_count(json.dumps(body, ensure_ascii=False)) > self.settings.model_input_token_budget and any(c.get("excerpt") for c in candidates):
            candidates = [{**c, "excerpt": truncate(c["excerpt"], token_count(c["excerpt"]) // 2)} if c.get("excerpt") else c
                          for c in candidates]
            body["state"]["candidate_children"] = candidates
        return await self.request("/api/alpha/decisions", body, "navigation", query_id, validate)


def cited(answer, evidence):
    """Known evidence IDs cited inline as [id] or [id, id], in order of first appearance."""
    available = {e["id"] for e in evidence}
    groups = [[p.strip() for p in g.split(",")] for g in re.findall(r"\[([^\[\]\n]+)\]", answer)]
    return list(dict.fromkeys(i for g in groups if all(p in available for p in g) for i in g))


def strict_schema(value):
    """OpenRouter strict providers require all object fields, including nullable ones."""
    if isinstance(value, list):
        return [strict_schema(v) for v in value]
    if not isinstance(value, dict):
        return value
    result = {k: strict_schema(v) for k, v in value.items() if k != "default"}
    if result.get("type") == "object" and "properties" in result:
        result["required"] = list(result["properties"])
        result["additionalProperties"] = False
    return result
