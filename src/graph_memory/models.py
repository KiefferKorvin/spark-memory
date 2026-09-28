from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from uuid import uuid4, uuid5, NAMESPACE_URL

from pydantic import BaseModel, ConfigDict, Field, model_validator


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_id(kind: str, key: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"graph-memory:{kind}:{key}"))


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# "shared" is the common memory; a private scope is "<kind>:<opaque id>" (e.g. "user:42"), chosen by the client, which
# the module never resolves to accounts. Scopes isolate what one subject's queries read, not access to the API.
SCOPE = r"^(shared|[a-z][a-z0-9_-]{0,30}:[A-Za-z0-9_.@-]{1,120})$"


class Node(Strict):
    id: str = Field(default_factory=lambda: str(uuid4()))
    kind: str
    label: str
    text: str = ""
    summary: str = ""
    routing_summary: str = ""
    embedding: list[float] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    scope: str = "shared"


class Source(Node):
    kind: Literal["Source"] = "Source"
    source_type: Literal["file", "url", "user_input", "api", "database", "generated"]
    uri: str | None = None
    filename: str | None = None
    mime_type: str
    author: str | None = None
    retrieved_at: str = Field(default_factory=now)
    published_at: str | None = None
    content_hash: str


class Document(Node):
    kind: Literal["Document"] = "Document"
    document_type: str = "note"
    language: str | None = None
    token_count: int = 0
    created_at: str = Field(default_factory=now)
    retrieval_leaf: bool = False


class Section(Node):
    kind: Literal["Section"] = "Section"
    level: int
    order: int
    section_path: list[str]
    token_count: int = 0


class Chunk(Node):
    kind: Literal["Chunk"] = "Chunk"
    token_count: int
    section_path: list[str]
    context_header: str
    order: int


class Concept(Node):
    kind: Literal["Concept"] = "Concept"
    preferred_label: str
    aliases: list[str] = Field(default_factory=list)
    description: str = ""
    ontology_status: Literal["external", "internal", "hybrid"] = "internal"
    origin: str = Field("LOCAL", min_length=1, max_length=80)
    uri: str | None = None
    external_id: str | None = None
    pref_labels: dict[str, list[str]] = Field(default_factory=dict)
    alt_labels: dict[str, list[str]] = Field(default_factory=dict)
    hidden_labels: dict[str, list[str]] = Field(default_factory=dict)
    descriptions: dict[str, list[str]] = Field(default_factory=dict)


class TaxonomyGroup(Concept):
    """An official SKOS Collection, not an indexing concept."""
    kind: Literal["TaxonomyGroup"] = "TaxonomyGroup"
    group_type: Literal["domain", "microthesaurus", "collection"]


class ExternalConcept(Node):
    kind: Literal["ExternalConcept"] = "ExternalConcept"
    system: str
    external_id: str
    uri: str | None = None


class Assertion(Node):
    kind: Literal["Assertion"] = "Assertion"
    assertion_type: Literal["fact", "claim", "event", "decision", "preference", "procedure"] = "claim"
    proposition: str
    confidence: float = Field(ge=0, le=1)
    valid_from: str | None = None
    valid_to: str | None = None
    created_at: str = Field(default_factory=now)


NODE_TYPES = {c.model_fields["kind"].default: c for c in
              [Source, Document, Section, Chunk, Concept, TaxonomyGroup, ExternalConcept, Assertion]}


def node_from(data: dict) -> Node:
    return NODE_TYPES[data["kind"]].model_validate(data)


class Relation(str, Enum):
    PROVIDES = "PROVIDES"
    CONTAINS = "CONTAINS"
    ABOUT = "ABOUT"
    MENTIONS = "MENTIONS"
    BROADER_THAN = "BROADER_THAN"
    RELATED_TO = "RELATED_TO"
    SEMANTIC_RELATION = "SEMANTIC_RELATION"
    HAS_MEMBER = "HAS_MEMBER"
    EXACT_MATCH = "EXACT_MATCH"
    CLOSE_MATCH = "CLOSE_MATCH"
    SUPPORTED_BY = "SUPPORTED_BY"
    CONTRADICTS = "CONTRADICTS"


class Edge(Strict):
    source: str
    target: str
    relation: Relation
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def id(self):
        predicate = ":" + str(self.metadata.get("predicate", "")) + ":" + str(self.metadata.get("rf2", {}).get("id", "")) if self.relation == Relation.SEMANTIC_RELATION else ""
        return stable_id("edge", f"{self.source}:{self.relation.value}:{self.target}{predicate}")


class IngestRequest(Strict):
    title: str = Field("", max_length=300)
    text: str | None = None
    content_base64: str | None = None
    url: str | None = Field(None, max_length=2048)
    mime_type: str = "text/plain"
    filename: str | None = Field(None, max_length=255)
    source_type: Literal["file", "url", "user_input", "api", "database", "generated"] = "user_input"
    author: str | None = Field(None, max_length=300)
    published_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    scope: str = Field("shared", pattern=SCOPE)  # a private scope keeps its own copy, out of the shared concept graph

    @model_validator(mode="after")
    def one_content(self):
        if sum(x is not None for x in (self.text, self.content_base64, self.url)) != 1:
            raise ValueError("Provide exactly one of text, content_base64, url")
        return self


FactKind = Literal["preference", "constraint", "objective", "profile"]


class Fact(Strict):
    """Something about the user of a private scope: stated by the client, or inferred and then confirmed."""
    kind: FactKind
    text: str = Field(min_length=1, max_length=500)  # a short third-person statement, e.g. "Avoids red meat"


class FactUpdate(Strict):
    kind: FactKind | None = None
    text: str | None = Field(None, min_length=1, max_length=500)
    status: Literal["active", "proposed", "rejected"] | None = None  # confirm (active) or reject a proposal


class ProposedFact(Fact):
    evidence: list[int] = Field(default_factory=list, max_length=50)  # numbers of the questions that support it


class FactProposals(Strict):
    facts: list[ProposedFact] = Field(max_length=20)


class BlockRequest(Strict):
    reason: str = Field(min_length=1, max_length=300)


class ForgetRequest(Strict):
    reason: str = Field(min_length=1, max_length=300)  # e.g. outdated, wrong, low quality, user request
    hard: bool = False  # erase (irreversible) instead of excluding from retrieval
    url: str | None = Field(None, max_length=2048)  # forget every document from this URL; required on /memory/forget


IMAGE_ID = r"^[0-9a-f]{64}$"


class ImageRequest(Strict):
    """An image to keep for reuse: fetched from `url`, or generated from `prompt` (by `model`)."""
    data: str = Field(min_length=1)  # base64; the size limit applies to the decoded bytes
    origin: Literal["fetched", "generated"]
    url: str | None = Field(None, max_length=2048)
    prompt: str | None = Field(None, min_length=1, max_length=8000)
    model: str | None = Field(None, max_length=200)
    title: str = Field("", max_length=300)
    description: str = Field("", max_length=2000)
    metadata: dict[str, Any] = Field(default_factory=dict)  # e.g. author, license, page_url
    scope: str = Field("shared", pattern=SCOPE)

    @model_validator(mode="after")
    def provenance(self):
        if self.origin == "fetched" and not (self.url or "").startswith(("http://", "https://")):
            raise ValueError("A fetched image needs the http(s) url it came from")
        if self.origin == "generated" and not self.prompt:
            raise ValueError("A generated image needs its prompt")
        return self


class SourceMetadata(Strict):
    title: str = Field(min_length=1, max_length=300)
    author: str | None = None
    published_at: str | None = None
    publisher: str | None = None
    language: str | None = None
    description: str = Field(max_length=2000)
    keywords: list[str] = Field(default_factory=list, max_length=20)


class ConceptSpec(Strict):
    label: str = Field(min_length=1, max_length=200)
    aliases: list[str] = Field(default_factory=list, max_length=20)
    description: str = Field(default="", max_length=1500)
    broader: list[str] = Field(default_factory=list, max_length=10)
    related: list[str] = Field(default_factory=list, max_length=10)


class Understanding(Strict):
    document_type: str
    language: str | None = None
    summary: str = Field(max_length=2000)
    routing_summary: str = Field(max_length=2000)
    temporal_scope: str | None = None
    concepts: list[ConceptSpec] = Field(default_factory=list, max_length=5)


class Resolution(Strict):
    reuse_id: str | None


class TaxonomyResolution(Strict):
    reuse_id: str | None
    parent_ids: list[str] = Field(max_length=6)
    confidence: float = Field(ge=0, le=1)


class Need(Strict):
    id: str
    description: str = Field(min_length=1, max_length=1500)


class Decomposition(Strict):
    information_needs: list[Need] = Field(min_length=1, max_length=12)

    @model_validator(mode="after")
    def unique(self):
        if len({n.id for n in self.information_needs}) != len(self.information_needs):
            raise ValueError("Information need IDs must be unique")
        return self


class Decision(Strict):
    node_id: str
    action: Literal["EXPAND", "SELECT", "PRUNE"]


class NavigationDecision(Strict):
    decisions: list[Decision]
    current_node_action: Literal["CONTINUE", "FOUND_EVIDENCE", "DEAD_END"]


class Branch(Strict):
    id: str = Field(default_factory=lambda: str(uuid4()))
    information_need_id: str
    node_id: str | None = None
    parent_branch_id: str | None = None
    path: list[str] = Field(default_factory=list)
    depth: int = 0
    priority: float = 0
    action: Literal["EXPAND", "SELECT"] = "EXPAND"
    status: Literal["ACTIVE", "COMPLETE", "DEAD_END", "PRUNED"] = "ACTIVE"
    evidence_found: list[str] = Field(default_factory=list)
    visited_nodes: list[str] = Field(default_factory=list)


class Evidence(Strict):
    id: str
    information_need_ids: list[str]
    source_node_id: str
    source_type: str
    text: str
    relevance_score: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    provenance: dict[str, Any]


class Relevance(Strict):
    relevant: bool
    confidence: float = Field(ge=0, le=1)


class CoverageItem(Strict):
    information_need_id: str
    status: Literal["COVERED", "PARTIAL", "MISSING"]
    evidence_ids: list[str]
    missing: str = ""


class Coverage(Strict):
    coverage: list[CoverageItem]
    overall_status: Literal["SUFFICIENT", "INSUFFICIENT"]


class AssertionSpec(Strict):
    proposition: str = Field(min_length=1, max_length=2000)
    assertion_type: Literal["fact", "claim", "event", "decision", "preference", "procedure"]
    confidence: float = Field(ge=0, le=1)
    supporting_quote: str = Field(min_length=1)
    contradicts_ids: list[str] = Field(default_factory=list)
    valid_from: datetime | None = None
    valid_to: datetime | None = None


class Enrichment(Strict):
    assertions: list[AssertionSpec] = Field(max_length=12)


class Answer(Strict):
    answer: str
    evidence_ids: list[str]


class QueryRequest(Strict):
    query: str = Field(min_length=1, max_length=5000)
    allow_external: bool = True
    answer_max_words: int | None = Field(None, ge=1, le=2000)  # overrides ANSWER_MAX_WORDS for this query
    synthesize: bool = True  # false: retrieval only, for clients that use the evidence and not the answer
    # dossier: what the sources say about a topic (wider exploration, points with agreements and disagreements)
    mode: Literal["answer", "dossier"] = "answer"
    scope: str | None = Field(None, pattern=SCOPE)  # a private scope read beside the shared memory
    reuse: bool = True  # false: answer afresh even if the same question was answered recently (see episodes)
    original_sources_only: bool = False


class TraceEvent(Strict):
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    query_id: str
    sequence: int
    timestamp: str = Field(default_factory=now)
    event_type: str
    node_id: str | None = None
    branch_id: str | None = None
    parent_branch_id: str | None = None
    information_need_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
