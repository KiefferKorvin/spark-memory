from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    memory_mode: Literal["live", "demo", "online_demo"] = "live"
    openrouter_api_key: SecretStr = SecretStr("")
    semantic_model: str = "z-ai/glm-5.3-flash"
    synthesis_model: str = "z-ai/glm-5.3-flash"
    # Answer length cap for synthesis (0: none). A comparison sets one standard for every system it judges,
    # since judges tend to prefer longer answers.
    answer_max_words: int = Field(0, ge=0, le=2000)
    # Merged into every structured chat request. OpenRouter's default routing is price-weighted and can land on
    # very slow providers (observed: 84 s vs 3 s per call). These extraction tasks need little hidden reasoning, but
    # GLM 5.3 rejects disabling it (HTTP 400), so it runs at minimal effort (~10 tokens) and is not returned.
    openrouter_options: dict = {"provider": {"sort": "throughput", "require_parameters": True},
                                "reasoning": {"effort": "minimal", "exclude": True}}
    # Per-operation overrides of openrouter_options. Input-heavy work nobody waits on (concept classification, concept
    # resolution, bibliographic metadata: ~3,500 tokens in, ~100 out) goes first to Sail Research's FP8 endpoint, billed at
    # GLM's list price on input ($0.045/M against $0.15/M on the throughput-routed FP8 providers): 2.6x cheaper on these
    # calls and 2x slower (2026-09-28 bake-off). Its output costs more, so output-heavy calls stay put. The endpoints 3.3x
    # cheaper overall (InferenceNet, DeepInfra, OpenInference) serve FP4; they were not proven worse (SPARK extractions
    # degenerated about 4% of the time on the FP8 route too), but DeepInfra throttles this account and FP8 costs nothing more here.
    openrouter_operation_options: dict = {op: {"provider": {"order": ["Sail Research"], "sort": "throughput", "allow_fallbacks": True,
                                                            "require_parameters": True}}
                                          for op in ("taxonomy_resolution", "resolution", "source_metadata")}
    primary_ontology: str = "UNESCO"
    public_web_enabled: bool = True
    embedding_model: str = ""
    embedding_dimensions: int = Field(1536, ge=8, le=4096)
    navigation_model: str = "typesafe/jev-1.13"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: SecretStr = SecretStr("")
    neo4j_database: str = "neo4j"
    unesco_thesaurus_path: str = ""
    unesco_label_language: str = "fr"
    unesco_required: bool = True
    taxonomy_match_threshold: float = Field(0.75, ge=0, le=1)
    # Confidence from here up to the match threshold links a new LOCAL concept provisionally instead of skipping it.
    taxonomy_provisional_threshold: float = Field(0.5, ge=0, le=1)
    memory_api_token: SecretStr = SecretStr("")
    allowed_source_hosts: str = ""
    # Comma-separated online fallbacks for gaps in memory: wikipedia, pubmed, web (DuckDuckGo). Empty disables.
    external_retrievers: str = "wikipedia,pubmed,web"
    pakt_web_browser: Literal['obscura', 'direct'] = 'obscura'
    pakt_obscura_command: str = ''
    candidate_limit: int = Field(20, ge=1, le=100)
    # Opening text of retrievable leaves shown to the navigation policy; shortened if the request would not fit.
    navigation_excerpt_tokens: int = Field(80, ge=0, le=1000)
    max_parallel_branches: int = Field(6, ge=1, le=20)
    max_children_per_decision: int = Field(3, ge=1, le=100)
    # The root decision picks entry points for a whole need among the index candidates, so it may take more.
    max_root_children: int = Field(8, ge=1, le=100)
    # If the policy prunes every root candidate, the top-ranked ones are still explored (0 disables the floor).
    min_root_children: int = Field(2, ge=0, le=100)
    max_depth: int = Field(10, ge=1, le=50)
    max_total_nodes_explored: int = Field(60, ge=1, le=10000)
    small_document_token_threshold: int = Field(500, ge=1)
    target_chunk_tokens: int = Field(600, ge=8)
    max_chunk_tokens: int = Field(900, ge=8)
    context_token_budget: int = Field(6000, ge=100, le=64000)
    model_input_token_budget: int = Field(24000, ge=1000, le=100000)
    max_source_bytes: int = Field(5_000_000, ge=1024, le=50_000_000)
    max_document_tokens: int = Field(150000, ge=1000)
    max_sections: int = Field(2000, ge=1)
    max_external_sources: int = Field(6, ge=0, le=20)
    max_external_rounds: int = Field(1, ge=0, le=3)
    max_active_queries: int = Field(4, ge=1, le=20)
    # Search memory: a need searched online within this many days is not searched again (0 disables), when its text
    # or embedding matches. On live needs, paraphrases scored 0.94-0.99 and distinct topics at most 0.924.
    search_memory_days: float = Field(30, ge=0, le=3650)
    search_memory_similarity: float = Field(0.94, ge=0, le=1)
    # Query records and their traces older than this are deleted (0 keeps them); usage records stay for cost accounting.
    trace_retention_days: float = Field(30, ge=0, le=3650)
    # Episodes: a question asked again within this many hours (0 disables), in the same scope with the same options,
    # gets the earlier result unless what the scope reads changed since; paraphrases qualify from this similarity.
    episode_reuse_hours: float = Field(24, ge=0, le=8760)
    episode_similarity: float = Field(0.94, ge=0, le=1)
    episode_retention_days: float = Field(365, ge=0, le=3650)
    # Retrieval know-how: a web host whose found sources were never accepted nor used after this many is no longer
    # relevance-checked (0 disables); blocked hosts are always skipped.
    host_min_trials: int = Field(5, ge=0, le=1000)
    # External sources are ingested and indexed before retrieval resumes.
    query_timeout_seconds: float = Field(600, ge=1, le=3600)
    provider_timeout_seconds: float = Field(45, ge=1, le=300)
    provider_retries: int = Field(2, ge=0, le=4)
    enrichment_enabled: bool = True
    enrichment_policy: Literal["on_demand", "frequent", "disabled"] = "on_demand"
    enrichment_hit_threshold: int = Field(3, ge=1)
    cache_max_entries: int = Field(1000, ge=1, le=100000)

    @model_validator(mode="after")
    def bounds(self):
        if self.target_chunk_tokens > self.max_chunk_tokens:
            raise ValueError("target_chunk_tokens must not exceed max_chunk_tokens")
        if self.min_root_children > self.max_root_children:
            raise ValueError("min_root_children must not exceed max_root_children")
        if self.taxonomy_provisional_threshold > self.taxonomy_match_threshold:
            raise ValueError("taxonomy_provisional_threshold must not exceed taxonomy_match_threshold")
        return self

    def validate_live(self):
        if self.memory_mode == "online_demo" and not self.openrouter_api_key.get_secret_value():
            raise ValueError("Online demo requires OPENROUTER_API_KEY")
        if self.memory_mode == "live" and not all([
            self.openrouter_api_key.get_secret_value(), self.semantic_model,
            self.embedding_model, self.neo4j_password.get_secret_value(),
        ]):
            raise ValueError("Live mode requires OpenRouter key, semantic/embedding models and Neo4j password")
