from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    memory_mode: Literal["live", "demo", "online_demo"] = "live"
    openrouter_api_key: SecretStr = SecretStr("")
    semantic_model: str = "deepseek/deepseek-v4.1-flash"
    synthesis_model: str = "deepseek/deepseek-v4.1-flash"
    # Merged into every structured chat request. OpenRouter's default routing is price-weighted and can land on
    # very slow providers (observed: 84 s vs 3 s per call); these extraction tasks also do not need hidden reasoning.
    openrouter_options: dict = {"provider": {"sort": "throughput", "require_parameters": True}, "reasoning": {"enabled": False}}
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
    memory_api_token: SecretStr = SecretStr("")
    allowed_source_hosts: str = ""
    # Comma-separated online fallbacks for gaps in memory: wikipedia, pubmed, web (DuckDuckGo). Empty disables.
    external_retrievers: str = "wikipedia,pubmed,web"
    candidate_limit: int = Field(20, ge=1, le=100)
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
    query_timeout_seconds: float = Field(180, ge=1, le=3600)
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
        return self

    def validate_live(self):
        if self.memory_mode == "online_demo" and not self.openrouter_api_key.get_secret_value():
            raise ValueError("Online demo requires OPENROUTER_API_KEY")
        if self.memory_mode == "live" and not all([
            self.openrouter_api_key.get_secret_value(), self.semantic_model,
            self.embedding_model, self.neo4j_password.get_secret_value(),
        ]):
            raise ValueError("Live mode requires OpenRouter key, semantic/embedding models and Neo4j password")
