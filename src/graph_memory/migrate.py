r"""Re-embed the graph after an EMBEDDING_DIMENSIONS change (paid embeddings; resumable).

Rebuilds vector indexes whose dimension differs, re-embeds every node whose vector has another size, then the
thesaurus. Reads .env like the API; set NEO4J_URI/NEO4J_PASSWORD in the environment to migrate another database.
A model change that keeps the dimension is not detected here: use a fresh database for that.
    .venv\Scripts\python -m graph_memory.migrate
"""
import asyncio
import json

from .ingestion import embedding_text
from .unesco import embed_taxonomy


async def reembed(repository, models, settings, batch=8):
    """Nodes whose vector already has EMBEDDING_DIMENSIONS are skipped, so an interrupted run resumes.
    Eight texts of at most a chunk (~1k tokens) each keep a request inside MODEL_INPUT_TOKEN_BUDGET."""
    count = 0
    while nodes := await repository.stale_embeddings(settings.embedding_dimensions, batch):
        for node, vector in zip(nodes, await models.embed_batch([embedding_text(n) for n in nodes])):
            node.embedding = vector
        await repository.update_nodes(nodes)
        count += len(nodes)
    return {"nodes": count, "taxonomy": await embed_taxonomy(repository, models, settings)}


async def main():
    from .config import Settings
    from .graph import Neo4jGraph
    from .llm import OpenRouter
    settings = Settings()
    repository = Neo4jGraph(settings)
    models = OpenRouter(settings, repository)
    try:
        await repository.vector_indexes(migrate=True)
        await repository.initialize()
        print(json.dumps(await reembed(repository, models, settings), indent=2))
    finally:
        await models.close()
        await repository.close()


if __name__ == "__main__":
    asyncio.run(main())
