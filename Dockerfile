FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir .
ENV TIKTOKEN_CACHE_DIR=/app/tokenizer
RUN python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')" && useradd --create-home memory
USER memory
EXPOSE 8010
CMD ["uvicorn", "graph_memory.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8010", "--workers", "1"]
