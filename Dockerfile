FROM python:3.12-slim AS app

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README.md ./
COPY LICENSE*.txt NOTICE.txt ./
COPY src ./src
COPY scripts/run_api.py ./scripts/run_api.py

RUN uv sync --frozen --extra app --extra openai --extra ollama --no-dev

EXPOSE 8000

CMD ["/app/.venv/bin/python", "scripts/run_api.py"]
