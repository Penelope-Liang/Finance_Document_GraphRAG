# Runbook

## Health Fails

`GET /health` checks Neo4j and the configured answer LLM concurrently, with a
10-second timeout per check. Neo4j executes `RETURN 1` in `NEO4J_DATABASE`;
the LLM receives a minimal generation request using the configured model or
Azure deployment. It returns HTTP 200 when both pass, or HTTP 503 with
`checks.neo4j` and `checks.llm` statuses, elapsed times and sanitized errors.
Provider error text and credentials are not returned.

`GET /health/live` checks only that the API process responds. Use this endpoint
for frequent startup/liveness probes and container checks without credentials.
The deeper `/health` call can incur model usage and cold-start latency; use it
for deployment checks or on-demand diagnostics. Neither endpoint validates
stored documents, search indexes, the embedding model, or answer correctness.

If `/health/live` fails, check container startup logs, `API_PORT`, and Python
import errors. If only `/health` fails, inspect the failing dependency and its
configuration; a cold Ollama model may exceed the check timeout.

## Query Returns `Missing env var: NEO4J_PASSWORD`

The service reached `/query` and tried to initialize Neo4j without a password.
Add `NEO4J_PASSWORD` as a Container Apps secret and bind it as an environment
variable.

## Query Returns Azure OpenAI Credential Errors

Check:

```text
AZURE_OPENAI_ENDPOINT
AZURE_OPENAI_API_KEY
AZURE_OPENAI_API_VERSION
AZURE_OPENAI_DEPLOYMENT
```

For cloud runs, confirm `APP_PROFILE=cloud` and `LLM_BACKEND=azure-openai`.

## Neo4j Connection Errors

Check `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, network access, and database
name. Confirm the financial document graph and vector/full-text indexes exist.

## High Latency

Check request logs for latency, then inspect:

```text
LLM_BACKEND
USE_RERANKER
RETRIEVER_MODE
top_k
effective_search_ratio
```

If cloud latency is high, keep reranking disabled and reduce retrieval breadth.

## Unexpected Reranker Use

Confirm `USE_RERANKER=false`. `DualRetriever` only enables reranking when the
service explicitly passes a reranker model.

## Incident Notes

Record:

```text
request_id
APP_PROFILE
RETRIEVER_MODE
LLM_BACKEND
error message
time of failure
recent deployment or config change
```
