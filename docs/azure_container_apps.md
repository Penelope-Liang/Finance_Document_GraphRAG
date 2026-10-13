# Azure Container Apps Deployment

This service is designed to run the orchestration layer in a CPU container and
delegate LLM inference to Azure OpenAI in the cloud profile.

## Runtime Profile

Use these settings for Azure Container Apps:

```text
APP_PROFILE=cloud
RETRIEVER_MODE=dual
LLM_BACKEND=azure-openai
USE_RERANKER=false
```

`APP_PROFILE=cloud` defaults to Azure OpenAI and keeps the CrossEncoder reranker
off unless explicitly overridden.

## Required Secrets

Store secrets in Azure Container Apps, not in code or images:

```text
NEO4J_PASSWORD
AZURE_OPENAI_API_KEY
```

Bind them to container environment variables along with non-secret settings:

```text
NEO4J_URI
NEO4J_USER
NEO4J_DATABASE
AZURE_OPENAI_ENDPOINT
AZURE_OPENAI_API_VERSION
AZURE_OPENAI_DEPLOYMENT
```

## Build And Push

```bash
az acr build \
  --registry <acr-name> \
  --image finance-document-qa:latest \
  .
```

## Deploy

```bash
az containerapp create \
  --name finance-document-qa \
  --resource-group <resource-group> \
  --environment <container-apps-env> \
  --image <acr-name>.azurecr.io/finance-document-qa:latest \
  --target-port 8000 \
  --ingress external \
  --min-replicas 0 \
  --max-replicas 3
```

Set environment variables and secrets after creation, or include them in a
Bicep/Terraform template once the deployment is stable.

## Health Check

Use the shallow health endpoint for container liveness:

```text
GET /health
```

This endpoint does not initialize Neo4j or the LLM backend.

## Cost Controls

- Keep `USE_RERANKER=false` in cloud unless quality requires it.
- Use Azure OpenAI for pay-per-use inference instead of hosting a persistent GPU.
- Keep `min-replicas=0` for demo or low-traffic environments.

## Operational Notes

- Query traffic enters through `POST /query`.
- The production code path is `API -> service.query() -> LangGraph`.
- Request logs include `request_id`, path, status code, and latency.
- Use Container Apps logs to trace failures by `X-Request-ID`.
