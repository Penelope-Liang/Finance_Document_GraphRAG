# Azure deployment

The [CD workflow](../.github/workflows/cd.yaml) deploys the API to an existing
Azure Container App using Azure Container Registry (ACR).

```text
main CI succeeds → checks → build container → local /health
                → push image to ACR → update Container App → deployed /health
```

Each image tag includes the commit SHA, workflow run ID, and attempt number.
Deployments are serialized, and outdated commits are rejected before updating the
app. Manual runs are available from **Actions → CD → Run workflow**, on `main`.
They run the same checks. Failed CI and pull-request runs do not deploy.

## One-time Azure setup

Create an ACR registry and a Container App with:

- A single application container, target port `8000`, and single-revision mode.
- HTTPS ingress reachable from the GitHub runner. `/health` must be accessible
  to the workflow; configure access controls for `/query` separately.
- A managed identity configured to pull images from your ACR registry.
- Startup/readiness probes appropriate for the API, and the environment settings
  and secrets below. A `/health` probe checks process health only.

CD updates the image and four runtime flags. It does not create cloud resources,
import PDFs, build graph indexes, or replace your application secrets.

Create an Entra application or managed identity for GitHub deployments, with a
federated credential:

```text
Issuer:   https://token.actions.githubusercontent.com
Subject:  repo:Penelope-Liang/Finance_Document_GraphRAG:environment:production
Audience: api://AzureADTokenExchange
```

Grant the deployment identity permission to read/push images in ACR and read/update
the Container App. The app's image-pull identity is separate from the GitHub
identity and needs registry pull permissions.

See [Azure OIDC setup](https://learn.microsoft.com/en-us/azure/developer/github/connect-from-azure-openid-connect)
and [managed-identity image pulls](https://learn.microsoft.com/en-us/azure/container-apps/managed-identity-image-pull)
for the identity configuration steps.

## GitHub configuration

Create a GitHub environment named **production**, restricted to the `main` branch.
Add these environment secrets:

| Secret | Value |
| --- | --- |
| `AZURE_CLIENT_ID` | Deployment identity client ID |
| `AZURE_TENANT_ID` | Entra tenant ID |
| `AZURE_SUBSCRIPTION_ID` | Target subscription ID |

Add these environment variables:

| Variable | Value |
| --- | --- |
| `AZURE_RESOURCE_GROUP` | Resource group containing the Container App |
| `AZURE_ACR_NAME` | Registry resource name, without `.azurecr.io` |
| `AZURE_CONTAINER_APP` | Existing Container App name |

After setup, set the **repository variable** `ENABLE_AZURE_CD=true` to enable the
workflow. Until then the deployment job is skipped. OIDC avoids storing an Azure
client password in GitHub.

## Application configuration

Store `NEO4J_PASSWORD` and `AZURE_OPENAI_API_KEY` as Container App secrets and bind
them to the matching environment variables. Configure:

```text
NEO4J_URI
NEO4J_USER
NEO4J_DATABASE
NEO4J_VECTOR_INDEX
NEO4J_FULLTEXT_INDEX
AZURE_OPENAI_ENDPOINT
AZURE_OPENAI_API_VERSION
AZURE_OPENAI_DEPLOYMENT
OLLAMA_HOST
OLLAMA_EMBED_MODEL
```

The deployment sets `APP_PROFILE=cloud`, `LLM_BACKEND=azure-openai`,
`RETRIEVER_MODE=auto`, and `USE_RERANKER=false`.

Azure OpenAI handles answer generation. **Embeddings still use Ollama in the
current implementation.** `OLLAMA_HOST` must reach a running service from Azure;
`localhost:11434` does not point to your laptop. Use the same embedding model as
the ingested vectors. Optional Ollama calculation/judge models also require that
service when enabled.

## Deployment checks and rollback

The workflow checks the built container before pushing it, waits for the new
revision to become ready, then calls its application's `/health` endpoint. A
failure marks the deployment run as failed. This does not verify Neo4j queries,
model inference, or `/query` answer quality; check those separately after setup.

The Actions summary records the new image, revision, URL, and previous image.
To restore that previous image:

```bash
az containerapp update \
  --name <container-app> \
  --resource-group <resource-group> \
  --image <previous-image-from-actions-summary>
```

Rollback is manual. Inspect Container Apps logs and verify readiness after a
rollback. The workflow does not delete old revisions or images.
