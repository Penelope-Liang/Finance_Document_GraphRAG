# Batch queries and evaluation

Complete [ingestion](ingestion.md) and configure `.env` before running queries.
Local examples use Ollama; cloud generation uses the Azure settings described in
[deployment](azure_container_apps.md).

## Query routing

| Mode | Retrieval |
| --- | --- |
| `auto` | Open-book → hybrid; numerical → graph, with hybrid fallback |
| `hybrid` | Vector and full-text search over chunks |
| `graph` | Graph evidence only, without automatic hybrid fallback |
| `text2cypher` | LLM-generated Cypher queries |
| `dual` | Hybrid chunks combined with graph evidence |

`auto` is the API and CLI default. Dataset `question_reasoning` labels select the
route when present; otherwise the question-style classifier chooses it. The
classifier is rule-based by default; `--question-style llm-auto` uses an LLM.
Graph fallback occurs on errors or when all graph branches return no evidence.
An empty individual branch does not trigger fallback if another has evidence.
The initial route, actual route, and fallback reason are recorded in `run_config`.

## Run a batch

```bash
uv run --env-file .env --extra ollama --extra openai python \
  scripts/run_document.py \
  --questions-path dataset/financebench/financebench_metrics_remaining_37.jsonl \
  --retriever-mode auto \
  --use-calculation-executor \
  --output output/answers.json
```

Set `--retriever-mode hybrid` for the text-only baseline. The optional calculation
executor uses the configured calculation model; see `scripts/run_document.py --help`
for model, query-rewriting, and retrieval settings. Install any selected local
models in Ollama before running the batch.

## Evaluate answers

Evaluation requires Azure OpenAI credentials and the `evaluation` dependencies.
Configure `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`,
`AZURE_OPENAI_API_VERSION`, `AZURE_OPENAI_DEPLOYMENT`, and
`AZURE_OPENAI_EMBEDDING_DEPLOYMENT` as required by your evaluator setup.

```bash
uv run --env-file .env --extra evaluation python scripts/evaluate_answer.py \
  --input output/answers.json \
  --output output/answer_evaluation.json \
  --judge-model gpt-4.1-mini
```

The evaluator reports RAGAS faithfulness and answer accuracy, plus deterministic
checks for numerical and yes/no answers. Retrieval evaluation is available through
`scripts/evaluate_retrieval.py --help`. Generated outputs stay under `output/`.
