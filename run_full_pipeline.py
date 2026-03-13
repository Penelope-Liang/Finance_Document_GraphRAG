from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _format_duration_minsec(seconds: float) -> str:
    total = max(0.0, float(seconds))
    hours = int(total // 3600)
    minutes = int((total % 3600) // 60)
    secs = total % 60
    if hours > 0:
        return f"{hours}h {minutes}min {secs:.1f}s"
    return f"{minutes}min {secs:.1f}s"


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required env var: {name}")
    return value


def _run_step(
    *,
    name: str,
    cmd: list[str],
    env: dict[str, str],
    log_file: Path,
    cwd: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    with log_file.open("a", encoding="utf-8") as log:
        log.write(f"\n[{_now_utc()}] STEP {name} START\n")
        log.write(f"CMD: {' '.join(cmd)}\n")
        log.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        elapsed = round(time.perf_counter() - started, 2)
        elapsed_human = _format_duration_minsec(elapsed)
        log.write(
            f"[{_now_utc()}] STEP {name} END "
            f"(exit={proc.returncode}, elapsed={elapsed_human})\n"
        )
        log.flush()
    if proc.returncode != 0:
        raise RuntimeError(
            f"Step failed: {name} (exit={proc.returncode}). See log: {log_file}"
        )
    return {
        "name": name,
        "exit_code": proc.returncode,
        "elapsed_seconds": elapsed,
        "elapsed": elapsed_human,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run full pipeline: preprocess.ingest_chunks -> build_graph -> "
            "dense_retrieval.faiss_builder -> query preflight -> query compare."
        )
    )
    parser.add_argument(
        "--input-path",
        default="dataset/financebench/pdfs",
        help="Input path for preprocess.ingest_chunks.",
    )
    parser.add_argument(
        "--unstructured-chunking",
        choices=["by_title", "basic", "item10k"],
        default="by_title",
        help="Chunking strategy for unstructured route.",
    )
    parser.add_argument(
        "--output-root",
        default="output",
        help="Root directory for run output.",
    )
    parser.add_argument(
        "--run-id",
        help="Optional run id. Default: UTC timestamp.",
    )
    parser.add_argument(
        "--questions-file",
        default="questions.txt",
        help="Questions file for compare query.",
    )
    parser.add_argument(
        "--embedder-backend",
        choices=["sentence-transformers", "ollama"],
        default="sentence-transformers",
        help="Embedder backend for dense_retrieval.faiss_builder.",
    )
    parser.add_argument(
        "--embedder-model",
        default="BAAI/bge-m3",
        help="Embedder model for dense_retrieval.faiss_builder.",
    )
    parser.add_argument(
        "--retrieval",
        choices=["vector", "hybrid"],
        default="hybrid",
        help="Retrieval mode for compare query.",
    )
    parser.add_argument(
        "--mode",
        choices=["rag", "graphrag", "compare"],
        default="compare",
        help="Mode for query.router.",
    )
    parser.add_argument(
        "--skip-query",
        action="store_true",
        help="Skip compare query step (only build output).",
    )
    parser.add_argument(
        "--question",
        help="Optional inline question. If set, ignores --questions-file.",
    )
    parser.add_argument(
        "--neo4j-uri",
        default=os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7687"),
        help="Neo4j URI.",
    )
    parser.add_argument(
        "--neo4j-user",
        default=os.environ.get("NEO4J_USER", "neo4j"),
        help="Neo4j user.",
    )
    parser.add_argument(
        "--build-entity-graph",
        action="store_true",
        help=(
            "Run entity/relationship extraction from chunks and import into Neo4j "
            "(Entity-RELATED-Entity + Entity-FROM_CHUNK-Chunk)."
        ),
    )
    parser.add_argument(
        "--entity-model",
        default=os.environ.get("OLLAMA_LLM_MODEL", "qwen2.5:72b"),
        help="Ollama model used for entity extraction.",
    )
    parser.add_argument(
        "--entity-types",
        default=(
            "company,organization,person,location,date,event,"
            "financial_metric,revenue,operating_profit,acquisition,ceo"
        ),
        help="Comma-separated entity types for extraction prompt.",
    )
    parser.add_argument(
        "--entity-ollama-host",
        default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"),
        help="Ollama host for entity extraction.",
    )
    parser.add_argument(
        "--entity-max-gleanings",
        type=int,
        default=1,
        help="Extra extraction passes per chunk.",
    )
    parser.add_argument(
        "--entity-limit",
        type=int,
        default=0,
        help="Limit number of chunks for extraction (0 = all).",
    )
    parser.add_argument(
        "--entity-progress-every",
        type=int,
        default=20,
        help="Print entity extraction progress every N chunks.",
    )
    parser.add_argument(
        "--entity-temperature",
        type=float,
        default=0.0,
        help="LLM temperature for entity extraction.",
    )
    parser.add_argument(
        "--entity-batch-size",
        type=int,
        default=200,
        help="Batch size for importing entity graph into Neo4j.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    repo_root = Path(__file__).resolve().parent
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = (repo_root / args.output_root / f"run_{run_id}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = (repo_root / "logs").resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / f"pipeline_{run_id}.log"

    neo4j_password = _require_env("NEO4J_PASSWORD")
    env = os.environ.copy()
    env["NEO4J_URI"] = args.neo4j_uri
    env["NEO4J_USER"] = args.neo4j_user
    env["NEO4J_PASSWORD"] = neo4j_password

    chunks_path = out_dir / "chunks.ndjson"
    graph_state_path = out_dir / "graph_state.json"
    faiss_index_path = out_dir / "chunks.index"
    faiss_mapping_path = out_dir / "intid_to_chunkid.jsonl"
    faiss_meta_path = out_dir / "faiss_meta.json"
    query_results_path = out_dir / "query_results_compare.json"
    entities_path = out_dir / "entities.ndjson"
    relationships_path = out_dir / "relationships.ndjson"

    py = sys.executable
    steps: list[dict[str, Any]] = []
    start = time.perf_counter()

    try:
        steps.append(
            _run_step(
                name="ingest_chunks",
                cmd=[
                    py, "-m",
                    "neo4j_graphrag.preprocess.ingest_chunks",
                    "--input-path",
                    args.input_path,
                    "--output",
                    str(chunks_path),
                    "--unstructured-chunking",
                    args.unstructured_chunking,
                ],
                env=env,
                log_file=log_file,
                cwd=repo_root,
            )
        )

        steps.append(
            _run_step(
                name="build_graph",
                cmd=[
                    py, "-m",
                    "neo4j_graphrag.tools.build_graph",
                    "--chunks-path",
                    str(chunks_path),
                    "--state-path",
                    str(graph_state_path),
                ],
                env=env,
                log_file=log_file,
                cwd=repo_root,
            )
        )

        if args.build_entity_graph:
            extraction_cmd = [
                py,
                "-m",
                "neo4j_graphrag.extraction.extractor",
                "--chunks-path",
                str(chunks_path),
                "--entities-out",
                str(entities_path),
                "--relationships-out",
                str(relationships_path),
                "--entity-types",
                args.entity_types,
                "--model",
                args.entity_model,
                "--ollama-host",
                args.entity_ollama_host,
                "--max-gleanings",
                str(args.entity_max_gleanings),
                "--limit",
                str(args.entity_limit),
                "--progress-every",
                str(args.entity_progress_every),
                "--temperature",
                str(args.entity_temperature),
            ]
            steps.append(
                _run_step(
                    name="extract_entities",
                    cmd=extraction_cmd,
                    env=env,
                    log_file=log_file,
                    cwd=repo_root,
                )
            )

            import_cmd = [
                py,
                "-m",
                "neo4j_graphrag.extraction.importer",
                "--entities-path",
                str(entities_path),
                "--relationships-path",
                str(relationships_path),
                "--batch-size",
                str(args.entity_batch_size),
                "--neo4j-uri",
                args.neo4j_uri,
                "--neo4j-user",
                args.neo4j_user,
            ]
            if args.neo4j_database:
                import_cmd.extend(["--neo4j-database", args.neo4j_database])

            steps.append(
                _run_step(
                    name="import_entity_graph",
                    cmd=import_cmd,
                    env=env,
                    log_file=log_file,
                    cwd=repo_root,
                )
            )

        steps.append(
            _run_step(
                name="build_faiss_index",
                cmd=[
                    py, "-m",
                    "neo4j_graphrag.dense_retrieval.faiss_builder",
                    "--chunks-path",
                    str(chunks_path),
                    "--index-out",
                    str(faiss_index_path),
                    "--mapping-out",
                    str(faiss_mapping_path),
                    "--meta-out",
                    str(faiss_meta_path),
                    "--embedder-backend",
                    args.embedder_backend,
                    "--embedder-model",
                    args.embedder_model,
                ],
                env=env,
                log_file=log_file,
                cwd=repo_root,
            )
        )

        steps.append(
            _run_step(
                name="preflight",
                cmd=[
                    py, "-m",
                    "neo4j_graphrag.query.router",
                    "--preflight-only",
                    "--mode",
                    "rag",
                    "--retrieval",
                    "vector",
                    "--dense-backend",
                    "faiss",
                    "--faiss-index-path",
                    str(faiss_index_path),
                    "--faiss-mapping-path",
                    str(faiss_mapping_path),
                ],
                env=env,
                log_file=log_file,
                cwd=repo_root,
            )
        )

        if not args.skip_query:
            resolved_questions_file: Path | None = None
            if not args.question:
                resolved_questions_file = Path(args.questions_file).expanduser()
                if not resolved_questions_file.is_absolute():
                    resolved_questions_file = (repo_root / resolved_questions_file).resolve()
                if not resolved_questions_file.exists():
                    raise FileNotFoundError(
                        "Query step requires question input. "
                        f"File not found: {resolved_questions_file}. "
                        "Provide --question or a valid --questions-file."
                    )
            query_cmd = [
                py, "-m",
                "neo4j_graphrag.query.router",
                "--mode",
                args.mode,
                "--retrieval",
                args.retrieval,
                "--dense-backend",
                "faiss",
                "--faiss-index-path",
                str(faiss_index_path),
                "--faiss-mapping-path",
                str(faiss_mapping_path),
                "--output",
                str(query_results_path),
            ]
            if args.question:
                query_cmd.extend(["-q", args.question])
            else:
                query_cmd.extend(
                    [
                        "--questions-file",
                        str(resolved_questions_file if resolved_questions_file else args.questions_file),
                    ]
                )

            steps.append(
                _run_step(
                    name="query_compare",
                    cmd=query_cmd,
                    env=env,
                    log_file=log_file,
                    cwd=repo_root,
                )
            )

        status = "success"
        error = None
    except Exception as exc:
        status = "failed"
        error = str(exc)
        raise
    finally:
        total_elapsed = round(time.perf_counter() - start, 2)
        total_elapsed_human = _format_duration_minsec(total_elapsed)
        summary = {
            "status": status,
            "error": error if status == "failed" else None,
            "started_at_utc": _now_utc(),
            "run_id": run_id,
            "out_dir": str(out_dir),
            "log_file": str(log_file),
            "input_path": args.input_path,
            "chunks_path": str(chunks_path),
            "graph_state_path": str(graph_state_path),
            "faiss_index_path": str(faiss_index_path),
            "faiss_mapping_path": str(faiss_mapping_path),
            "faiss_meta_path": str(faiss_meta_path),
            "query_results_path": str(query_results_path),
            "build_entity_graph": args.build_entity_graph,
            "entities_path": str(entities_path) if args.build_entity_graph else None,
            "relationships_path": (
                str(relationships_path) if args.build_entity_graph else None
            ),
            "total_elapsed_seconds": total_elapsed,
            "total_elapsed": total_elapsed_human,
            "steps": steps,
        }
        (out_dir / "run_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"status={status}")
        print(f"out_dir={out_dir}")
        print(f"log_file={log_file}")


if __name__ == "__main__":
    main()
