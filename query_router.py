from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Optional

import neo4j
from neo4j_graphrag.embeddings import OllamaEmbeddings
from neo4j_graphrag.llm import OllamaLLM
from neo4j_graphrag.rag import GraphRAG
from neo4j_graphrag.rag.prompts import RagTemplate
from neo4j_graphrag.retrievers import (
    HybridCypherRetriever,
    HybridRetriever,
    VectorCypherRetriever,
    VectorRetriever,
)
from neo4j_graphrag.types import RetrieverResultItem


warnings.filterwarnings("ignore", category=DeprecationWarning, module="neo4j_graphrag")

DEFAULT_CYPHER_RETRIEVAL_QUERY = """
WITH node AS chunk, score
OPTIONAL MATCH (chunk)<-[:FROM_CHUNK]-(entity)
WITH chunk, score, collect(DISTINCT {
    label: coalesce(entity.type, head(labels(entity))),
    name: coalesce(entity.title, entity.name, elementId(entity))
}) AS entities
RETURN chunk.text AS text,
       score AS similarity_score,
       coalesce(chunk.chunk_id, elementId(chunk)) AS chunk_id,
       entities
"""


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        print(f"Missing env var: {name}", file=sys.stderr)
        sys.exit(1)
    return value


def _load_questions(args: argparse.Namespace) -> list[str]:
    questions: list[str] = []
    if args.questions_file:
        with open(args.questions_file, "r", encoding="utf-8") as f:
            questions.extend(
                line.strip()
                for line in f
                if line.strip() and not line.lstrip().startswith("#")
            )
    if args.question:
        questions.extend(q.strip() for q in args.question if q.strip())
    if args.questions:
        questions.extend(q.strip() for q in args.questions if q.strip())
    return questions


def _read_text_file(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    return Path(path).read_text(encoding="utf-8")


def _resolve_mode_list(mode: str) -> list[str]:
    return ["rag", "graphrag"] if mode == "compare" else [mode]


def _node_result_formatter(record: neo4j.Record) -> RetrieverResultItem:
    node = record.get("node")
    score = record.get("score")
    metadata: dict[str, Any] = {"score": score}
    content = None

    if node is not None:
        chunk_id = node.get("chunk_id") or node.get("id")
        if chunk_id is not None:
            metadata["chunk_id"] = str(chunk_id)
        content = node.get("text")

    return RetrieverResultItem(
        content=content if content is not None else str(node) if node is not None else str(record),
        metadata=metadata,
    )


def _cypher_result_formatter(record: neo4j.Record) -> RetrieverResultItem:
    metadata: dict[str, Any] = {}
    for key in ("chunk_id", "similarity_score", "score", "entities", "form_id", "company_id"):
        value = record.get(key)
        if value is not None:
            metadata[key] = value

    if "score" not in metadata and "similarity_score" in metadata:
        metadata["score"] = metadata["similarity_score"]

    return RetrieverResultItem(
        content=record.get("text") or str(record),
        metadata=metadata or None,
    )


def _build_retriever(
    *,
    mode: str,
    retrieval: str,
    driver: neo4j.Driver,
    vector_index_name: str,
    fulltext_index_name: str,
    embedder: OllamaEmbeddings,
    retrieval_query: str,
    database: Optional[str],
):
    if mode == "rag":
        if retrieval == "vector":
            return VectorRetriever(
                driver=driver,
                index_name=vector_index_name,
                embedder=embedder,
                result_formatter=_node_result_formatter,
                neo4j_database=database,
            )
        return HybridRetriever(
            driver=driver,
            vector_index_name=vector_index_name,
            fulltext_index_name=fulltext_index_name,
            embedder=embedder,
            result_formatter=_node_result_formatter,
            neo4j_database=database,
        )

    if retrieval == "vector":
        return VectorCypherRetriever(
            driver=driver,
            index_name=vector_index_name,
            retrieval_query=retrieval_query,
            embedder=embedder,
            result_formatter=_cypher_result_formatter,
            neo4j_database=database,
        )
    return HybridCypherRetriever(
        driver=driver,
        vector_index_name=vector_index_name,
        fulltext_index_name=fulltext_index_name,
        retrieval_query=retrieval_query,
        embedder=embedder,
        result_formatter=_cypher_result_formatter,
        neo4j_database=database,
    )


def _extract_chunk_ids(items: list[RetrieverResultItem]) -> list[str]:
    ids: list[str] = []
    for item in items:
        metadata = item.metadata or {}
        chunk_id = (
            metadata.get("chunk_id")
            or metadata.get("id")
            or metadata.get("elementId")
            or metadata.get("element_id")
        )
        if chunk_id is None:
            continue
        value = str(chunk_id)
        if value not in ids:
            ids.append(value)
    return ids


def _build_context_preview(
    items: list[RetrieverResultItem], preview_items: int, preview_chars: int
) -> str:
    content = "\n---\n".join(str(item.content) for item in items[:preview_items])
    if len(content) > preview_chars:
        return f"{content[:preview_chars].rstrip()}..."
    return content


def _import_faiss():
    try:
        import faiss
    except ImportError as exc:
        raise ImportError(
            "Could not import faiss. Install with `uv pip install faiss-cpu` "
            "or install faiss-gpu on your GPU machine."
        ) from exc
    return faiss


def _import_numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError("Could not import numpy. Install with `uv pip install numpy`.") from exc
    return np


def _load_faiss_mapping(mapping_path: str) -> dict[int, dict[str, Any]]:
    mapping: dict[int, dict[str, Any]] = {}
    with open(mapping_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            payload = json.loads(raw)
            int_id = payload.get("int_id")
            if int_id is None:
                raise ValueError(f"Missing int_id in {mapping_path}:{line_no}")
            mapping[int(int_id)] = payload
    return mapping


def _neo4j_chunk_count(driver: neo4j.Driver, database: Optional[str]) -> int:
    result = driver.execute_query(
        "MATCH (c:Chunk) RETURN count(c) AS n",
        database_=database,
        routing_=neo4j.RoutingControl.READ,
    )
    if not result.records:
        return 0
    return int(result.records[0].get("n", 0) or 0)


def _neo4j_index_exists(
    driver: neo4j.Driver, *, index_name: str, index_type: str, database: Optional[str]
) -> bool:
    if index_type == "vector":
        query = """
        SHOW VECTOR INDEXES
        YIELD name
        WHERE name = $name
        RETURN count(*) > 0 AS exists
        """
    elif index_type == "fulltext":
        query = """
        SHOW FULLTEXT INDEXES
        YIELD name
        WHERE name = $name
        RETURN count(*) > 0 AS exists
        """
    else:
        raise ValueError(f"Unsupported index_type: {index_type}")

    result = driver.execute_query(
        query,
        {"name": index_name},
        database_=database,
        routing_=neo4j.RoutingControl.READ,
    )
    return bool(result.records and result.records[0].get("exists"))


def _count_chunk_ids_present(
    driver: neo4j.Driver, *, chunk_ids: list[str], database: Optional[str]
) -> int:
    if not chunk_ids:
        return 0
    result = driver.execute_query(
        """
        UNWIND $chunk_ids AS cid
        OPTIONAL MATCH (c:Chunk {chunk_id: cid})
        RETURN count(c) AS n
        """,
        {"chunk_ids": chunk_ids},
        database_=database,
        routing_=neo4j.RoutingControl.READ,
    )
    if not result.records:
        return 0
    return int(result.records[0].get("n", 0) or 0)


def _run_preflight_checks(
    *,
    args: argparse.Namespace,
    driver: neo4j.Driver,
    faiss_mapping: dict[int, dict[str, Any]],
) -> None:
    chunk_count = _neo4j_chunk_count(driver, args.neo4j_database)
    if chunk_count <= 0:
        raise RuntimeError(
            "Preflight failed: Neo4j has no :Chunk nodes. "
            "Run graph import first, then build FAISS, then query."
        )

    if args.retrieval == "hybrid":
        if not _neo4j_index_exists(
            driver,
            index_name=args.fulltext_index,
            index_type="fulltext",
            database=args.neo4j_database,
        ):
            raise RuntimeError(
                f"Preflight failed: fulltext index not found: {args.fulltext_index}"
            )

    if args.dense_backend == "neo4j":
        if not _neo4j_index_exists(
            driver,
            index_name=args.vector_index,
            index_type="vector",
            database=args.neo4j_database,
        ):
            raise RuntimeError(
                f"Preflight failed: vector index not found: {args.vector_index}"
            )
        print(
            f"[preflight] ok: chunk_count={chunk_count}, vector_index={args.vector_index}, "
            f"fulltext_index={'required' if args.retrieval == 'hybrid' else 'not-required'}"
        )
        return

    if not faiss_mapping:
        raise RuntimeError("Preflight failed: FAISS mapping is empty.")

    sample_chunk_ids: list[str] = []
    for row in faiss_mapping.values():
        chunk_id = row.get("chunk_id")
        if chunk_id is None:
            continue
        sample_chunk_ids.append(str(chunk_id))
        if len(sample_chunk_ids) >= 50:
            break
    if not sample_chunk_ids:
        raise RuntimeError("Preflight failed: FAISS mapping has no chunk_id entries.")

    present = _count_chunk_ids_present(
        driver, chunk_ids=sample_chunk_ids, database=args.neo4j_database
    )
    if present <= 0:
        raise RuntimeError(
            "Preflight failed: sample chunk_ids from FAISS mapping do not exist in Neo4j."
        )

    print(
        f"[preflight] ok: chunk_count={chunk_count}, faiss_ids={len(faiss_mapping)}, "
        f"faiss_sample_matched={present}/{len(sample_chunk_ids)}, "
        f"fulltext_index={'required' if args.retrieval == 'hybrid' else 'not-required'}"
    )


def _faiss_dense_search(
    *,
    index: Any,
    mapping: dict[int, dict[str, Any]],
    embedder: OllamaEmbeddings,
    query_text: str,
    top_k: int,
    effective_search_ratio: int,
    normalize: bool,
) -> list[dict[str, Any]]:
    faiss = _import_faiss()
    np = _import_numpy()

    query_vec = np.asarray(embedder.embed_query(query_text), dtype="float32").reshape(1, -1)
    if normalize:
        faiss.normalize_L2(query_vec)

    search_k = max(top_k, top_k * max(1, effective_search_ratio))
    scores, ids = index.search(query_vec, search_k)

    dense_hits: list[dict[str, Any]] = []
    seen: set[str] = set()
    for score, int_id in zip(scores[0], ids[0]):
        int_id = int(int_id)
        if int_id < 0:
            continue
        row = mapping.get(int_id, {})
        chunk_id = row.get("chunk_id")
        if not chunk_id:
            continue
        chunk_id_str = str(chunk_id)
        if chunk_id_str in seen:
            continue
        seen.add(chunk_id_str)
        dense_hits.append(
            {
                "chunk_id": chunk_id_str,
                "dense_score": float(score),
                "int_id": int_id,
                "sources": ["dense"],
            }
        )
    return dense_hits


def _neo4j_fulltext_search(
    *,
    driver: neo4j.Driver,
    fulltext_index_name: str,
    query_text: str,
    top_k: int,
    database: Optional[str],
) -> list[dict[str, Any]]:
    query = """
    CALL db.index.fulltext.queryNodes($index_name, $query_text) YIELD node, score
    RETURN coalesce(node.chunk_id, elementId(node)) AS chunk_id, score
    ORDER BY score DESC
    LIMIT $top_k
    """
    result = driver.execute_query(
        query,
        {
            "index_name": fulltext_index_name,
            "query_text": query_text,
            "top_k": top_k,
        },
        database_=database,
        routing_=neo4j.RoutingControl.READ,
    )
    hits: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in result.records:
        chunk_id = record.get("chunk_id")
        if chunk_id is None:
            continue
        chunk_id_str = str(chunk_id)
        if chunk_id_str in seen:
            continue
        seen.add(chunk_id_str)
        hits.append(
            {
                "chunk_id": chunk_id_str,
                "sparse_score": float(record.get("score", 0.0)),
                "sources": ["sparse"],
            }
        )
    return hits


def _fuse_hits_rrf(
    *,
    dense_hits: list[dict[str, Any]],
    sparse_hits: list[dict[str, Any]],
    top_k: int,
    rrf_k: int,
) -> list[dict[str, Any]]:
    fused: dict[str, dict[str, Any]] = {}
    for source_name, hits in (("dense", dense_hits), ("sparse", sparse_hits)):
        for rank, hit in enumerate(hits, start=1):
            chunk_id = hit["chunk_id"]
            if chunk_id not in fused:
                fused[chunk_id] = {
                    "chunk_id": chunk_id,
                    "dense_score": hit.get("dense_score"),
                    "sparse_score": hit.get("sparse_score"),
                    "fused_score": 0.0,
                    "sources": [],
                }
            row = fused[chunk_id]
            row["fused_score"] += 1.0 / (rrf_k + rank)
            if source_name not in row["sources"]:
                row["sources"].append(source_name)
            if hit.get("dense_score") is not None:
                row["dense_score"] = hit["dense_score"]
            if hit.get("sparse_score") is not None:
                row["sparse_score"] = hit["sparse_score"]

    return sorted(fused.values(), key=lambda x: x["fused_score"], reverse=True)[:top_k]


def _fuse_hits_linear(
    *,
    dense_hits: list[dict[str, Any]],
    sparse_hits: list[dict[str, Any]],
    top_k: int,
    alpha: float,
) -> list[dict[str, Any]]:
    def normalize(values: list[float]) -> tuple[float, float]:
        if not values:
            return (0.0, 0.0)
        return (min(values), max(values))

    dense_min, dense_max = normalize([x.get("dense_score", 0.0) for x in dense_hits])
    sparse_min, sparse_max = normalize([x.get("sparse_score", 0.0) for x in sparse_hits])

    def scaled(score: float, lower: float, upper: float) -> float:
        if upper <= lower:
            return 1.0 if score > 0 else 0.0
        return (score - lower) / (upper - lower)

    merged: dict[str, dict[str, Any]] = {}
    for hit in dense_hits:
        merged[hit["chunk_id"]] = {
            "chunk_id": hit["chunk_id"],
            "dense_score": hit.get("dense_score"),
            "sparse_score": None,
            "sources": ["dense"],
        }
    for hit in sparse_hits:
        row = merged.setdefault(
            hit["chunk_id"],
            {
                "chunk_id": hit["chunk_id"],
                "dense_score": None,
                "sparse_score": None,
                "sources": [],
            },
        )
        row["sparse_score"] = hit.get("sparse_score")
        if "sparse" not in row["sources"]:
            row["sources"].append("sparse")

    for row in merged.values():
        d = row.get("dense_score") or 0.0
        s = row.get("sparse_score") or 0.0
        d_norm = scaled(float(d), dense_min, dense_max)
        s_norm = scaled(float(s), sparse_min, sparse_max)
        row["fused_score"] = alpha * d_norm + (1 - alpha) * s_norm

    return sorted(merged.values(), key=lambda x: x["fused_score"], reverse=True)[:top_k]


def _fetch_faiss_context_items(
    *,
    driver: neo4j.Driver,
    chunk_ids: list[str],
    mode: str,
    score_by_chunk_id: dict[str, dict[str, Any]],
    database: Optional[str],
) -> list[RetrieverResultItem]:
    if not chunk_ids:
        return []

    if mode == "rag":
        query = """
        UNWIND range(0, size($chunk_ids)-1) AS i
        WITH i, $chunk_ids[i] AS cid
        OPTIONAL MATCH (c)
        WHERE c.chunk_id = cid OR elementId(c) = cid
        RETURN i,
               coalesce(c.chunk_id, elementId(c), cid) AS chunk_id,
               c.text AS text,
               c.section_name AS section_name,
               c.chunk_index AS chunk_index
        ORDER BY i
        """
    else:
        query = """
        UNWIND range(0, size($chunk_ids)-1) AS i
        WITH i, $chunk_ids[i] AS cid
        OPTIONAL MATCH (c)
        WHERE c.chunk_id = cid OR elementId(c) = cid
        OPTIONAL MATCH (prev)-[:NEXT]->(c)
        OPTIONAL MATCH (c)-[:NEXT]->(next)
        OPTIONAL MATCH (c)-[:IN_SECTION]->(sec)
        OPTIONAL MATCH (c)-[:IN_FORM]->(form_direct)
        OPTIONAL MATCH (form_from_section)-[:HAS_SECTION]->(sec)
        OPTIONAL MATCH (c)<-[:FROM_CHUNK]-(entity:Entity)
        WITH i, c, prev, next, sec, coalesce(form_direct, form_from_section) AS form,
             collect(DISTINCT coalesce(entity.title, entity.name, elementId(entity)))[0..20] AS entities
        OPTIONAL MATCH (company)-[:FILED]->(form)
        RETURN i,
               coalesce(c.chunk_id, elementId(c), $chunk_ids[i]) AS chunk_id,
               c.text AS text,
               prev.text AS prev_text,
               next.text AS next_text,
               coalesce(sec.section_name, sec.name, c.section_name) AS section_name,
               c.chunk_index AS chunk_index,
               entities,
               coalesce(form.form_id, elementId(form)) AS form_id,
               coalesce(company.company_id, elementId(company)) AS company_id
        ORDER BY i
        """

    result = driver.execute_query(
        query,
        {"chunk_ids": chunk_ids},
        database_=database,
        routing_=neo4j.RoutingControl.READ,
    )

    items: list[RetrieverResultItem] = []
    for record in result.records:
        text = record.get("text")
        if not text:
            continue
        chunk_id = str(record.get("chunk_id"))
        score_info = score_by_chunk_id.get(chunk_id, {})
        metadata: dict[str, Any] = {"chunk_id": chunk_id}
        metadata.update(score_info)
        if record.get("section_name") is not None:
            metadata["section_name"] = record.get("section_name")
        if record.get("chunk_index") is not None:
            metadata["chunk_index"] = record.get("chunk_index")

        if mode == "graphrag":
            if record.get("form_id") is not None:
                metadata["form_id"] = record.get("form_id")
            if record.get("company_id") is not None:
                metadata["company_id"] = record.get("company_id")
            if record.get("entities") is not None:
                metadata["entities"] = record.get("entities")
            prev_text = record.get("prev_text")
            next_text = record.get("next_text")
            entity_names = record.get("entities") or []
            parts: list[str] = []
            if prev_text:
                parts.append(f"[Previous]\n{prev_text}")
            if entity_names:
                parts.append("[Entities]\n" + ", ".join(str(x) for x in entity_names))
            parts.append(f"[Chunk]\n{text}")
            if next_text:
                parts.append(f"[Next]\n{next_text}")
            content = "\n\n".join(parts)
        else:
            content = str(text)

        items.append(RetrieverResultItem(content=content, metadata=metadata))

    return items


def _run_faiss_round(
    *,
    mode: str,
    retrieval: str,
    query_text: str,
    driver: neo4j.Driver,
    embedder: OllamaEmbeddings,
    llm: OllamaLLM,
    index: Any,
    mapping: dict[int, dict[str, Any]],
    database: Optional[str],
    fulltext_index_name: str,
    top_k: int,
    effective_search_ratio: int,
    hybrid_ranker: str,
    alpha: Optional[float],
    normalize: bool,
    rrf_k: int,
) -> tuple[str, str, list[RetrieverResultItem], list[str]]:
    dense_hits = _faiss_dense_search(
        index=index,
        mapping=mapping,
        embedder=embedder,
        query_text=query_text,
        top_k=top_k,
        effective_search_ratio=effective_search_ratio,
        normalize=normalize,
    )

    if retrieval == "vector":
        fused_hits = dense_hits[:top_k]
        retriever_name = "FaissVectorRetriever"
    else:
        sparse_hits = _neo4j_fulltext_search(
            driver=driver,
            fulltext_index_name=fulltext_index_name,
            query_text=query_text,
            top_k=max(top_k, top_k * max(1, effective_search_ratio)),
            database=database,
        )
        if hybrid_ranker == "linear":
            fused_hits = _fuse_hits_linear(
                dense_hits=dense_hits,
                sparse_hits=sparse_hits,
                top_k=top_k,
                alpha=alpha if alpha is not None else 0.5,
            )
            retriever_name = "FaissHybridLinearRetriever"
        else:
            fused_hits = _fuse_hits_rrf(
                dense_hits=dense_hits,
                sparse_hits=sparse_hits,
                top_k=top_k,
                rrf_k=rrf_k,
            )
            retriever_name = "FaissHybridRRFRetriever"

    selected_chunk_ids = [row["chunk_id"] for row in fused_hits]
    score_by_chunk_id = {row["chunk_id"]: row for row in fused_hits}
    items = _fetch_faiss_context_items(
        driver=driver,
        chunk_ids=selected_chunk_ids,
        mode=mode,
        score_by_chunk_id=score_by_chunk_id,
        database=database,
    )

    if not items:
        return ("No relevant context found.", retriever_name, items, selected_chunk_ids)

    prompt_template = RagTemplate()
    context = "\n".join(str(item.content) for item in items)
    prompt = prompt_template.format(query_text=query_text, context=context, examples="")
    llm_response = llm.invoke(
        input=prompt,
        system_instruction=prompt_template.system_instructions,
    )
    return (llm_response.content, retriever_name, items, selected_chunk_ids)


def _save_results(
    output_path: str,
    output_format: str,
    rows: list[dict[str, Any]],
) -> None:
    if output_format == "json":
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)
        return

    if output_format == "jsonl":
        with open(output_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return

    headers = [
        "mode",
        "retrieval",
        "question",
        "answer",
        "latency_ms",
        "retriever",
        "retrieved_chunk_ids",
        "context_preview",
        "error",
    ]
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            csv_row = row.copy()
            csv_row["retrieved_chunk_ids"] = "|".join(row["retrieved_chunk_ids"])
            writer.writerow(csv_row)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified RAG/GraphRAG query runner with selectable retriever mode."
    )
    parser.add_argument("questions", nargs="*", help="Inline question text")
    parser.add_argument(
        "-q",
        "--question",
        action="append",
        help="Inline question (can be repeated)",
    )
    parser.add_argument(
        "--questions-file",
        help="File containing questions (one per line, '#' for comments)",
    )
    parser.add_argument(
        "--mode",
        choices=["rag", "graphrag", "compare"],
        default="graphrag",
        help="Run plain RAG, GraphRAG, or both (compare).",
    )
    parser.add_argument(
        "--retrieval",
        choices=["vector", "hybrid"],
        default="hybrid",
        help="Retriever family used in the selected mode(s).",
    )
    parser.add_argument(
        "--dense-backend",
        choices=["neo4j", "faiss"],
        default="neo4j",
        help="Dense retrieval backend. neo4j uses vector index, faiss uses local FAISS index.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Top K to retrieve.")
    parser.add_argument(
        "--effective-search-ratio",
        type=int,
        default=1,
        help="Multiplier for vector candidate pool size.",
    )
    parser.add_argument(
        "--hybrid-ranker",
        choices=["naive", "linear"],
        default="naive",
        help="Hybrid ranker (only applies when --retrieval=hybrid).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        help="Weight for vector score when --hybrid-ranker=linear (0..1).",
    )
    parser.add_argument(
        "--faiss-index-path",
        default=os.environ.get("FAISS_INDEX_PATH", "artifacts/faiss/chunks.index"),
        help="FAISS index path when --dense-backend=faiss.",
    )
    parser.add_argument(
        "--faiss-mapping-path",
        default=os.environ.get(
            "FAISS_MAPPING_PATH", "artifacts/faiss/intid_to_chunkid.jsonl"
        ),
        help="FAISS int_id->chunk_id mapping path when --dense-backend=faiss.",
    )
    parser.add_argument(
        "--faiss-no-normalize",
        action="store_true",
        help="Disable L2 normalization for query vectors in FAISS mode.",
    )
    parser.add_argument(
        "--faiss-rrf-k",
        type=int,
        default=60,
        help="RRF k constant for --dense-backend=faiss --retrieval=hybrid --hybrid-ranker=naive.",
    )
    parser.add_argument(
        "--output",
        default="query_results.json",
        help="Output path for structured run results.",
    )
    parser.add_argument(
        "--output-format",
        choices=["json", "jsonl", "csv"],
        default="json",
        help="Output file format.",
    )
    parser.add_argument(
        "--preview-items",
        type=int,
        default=3,
        help="Number of retrieved items included in context preview.",
    )
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=500,
        help="Max character length of context preview.",
    )
    parser.add_argument(
        "--retrieval-query",
        help="Cypher retrieval query used in GraphRAG mode.",
    )
    parser.add_argument(
        "--retrieval-query-file",
        help="Read retrieval query from a file path.",
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
        "--neo4j-database",
        default=os.environ.get("NEO4J_DATABASE"),
        help="Neo4j database name (optional).",
    )
    parser.add_argument(
        "--vector-index",
        default=os.environ.get("NEO4J_VECTOR_INDEX", "chunk_embedding_index"),
        help="Neo4j vector index name.",
    )
    parser.add_argument(
        "--fulltext-index",
        default=os.environ.get("NEO4J_FULLTEXT_INDEX", "chunk_fulltext_index"),
        help="Neo4j fulltext index name.",
    )
    parser.add_argument(
        "--llm-model",
        default=os.environ.get("OLLAMA_LLM_MODEL", "qwen2.5:72b"),
        help="LLM model name for Ollama.",
    )
    parser.add_argument(
        "--embedder-model",
        default=os.environ.get("OLLAMA_EMBEDDER_MODEL", "bge-m3:latest"),
        help="Embedding model name for Ollama.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(os.environ.get("OLLAMA_TEMPERATURE", "0")),
        help="LLM temperature.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Only run readiness checks (Neo4j + indexes + FAISS mapping) and exit.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    questions = _load_questions(args)
    if not questions and not args.preflight_only:
        print("No questions provided. Use --questions-file or -q.", file=sys.stderr)
        sys.exit(1)

    if not args.preflight_only and args.hybrid_ranker == "linear" and args.alpha is None:
        print("--alpha is required when --hybrid-ranker=linear", file=sys.stderr)
        sys.exit(1)

    if args.retrieval_query and args.retrieval_query_file:
        print("Use either --retrieval-query or --retrieval-query-file, not both.", file=sys.stderr)
        sys.exit(1)

    password = _require_env("NEO4J_PASSWORD")

    file_query = _read_text_file(args.retrieval_query_file)
    retrieval_query = args.retrieval_query or file_query or DEFAULT_CYPHER_RETRIEVAL_QUERY

    modes = _resolve_mode_list(args.mode)
    run_rows: list[dict[str, Any]] = []
    llm: Optional[OllamaLLM] = None
    embedder: Optional[OllamaEmbeddings] = None

    try:
        faiss_index = None
        faiss_mapping: dict[int, dict[str, Any]] = {}
        if args.dense_backend == "faiss":
            faiss = _import_faiss()
            if not Path(args.faiss_index_path).exists():
                print(f"FAISS index not found: {args.faiss_index_path}", file=sys.stderr)
                sys.exit(1)
            if not Path(args.faiss_mapping_path).exists():
                print(f"FAISS mapping not found: {args.faiss_mapping_path}", file=sys.stderr)
                sys.exit(1)
            faiss_index = faiss.read_index(args.faiss_index_path)
            faiss_mapping = _load_faiss_mapping(args.faiss_mapping_path)

        with neo4j.GraphDatabase.driver(
            args.neo4j_uri, auth=(args.neo4j_user, password)
        ) as driver:
            _run_preflight_checks(args=args, driver=driver, faiss_mapping=faiss_mapping)
            if args.preflight_only:
                print("Preflight passed. Exiting without running queries.")
                return

            llm = OllamaLLM(
                model_name=args.llm_model,
                model_params={"options": {"temperature": args.temperature}},
            )
            embedder = OllamaEmbeddings(model=args.embedder_model)

            rag_by_mode: dict[str, GraphRAG] = {}
            if args.dense_backend == "neo4j":
                for mode in modes:
                    retriever = _build_retriever(
                        mode=mode,
                        retrieval=args.retrieval,
                        driver=driver,
                        vector_index_name=args.vector_index,
                        fulltext_index_name=args.fulltext_index,
                        embedder=embedder,
                        retrieval_query=retrieval_query,
                        database=args.neo4j_database,
                    )
                    rag_by_mode[mode] = GraphRAG(retriever=retriever, llm=llm)

            total_runs = len(questions) * len(modes)
            done = 0
            for question in questions:
                for mode in modes:
                    done += 1
                    print(
                        f"[{done}/{total_runs}] mode={mode} retrieval={args.retrieval} "
                        f"dense_backend={args.dense_backend} | {question}"
                    )
                    retriever_config: dict[str, Any] = {
                        "top_k": args.top_k,
                        "effective_search_ratio": args.effective_search_ratio,
                    }
                    if args.retrieval == "hybrid":
                        retriever_config["ranker"] = args.hybrid_ranker
                        if args.hybrid_ranker == "linear":
                            retriever_config["alpha"] = args.alpha

                    start = time.perf_counter()
                    answer = None
                    error = None
                    context_preview = ""
                    chunk_ids: list[str] = []
                    retriever_name = ""
                    try:
                        if args.dense_backend == "neo4j":
                            result = rag_by_mode[mode].search(
                                query_text=question,
                                retriever_config=retriever_config,
                                return_context=True,
                            )
                            answer = result.answer
                            retriever_result = result.retriever_result
                            retriever_name = (
                                (retriever_result.metadata or {}).get("__retriever", "")
                                if retriever_result
                                else ""
                            )
                            items = retriever_result.items if retriever_result else []
                            chunk_ids = _extract_chunk_ids(items)
                            context_preview = _build_context_preview(
                                items, args.preview_items, args.preview_chars
                            )
                        else:
                            answer, retriever_name, items, selected_ids = _run_faiss_round(
                                mode=mode,
                                retrieval=args.retrieval,
                                query_text=question,
                                driver=driver,
                                embedder=embedder,
                                llm=llm,
                                index=faiss_index,
                                mapping=faiss_mapping,
                                database=args.neo4j_database,
                                fulltext_index_name=args.fulltext_index,
                                top_k=args.top_k,
                                effective_search_ratio=args.effective_search_ratio,
                                hybrid_ranker=args.hybrid_ranker,
                                alpha=args.alpha,
                                normalize=not args.faiss_no_normalize,
                                rrf_k=args.faiss_rrf_k,
                            )
                            chunk_ids = selected_ids
                            context_preview = _build_context_preview(
                                items, args.preview_items, args.preview_chars
                            )

                        print(
                            f"  answer: {answer[:120]}..."
                            if answer is not None and len(answer) > 120
                            else f"  answer: {answer}"
                        )
                    except Exception as exc:
                        error = str(exc)
                        print(f"  error: {error}")

                    latency_ms = round((time.perf_counter() - start) * 1000, 2)
                    run_rows.append(
                        {
                            "mode": mode,
                            "retrieval": args.retrieval,
                            "question": question,
                            "answer": answer,
                            "latency_ms": latency_ms,
                            "retriever": retriever_name,
                            "retrieved_chunk_ids": chunk_ids,
                            "context_preview": context_preview,
                            "error": error,
                        }
                    )
                    print(f"  latency_ms: {latency_ms}")

    finally:
        if llm is not None and hasattr(llm, "client") and hasattr(llm.client, "_client"):
            llm.client._client.close()
        if (
            embedder is not None
            and hasattr(embedder, "client")
            and hasattr(embedder.client, "_client")
        ):
            embedder.client._client.close()

    _save_results(args.output, args.output_format, run_rows)
    print(f"Saved structured results to {args.output}")


if __name__ == "__main__":
    main()
