#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_DIR="${ROOT_DIR}/output/docling_direct_extract_13"
STATE_DIR="${ROOT_DIR}/output/graph_import/direct_extract_13"
PYTHON_BIN="${PYTHON_BIN:-python3}"

usage() {
  cat <<'EOF'
Usage:
  scripts/import_direct_extract_13_to_neo4j.sh \
    --neo4j-uri <uri> \
    --neo4j-user <user> \
    --neo4j-password <password> \
    [--neo4j-database <database>] \
    [--python-bin <python>] \
    [--input-dir <dir>]

This imports all *_chunks.ndjson and matching *_chunks_table_structured.ndjson
files from the direct-extraction batch folder into Neo4j.
EOF
}

NEO4J_URI=""
NEO4J_USER=""
NEO4J_PASSWORD=""
NEO4J_DATABASE="neo4j"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --neo4j-uri)
      NEO4J_URI="$2"
      shift 2
      ;;
    --neo4j-user)
      NEO4J_USER="$2"
      shift 2
      ;;
    --neo4j-password)
      NEO4J_PASSWORD="$2"
      shift 2
      ;;
    --neo4j-database)
      NEO4J_DATABASE="$2"
      shift 2
      ;;
    --python-bin)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --input-dir)
      INPUT_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${NEO4J_URI}" || -z "${NEO4J_USER}" || -z "${NEO4J_PASSWORD}" ]]; then
  echo "Missing required Neo4j credentials." >&2
  usage >&2
  exit 1
fi

if [[ ! -d "${INPUT_DIR}" ]]; then
  echo "Input directory not found: ${INPUT_DIR}" >&2
  exit 1
fi

mkdir -p "${STATE_DIR}"

mapfile -t CHUNK_FILES < <(find "${INPUT_DIR}" -maxdepth 1 -type f -name '*_chunks.ndjson' | sort)

if [[ ${#CHUNK_FILES[@]} -eq 0 ]]; then
  echo "No *_chunks.ndjson files found in ${INPUT_DIR}" >&2
  exit 1
fi

for chunks_path in "${CHUNK_FILES[@]}"; do
  base_name="$(basename "${chunks_path}" _chunks.ndjson)"
  tables_path="${INPUT_DIR}/${base_name}_chunks_table_structured.ndjson"
  state_path="${STATE_DIR}/${base_name}_state.json"

  if [[ ! -f "${tables_path}" ]]; then
    echo "[warn] missing table file for ${base_name}, skipping tables import" >&2
  fi

  echo
  echo "=== Importing ${base_name} ==="

  NEO4J_URI="${NEO4J_URI}" \
  NEO4J_USER="${NEO4J_USER}" \
  NEO4J_PASSWORD="${NEO4J_PASSWORD}" \
  NEO4J_DATABASE="${NEO4J_DATABASE}" \
  PYTHONPATH="${ROOT_DIR}/src" \
  "${PYTHON_BIN}" "${ROOT_DIR}/src/neo4j_graphrag/tools/build_graph.py" \
    --chunks-path "${chunks_path}" \
    --state-path "${state_path}" \
    --purge-doc-first

  if [[ -f "${tables_path}" ]]; then
    NEO4J_URI="${NEO4J_URI}" \
    NEO4J_USER="${NEO4J_USER}" \
    NEO4J_PASSWORD="${NEO4J_PASSWORD}" \
    NEO4J_DATABASE="${NEO4J_DATABASE}" \
    PYTHONPATH="${ROOT_DIR}/src" \
    "${PYTHON_BIN}" "${ROOT_DIR}/src/neo4j_graphrag/tools/import_tables.py" \
      --chunks-path "${tables_path}" \
      --cell-mode numeric \
      --purge-doc-first
  fi
done

echo
echo "Done."
echo "Imported files from: ${INPUT_DIR}"
