#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PDF_DIR="${ROOT_DIR}/dataset/financebench/pdfs"
OUT_DIR="${ROOT_DIR}/output/docling_metrics_remaining_37"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "${OUT_DIR}"

PDFS=(
  "ACTIVISIONBLIZZARD_2019_10K.pdf"
  "ADOBE_2015_10K.pdf"
  "ADOBE_2016_10K.pdf"
  "ADOBE_2017_10K.pdf"
  "AES_2022_10K.pdf"
  "AMAZON_2017_10K.pdf"
  "AMD_2015_10K.pdf"
  "AMERICANWATERWORKS_2021_10K.pdf"
  "BESTBUY_2017_10K.pdf"
  "BLOCK_2016_10K.pdf"
  "BLOCK_2020_10K.pdf"
  "COCACOLA_2017_10K.pdf"
  "COCACOLA_2021_10K.pdf"
  "COCACOLA_2022_10K.pdf"
  "CORNING_2020_10K.pdf"
  "CORNING_2021_10K.pdf"
  "CVSHEALTH_2018_10K.pdf"
  "GENERALMILLS_2019_10K.pdf"
  "GENERALMILLS_2020_10K.pdf"
  "GENERALMILLS_2022_10K.pdf"
  "KRAFTHEINZ_2019_10K.pdf"
  "LOCKHEEDMARTIN_2020_10K.pdf"
  "LOCKHEEDMARTIN_2021_10K.pdf"
  "LOCKHEEDMARTIN_2022_10K.pdf"
  "MGMRESORTS_2018_10K.pdf"
  "MGMRESORTS_2020_10K.pdf"
  "NETFLIX_2015_10K.pdf"
  "NIKE_2018_10K.pdf"
  "NIKE_2021_10K.pdf"
  "PEPSICO_2022_10K.pdf"
  "WALMART_2018_10K.pdf"
  "WALMART_2019_10K.pdf"
  "WALMART_2020_10K.pdf"
)

artifact_args=()

for pdf in "${PDFS[@]}"; do
  stem="${pdf%.pdf}"
  input_path="${PDF_DIR}/${pdf}"
  clean_out="${OUT_DIR}/${stem}_clean.jsonl"
  elements_out="${OUT_DIR}/${stem}_elements.ndjson"
  chunks_out="${OUT_DIR}/${stem}_chunks.ndjson"
  tables_out="${OUT_DIR}/${stem}_chunks_table_structured.ndjson"

  echo
  echo "=== ${pdf} ==="

  PYTHONPATH="${ROOT_DIR}/src" "${PYTHON_BIN}" -u -m neo4j_graphrag.preprocess.docling_adapter \
    --input-path "${input_path}" \
    --clean-output "${clean_out}" \
    --elements-output "${elements_out}"

  PYTHONPATH="${ROOT_DIR}/src" "${PYTHON_BIN}" -u -m neo4j_graphrag.preprocess.chunk_from_elements \
    --elements-path "${elements_out}" \
    --output "${chunks_out}"

  PYTHONPATH="${ROOT_DIR}/src" "${PYTHON_BIN}" -u -m neo4j_graphrag.preprocess.table_graph_prep \
    --chunks-path "${chunks_out}" \
    --output "${tables_out}"

  artifact_args+=(--artifact "${stem}_chunks=${chunks_out}")
  artifact_args+=(--artifact "${stem}_tables=${tables_out}")
done

PYTHONPATH="${ROOT_DIR}/src" "${PYTHON_BIN}" -u -m neo4j_graphrag.preprocess.write_preprocessing_outputs_summary \
  "${artifact_args[@]}" \
  --output "${OUT_DIR}/docling_metrics_remaining_37_manifest.json"

echo
echo "Done."
echo "Output directory: ${OUT_DIR}"
