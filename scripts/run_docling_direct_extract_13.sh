#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PDF_DIR="${ROOT_DIR}/dataset/financebench/pdfs"
OUT_DIR="${ROOT_DIR}/output/docling_direct_extract_13"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "${OUT_DIR}"

PDFS=(
  "3M_2018_10K.pdf"
  "AMAZON_2019_10K.pdf"
  "AMCOR_2020_10K.pdf"
  "AMERICANWATERWORKS_2020_10K.pdf"
  "AMERICANWATERWORKS_2021_10K.pdf"
  "BESTBUY_2019_10K.pdf"
  "BLOCK_2020_10K.pdf"
  "BOEING_2018_10K.pdf"
  "COSTCO_2021_10K.pdf"
  "MICROSOFT_2016_10K.pdf"
  "NETFLIX_2017_10K.pdf"
  "NIKE_2019_10K.pdf"
  "PEPSICO_2021_10K.pdf"
)

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
done

PYTHONPATH="${ROOT_DIR}/src" "${PYTHON_BIN}" -u -m neo4j_graphrag.preprocess.write_preprocessing_outputs_summary \
  --artifact 3m_chunks="${OUT_DIR}/3M_2018_10K_chunks.ndjson" \
  --artifact 3m_tables="${OUT_DIR}/3M_2018_10K_chunks_table_structured.ndjson" \
  --artifact amazon_chunks="${OUT_DIR}/AMAZON_2019_10K_chunks.ndjson" \
  --artifact amazon_tables="${OUT_DIR}/AMAZON_2019_10K_chunks_table_structured.ndjson" \
  --artifact amcor_chunks="${OUT_DIR}/AMCOR_2020_10K_chunks.ndjson" \
  --artifact amcor_tables="${OUT_DIR}/AMCOR_2020_10K_chunks_table_structured.ndjson" \
  --artifact aww2020_chunks="${OUT_DIR}/AMERICANWATERWORKS_2020_10K_chunks.ndjson" \
  --artifact aww2020_tables="${OUT_DIR}/AMERICANWATERWORKS_2020_10K_chunks_table_structured.ndjson" \
  --artifact aww2021_chunks="${OUT_DIR}/AMERICANWATERWORKS_2021_10K_chunks.ndjson" \
  --artifact aww2021_tables="${OUT_DIR}/AMERICANWATERWORKS_2021_10K_chunks_table_structured.ndjson" \
  --artifact bestbuy_chunks="${OUT_DIR}/BESTBUY_2019_10K_chunks.ndjson" \
  --artifact bestbuy_tables="${OUT_DIR}/BESTBUY_2019_10K_chunks_table_structured.ndjson" \
  --artifact block_chunks="${OUT_DIR}/BLOCK_2020_10K_chunks.ndjson" \
  --artifact block_tables="${OUT_DIR}/BLOCK_2020_10K_chunks_table_structured.ndjson" \
  --artifact boeing_chunks="${OUT_DIR}/BOEING_2018_10K_chunks.ndjson" \
  --artifact boeing_tables="${OUT_DIR}/BOEING_2018_10K_chunks_table_structured.ndjson" \
  --artifact costco_chunks="${OUT_DIR}/COSTCO_2021_10K_chunks.ndjson" \
  --artifact costco_tables="${OUT_DIR}/COSTCO_2021_10K_chunks_table_structured.ndjson" \
  --artifact microsoft_chunks="${OUT_DIR}/MICROSOFT_2016_10K_chunks.ndjson" \
  --artifact microsoft_tables="${OUT_DIR}/MICROSOFT_2016_10K_chunks_table_structured.ndjson" \
  --artifact netflix_chunks="${OUT_DIR}/NETFLIX_2017_10K_chunks.ndjson" \
  --artifact netflix_tables="${OUT_DIR}/NETFLIX_2017_10K_chunks_table_structured.ndjson" \
  --artifact nike_chunks="${OUT_DIR}/NIKE_2019_10K_chunks.ndjson" \
  --artifact nike_tables="${OUT_DIR}/NIKE_2019_10K_chunks_table_structured.ndjson" \
  --artifact pepsico_chunks="${OUT_DIR}/PEPSICO_2021_10K_chunks.ndjson" \
  --artifact pepsico_tables="${OUT_DIR}/PEPSICO_2021_10K_chunks_table_structured.ndjson" \
  --output "${OUT_DIR}/docling_direct_extract_13_manifest.json"

echo
echo "Done."
echo "Output directory: ${OUT_DIR}"
