# Docling vs Unstructured Chunk Evaluation

This note compares two parser backends under the same legacy chunking pipeline.

## Setup

- Parser A: `Unstructured`
- Parser B: `Docling`
- Intermediate schema: `unstructured_compatible`
- Chunker: `legacy_chunk_from_elements`
- Retrieval setting: same downstream chunking/indexing configuration

## Files Evaluated

| PDF | Notes |
|---|---|
| `3M_2018_10K.pdf` | Baseline financial report with multi-header tables |
| `<second_pdf>.pdf` | Add one more representative report |

## Pipeline Variants

### A. Unstructured

```text
Unstructured parser
-> old elements.ndjson
-> chunk_from_elements.py
-> chunks.ndjson
```

### B. Docling

```text
Docling parser
-> CleanDocument
-> unstructured-compatible elements.ndjson export
-> chunk_from_elements.py
-> chunks.ndjson
```

## Output Metadata Convention

Each chunk should clearly record:

- `parser_engine`
- `metadata.intermediate_schema`
- `metadata.chunker`

Expected Docling chunk labels:

```json
{
  "parser_engine": "docling",
  "metadata": {
    "intermediate_schema": "unstructured_compatible",
    "chunker": "legacy_chunk_from_elements"
  }
}
```

## Quantitative Comparison

| Metric | Unstructured | Docling | Notes |
|---|---:|---:|---|
| Total chunks |  |  |  |
| Table chunks |  |  |  |
| Table chunk ratio |  |  |  |
| Avg chunk length (chars) |  |  |  |
| Avg table chunk length (chars) |  |  |  |
| Non-empty `table_html` ratio |  |  |  |
| Chunks with section name |  |  |  |
| Chunks with page metadata |  |  |  |

## Qualitative Checklist

Sample 20-30 chunks across:

- TOC / front matter tables
- Simple financial tables
- Complex multi-header financial tables
- Narrative sections with nearby tables

For each sampled chunk, check:

1. Was the table split unnaturally?
2. Did the caption/title stay with the table?
3. Was the heading preserved correctly?
4. Did lists remain in order?
5. Did cross-page content break awkwardly?
6. Is there noisy footer/header text?
7. Is the table HTML structurally usable?

## Retrieval Check

Use the same retrieval setup for both variants and compare:

1. Whether top-k retrieves the correct evidence chunk
2. Whether the retrieved table chunk is complete
3. Whether nearby explanatory text is easier to recover
4. Whether multi-header tables remain understandable after retrieval

## Preliminary Notes

### Docling

- Cleaner table structure on `Selected Financial Data`
- Cleaner multi-header business segment tables
- Still has light normalization issues such as:
  - `Dat a.`
  - `MNNRDresolution`
  - `22.4%$`

### Unstructured

- More noisy table structure on complex financial tables
- More frequent broken tokens and header/value misalignment

## Decision

- If Docling + legacy chunker is already clearly better:
  - keep the compatibility layer for now
  - move next to a `CleanDocument`-native chunker
- If not clearly better:
  - inspect export loss
  - inspect markdown-to-HTML conversion
  - inspect metadata mapping
