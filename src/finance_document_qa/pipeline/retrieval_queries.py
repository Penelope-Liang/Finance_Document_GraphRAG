from __future__ import annotations


CHUNK_RETRIEVAL_QUERY = """
    WITH node AS chunk, score
    WHERE ($doc_id IS NULL OR chunk.doc_id = $doc_id)
      AND ($doc_ids = [] OR chunk.doc_id IN $doc_ids)
      AND ($company_doc_key IS NULL OR toLower(chunk.doc_id) STARTS WITH $company_doc_key)
    OPTIONAL MATCH (chunk)-[:HAS_TABLE]->(t:Table)-[:HAS_CELL]->(cell:TableCell)
    WITH chunk, score,
         collect(DISTINCT {
             row_header: cell.row_header,
             col_header: cell.col_header,
             text: cell.text,
             numeric_value: cell.numeric_value,
             numeric_kind: cell.numeric_kind,
             scale: cell.scale,
         table_title: t.title,
         table_title_source: t.title_source,
         table_title_source_refs: t.title_source_refs,
         table_context_json: t.table_context_json,
         pages: cell.pages,
         source_refs: cell.source_refs,
         provenance_json: cell.provenance_json,
         provenance_scope: cell.provenance_scope,
         section_paths_json: cell.section_paths_json,
         pipeline_version: cell.pipeline_version,
         source_sha256: cell.source_sha256,
         artifact_id: cell.artifact_id,
         docling_version: cell.docling_version,
         docling_core_version: cell.docling_core_version
         })[0..20] AS table_cells
    RETURN chunk.text AS text,
           chunk.doc_id AS doc_id,
       chunk.pages AS pages,
       chunk.source_refs AS source_refs,
       chunk.provenance_json AS provenance_json,
       chunk.section_paths_json AS section_paths_json,
       chunk.pipeline_version AS pipeline_version,
       chunk.source_sha256 AS source_sha256,
       chunk.artifact_id AS artifact_id,
       chunk.docling_version AS docling_version,
       chunk.docling_core_version AS docling_core_version,
           chunk.section_name AS section_name,
           score AS similarity_score,
           table_cells,
           [] AS entities
    """
METRICS_TABLE_CELL_QUERY = """
UNWIND $targets AS target
CALL (target) {
    MATCH (ch:Chunk)-[:HAS_TABLE]->(t:Table)-[:HAS_CELL]->(cell:TableCell)
    WITH target, ch, t, cell,
         toLower(coalesce(cell.row_header, "")) AS row_text,
         toLower(coalesce(cell.col_header, "")) AS col_text,
         toLower(coalesce(t.section_name, ch.section_name, "")) AS section_text
    WHERE ($doc_id IS NULL OR ch.doc_id = $doc_id)
      AND ($doc_ids = [] OR ch.doc_id IN $doc_ids)
      AND ($company_doc_key IS NULL OR toLower(ch.doc_id) STARTS WITH $company_doc_key)
      AND any(alias IN target.aliases WHERE row_text CONTAINS alias)
      AND ($years = [] OR any(year IN $years WHERE col_text CONTAINS year))
    WITH target, ch, t, cell, row_text, col_text, section_text,
         CASE WHEN any(hint IN target.section_hints WHERE section_text CONTAINS hint) THEN 1 ELSE 0 END AS section_hit,
         CASE
           WHEN row_text IN target.aliases THEN 2
           WHEN any(alias IN target.aliases WHERE row_text CONTAINS alias) THEN 1
           ELSE 0
         END AS row_hit,
         CASE WHEN $years = [] OR any(year IN $years WHERE col_text CONTAINS year) THEN 1 ELSE 0 END AS year_hit
    RETURN target.name AS target_name,
           ch.doc_id AS doc_id,
           cell.pages AS pages,
           cell.source_refs AS source_refs,
           cell.provenance_json AS provenance_json,
           cell.provenance_scope AS provenance_scope,
           cell.section_paths_json AS section_paths_json,
           cell.pipeline_version AS pipeline_version,
           cell.source_sha256 AS source_sha256,
           cell.artifact_id AS artifact_id,
           cell.docling_version AS docling_version,
           cell.docling_core_version AS docling_core_version,
           t.title AS table_title,
           t.title_source AS table_title_source,
           t.title_source_refs AS table_title_source_refs,
           t.table_context_json AS table_context_json,
           t.section_name AS section_name,
           cell.row_header AS row_header,
           cell.col_header AS col_header,
           cell.text AS text,
           cell.numeric_value AS numeric_value,
           cell.numeric_kind AS numeric_kind,
           cell.scale AS scale,
           section_hit,
           row_hit,
           year_hit
    ORDER BY section_hit DESC, row_hit DESC, year_hit DESC, col_header DESC
    LIMIT $per_target_limit
}
RETURN target_name,
       table_title,
       table_title_source,
       table_title_source_refs,
       table_context_json,
       doc_id,
       pages,
       source_refs,
       provenance_json,
       provenance_scope,
       section_paths_json,
       pipeline_version,
       source_sha256,
       artifact_id,
       docling_version,
       docling_core_version,
       section_name,
       row_header,
       col_header,
       text,
       numeric_value,
       numeric_kind,
       scale,
       section_hit,
       row_hit,
       year_hit
ORDER BY target_name, section_hit DESC, row_hit DESC, year_hit DESC, col_header DESC
LIMIT $limit
"""
PRIMARY_STATEMENT_QUERY = """
MATCH (chunk:Chunk)
WHERE ($doc_id IS NULL OR chunk.doc_id = $doc_id)
  AND ($doc_ids = [] OR chunk.doc_id IN $doc_ids)
  AND ($company_doc_key IS NULL OR toLower(chunk.doc_id) STARTS WITH $company_doc_key)
  AND chunk.should_index = true
  AND (
    chunk.statement_type = $statement_type
    OR (
      chunk.section_name IS NOT NULL
      AND ANY(pat IN $includes WHERE toLower(chunk.section_name) CONTAINS pat)
      AND NONE(pat IN $excludes WHERE toLower(chunk.section_name) CONTAINS pat)
    )
  )
WITH chunk,
     (CASE WHEN chunk.statement_type = $statement_type THEN 2 ELSE 0 END
      + CASE WHEN chunk.section_name IS NOT NULL
                  AND ANY(pat IN $includes WHERE toLower(chunk.section_name) CONTAINS pat)
                  AND NONE(pat IN $excludes WHERE toLower(chunk.section_name) CONTAINS pat)
             THEN 1 ELSE 0 END) AS match_priority
OPTIONAL MATCH (chunk)-[:HAS_TABLE]->(t:Table)-[:HAS_CELL]->(cell:TableCell)
WITH chunk, match_priority,
     collect(DISTINCT {
         row_header: cell.row_header,
         col_header: cell.col_header,
         text: cell.text,
         numeric_value: cell.numeric_value,
         numeric_kind: cell.numeric_kind,
         scale: cell.scale,
         table_title: t.title,
         table_title_source: t.title_source,
         table_title_source_refs: t.title_source_refs,
         table_context_json: t.table_context_json,
         pages: cell.pages,
         source_refs: cell.source_refs,
         provenance_json: cell.provenance_json,
         provenance_scope: cell.provenance_scope,
         section_paths_json: cell.section_paths_json,
         pipeline_version: cell.pipeline_version,
         source_sha256: cell.source_sha256,
         artifact_id: cell.artifact_id,
         docling_version: cell.docling_version,
         docling_core_version: cell.docling_core_version
     })[0..40] AS table_cells
RETURN chunk.text AS text,
       chunk.doc_id AS doc_id,
       chunk.pages AS pages,
       chunk.source_refs AS source_refs,
       chunk.provenance_json AS provenance_json,
       chunk.section_paths_json AS section_paths_json,
       chunk.pipeline_version AS pipeline_version,
       chunk.source_sha256 AS source_sha256,
       chunk.artifact_id AS artifact_id,
       chunk.docling_version AS docling_version,
       chunk.docling_core_version AS docling_core_version,
       chunk.section_name AS section_name,
       chunk.statement_type AS chunk_statement_type,
       1.0 AS similarity_score,
       table_cells,
       [] AS entities,
       match_priority
ORDER BY match_priority DESC
LIMIT $limit
"""
