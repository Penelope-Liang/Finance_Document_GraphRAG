from __future__ import annotations


CHUNK_RETRIEVAL_QUERY = """
    WITH node AS chunk, score
    WHERE ($doc_id IS NULL OR chunk.doc_id = $doc_id)
      AND ($doc_ids = [] OR chunk.doc_id IN $doc_ids)
      AND ($company_doc_key IS NULL OR toLower(chunk.doc_id) STARTS WITH $company_doc_key)
    OPTIONAL MATCH (chunk)-[:HAS_TABLE]->(:Table)-[:HAS_CELL]->(cell:TableCell)
    OPTIONAL MATCH (chunk)<-[:FROM_CHUNK]-(entity)
    WITH chunk, score,
         collect(DISTINCT {
             row_header: cell.row_header,
             col_header: cell.col_header,
             text: cell.text,
             numeric_value: cell.numeric_value,
             numeric_kind: cell.numeric_kind,
             scale: cell.scale
         })[0..20] AS table_cells,
         collect(DISTINCT {
             label: coalesce(entity.type, head(labels(entity))),
             name: coalesce(entity.title, entity.name, elementId(entity))
         }) AS entities
    RETURN chunk.text AS text,
           chunk.doc_id AS doc_id,
           chunk.section_name AS section_name,
           score AS similarity_score,
           table_cells,
           entities
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
       doc_id,
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
OPTIONAL MATCH (chunk)-[:HAS_TABLE]->(:Table)-[:HAS_CELL]->(cell:TableCell)
WITH chunk, match_priority,
     collect(DISTINCT {
         row_header: cell.row_header,
         col_header: cell.col_header,
         text: cell.text,
         numeric_value: cell.numeric_value,
         numeric_kind: cell.numeric_kind,
         scale: cell.scale
     })[0..40] AS table_cells
RETURN chunk.text AS text,
       chunk.doc_id AS doc_id,
       chunk.section_name AS section_name,
       chunk.statement_type AS chunk_statement_type,
       1.0 AS similarity_score,
       table_cells,
       [] AS entities,
       match_priority
ORDER BY match_priority DESC
LIMIT $limit
"""
