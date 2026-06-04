from __future__ import annotations


FINANCIAL_GRAPH_SCHEMA = """
Nodes:
- Chunk {chunk_id, doc_id, section_name, text, chunk_index, parser_engine, chunk_engine}
- Table {table_id, chunk_id, doc_id, section_name}
- TableCell {cell_id, chunk_id, doc_id, section_name, row_header, col_header, text, numeric_value, numeric_kind, scale, normalized_value}

Relationships:
- (:Chunk)-[:HAS_TABLE]->(:Table)
- (:Table)-[:HAS_CELL]->(:TableCell)

Important data conventions:
- Financial statement answers often live on TableCell nodes.
- Use doc_id to restrict to the target filing, e.g. "3M_2018_10K.pdf" or "AMAZON_2019_10K.pdf".
- Prefer primary statement sections over notes when ranking evidence:
  - balance sheet
  - statements of operations / income statement
  - cash flows / cash flow statement
- Typical table retrieval pattern:
  MATCH (ch:Chunk {doc_id: "...pdf"})-[:HAS_TABLE]->(:Table)-[:HAS_CELL]->(cell:TableCell)
  WHERE ...
  RETURN ch.section_name, cell.row_header, cell.col_header, cell.text, cell.numeric_value, cell.numeric_kind, cell.scale
- For direct extraction questions, favor exact row_header matches and exact year/period col_header matches.
- Use LIMIT 1 when the question asks for one final answer.
"""


TEXT2CYPHER_PROMPT = """
Task: Generate a Cypher statement for answering a financial-report question over a Neo4j graph.

Schema:
{schema}

Examples:
{examples}

Rules:
- Return only Cypher. No prose.
- Only generate read-only Cypher. Do not use CREATE, MERGE, SET, DELETE, DETACH, REMOVE, DROP, LOAD CSV, CALL apoc.*, or database-management procedures.
- Allowed procedure calls are limited to read-only index/search procedures such as db.index.vector.queryNodes and db.index.fulltext.queryNodes.
- Prefer TableCell evidence over Chunk text for metrics questions.
- Always filter to the target filing using doc_id when the filing can be inferred from the question.
- For financial statements, prefer primary statement sections over note sections when possible.
- Favor exact row_header matches first. Use col_header to match the requested fiscal year or period.
- Return the minimal evidence needed to answer the question.
- For direct extraction questions, return:
  ch.section_name AS section_name,
  cell.row_header AS row_header,
  cell.col_header AS col_header,
  cell.text AS text,
  cell.numeric_value AS numeric_value,
  cell.numeric_kind AS numeric_kind,
  cell.scale AS scale
- Use LIMIT 1 unless the question clearly requires multiple rows.

Question:
{query_text}

Cypher query:
"""


TEXT2CYPHER_EXAMPLES = [
    """User: What is the FY2018 capital expenditure amount (in USD millions) for 3M? Give a response to the question by relying on the details shown in the cash flow statement.
Cypher:
MATCH (ch:Chunk {doc_id: "3M_2018_10K.pdf"})-[:HAS_TABLE]->(:Table)-[:HAS_CELL]->(cell:TableCell)
WHERE cell.row_header = "Purchases of property, plant and equipment (PP&E)"
  AND cell.col_header CONTAINS "2018"
RETURN ch.section_name AS section_name,
       cell.row_header AS row_header,
       cell.col_header AS col_header,
       cell.text AS text,
       cell.numeric_value AS numeric_value,
       cell.numeric_kind AS numeric_kind,
       cell.scale AS scale
LIMIT 1""",
    """User: Assume that you are a public equities analyst. Answer the following question by primarily using information that is shown in the balance sheet: what is the year end FY2018 net PPNE for 3M? Answer in USD billions.
Cypher:
MATCH (ch:Chunk {doc_id: "3M_2018_10K.pdf"})-[:HAS_TABLE]->(:Table)-[:HAS_CELL]->(cell:TableCell)
WHERE replace(toLower(cell.row_header), " ", "") CONTAINS "property,plantandequipment-net"
  AND cell.col_header CONTAINS "2018"
RETURN ch.section_name AS section_name,
       cell.row_header AS row_header,
       cell.col_header AS col_header,
       cell.text AS text,
       cell.numeric_value AS numeric_value,
       cell.numeric_kind AS numeric_kind,
       cell.scale AS scale
LIMIT 1""",
    """User: By drawing conclusions from the information stated only in the income statement, what is Amazon's FY2019 net income attributable to shareholders (in USD millions)?
Cypher:
MATCH (ch:Chunk {doc_id: "AMAZON_2019_10K.pdf"})-[:HAS_TABLE]->(:Table)-[:HAS_CELL]->(cell:TableCell)
WHERE cell.row_header = "Net income"
  AND cell.col_header CONTAINS "2019"
  AND toLower(ch.section_name) CONTAINS "statements of operations"
RETURN ch.section_name AS section_name,
       cell.row_header AS row_header,
       cell.col_header AS col_header,
       cell.text AS text,
       cell.numeric_value AS numeric_value,
       cell.numeric_kind AS numeric_kind,
       cell.scale AS scale
LIMIT 1""",
]
