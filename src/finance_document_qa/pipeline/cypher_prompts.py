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


TEXT2CYPHER_EXAMPLES_CALCULATION = [
    """User: What is the FY2019 fixed asset turnover ratio for Activision Blizzard? Fixed asset turnover ratio is defined as: FY2019 revenue / (average PP&E between FY2018 and FY2019). Round your answer to two decimal places.
Cypher:
MATCH (ch:Chunk {doc_id: "ACTIVISIONBLIZZARD_2019_10K.pdf"})-[:HAS_TABLE]->(:Table)-[:HAS_CELL]->(cell:TableCell)
WHERE (
        cell.row_header IN [
            "Net revenues",
            "Net revenue",
            "Total net revenues",
            "Revenues",
            "Revenue"
        ]
        AND cell.col_header CONTAINS "2019"
        AND toLower(ch.section_name) CONTAINS "statement of operations"
      )
   OR (
        cell.row_header IN [
            "Property and equipment, net",
            "Property and equipment",
            "Property, plant and equipment, net",
            "Property, plant and equipment"
        ]
        AND (cell.col_header CONTAINS "2019" OR cell.col_header CONTAINS "2018")
        AND (
            toLower(ch.section_name) CONTAINS "financial position"
            OR toLower(ch.section_name) CONTAINS "balance sheet"
        )
      )
RETURN ch.section_name AS section_name,
       cell.row_header AS row_header,
       cell.col_header AS col_header,
       cell.text AS text,
       cell.numeric_value AS numeric_value,
       cell.numeric_kind AS numeric_kind,
       cell.scale AS scale
ORDER BY row_header, col_header DESC""",
    """User: What is the year over year change in revenue for Amazon from FY2016 to FY2017? Answer as a percentage.
Cypher:
MATCH (ch:Chunk {doc_id: "AMAZON_2017_10K.pdf"})-[:HAS_TABLE]->(:Table)-[:HAS_CELL]->(cell:TableCell)
WHERE cell.row_header IN ["Net sales", "Total net sales", "Net revenues", "Total revenues", "Revenue"]
  AND (cell.col_header CONTAINS "2017" OR cell.col_header CONTAINS "2016")
  AND toLower(ch.section_name) CONTAINS "statements of operations"
RETURN ch.section_name AS section_name,
       cell.row_header AS row_header,
       cell.col_header AS col_header,
       cell.text AS text,
       cell.numeric_value AS numeric_value,
       cell.numeric_kind AS numeric_kind,
       cell.scale AS scale
ORDER BY cell.col_header DESC""",
    """User: What is the FY2015 operating cash flow ratio for Adobe? Operating cash flow ratio is defined as: cash from operations / total current liabilities. Round your answer to two decimal places.
Cypher:
MATCH (ch:Chunk {doc_id: "ADOBE_2015_10K.pdf"})-[:HAS_TABLE]->(:Table)-[:HAS_CELL]->(cell:TableCell)
WHERE (
        cell.row_header IN [
            "Net cash provided by operating activities",
            "Cash flow from operating activities",
            "Cash provided by operating activities"
        ]
        AND cell.col_header CONTAINS "2015"
        AND toLower(ch.section_name) CONTAINS "cash flows"
      )
   OR (
        cell.row_header IN [
            "Total current liabilities",
            "Current liabilities"
        ]
        AND cell.col_header CONTAINS "2015"
        AND (
            toLower(ch.section_name) CONTAINS "financial position"
            OR toLower(ch.section_name) CONTAINS "balance sheet"
        )
      )
RETURN ch.section_name AS section_name,
       cell.row_header AS row_header,
       cell.col_header AS col_header,
       cell.text AS text,
       cell.numeric_value AS numeric_value,
       cell.numeric_kind AS numeric_kind,
       cell.scale AS scale
ORDER BY row_header, col_header DESC""",
    """User: What is Adobe's year-over-year change in unadjusted operating income from FY2015 to FY2016? Round your answer to one decimal place and answer as a percentage.
Cypher:
MATCH (ch:Chunk {doc_id: "ADOBE_2016_10K.pdf"})-[:HAS_TABLE]->(:Table)-[:HAS_CELL]->(cell:TableCell)
WHERE cell.row_header IN [
    "Operating income",
    "Income from operations"
]
  AND (cell.col_header CONTAINS "2016" OR cell.col_header CONTAINS "2015")
  AND (
      toLower(ch.section_name) CONTAINS "statement of income"
      OR toLower(ch.section_name) CONTAINS "statement of operations"
  )
RETURN ch.section_name AS section_name,
       cell.row_header AS row_header,
       cell.col_header AS col_header,
       cell.text AS text,
       cell.numeric_value AS numeric_value,
       cell.numeric_kind AS numeric_kind,
       cell.scale AS scale
ORDER BY cell.col_header DESC""",
    """User: What was Block's revenue growth rate from FY2019 to FY2020? Answer as a percentage.
Cypher:
MATCH (ch:Chunk {doc_id: "BLOCK_2020_10K.pdf"})-[:HAS_TABLE]->(:Table)-[:HAS_CELL]->(cell:TableCell)
WHERE cell.row_header IN ["Total net revenue", "Total net revenues", "Net revenue", "Total revenue", "Total revenues", "Revenue", "Revenues"]
  AND (cell.col_header CONTAINS "2020" OR cell.col_header CONTAINS "2019")
  AND toLower(ch.section_name) CONTAINS "statement of operations"
RETURN ch.section_name AS section_name,
       cell.row_header AS row_header,
       cell.col_header AS col_header,
       cell.text AS text,
       cell.numeric_value AS numeric_value,
       cell.numeric_kind AS numeric_kind,
       cell.scale AS scale
ORDER BY cell.col_header DESC""",
    """User: What is the three-year average net profit margin for Best Buy over FY2015 to FY2017?
Cypher:
MATCH (ch:Chunk {doc_id: "BESTBUY_2017_10K.pdf"})-[:HAS_TABLE]->(:Table)-[:HAS_CELL]->(cell:TableCell)
WHERE cell.row_header IN ["Revenue", "Net earnings", "Net income"]
  AND (cell.col_header CONTAINS "2017" OR cell.col_header CONTAINS "2016" OR cell.col_header CONTAINS "2015")
  AND toLower(ch.section_name) CONTAINS "statement of earnings"
RETURN ch.section_name AS section_name,
       cell.row_header AS row_header,
       cell.col_header AS col_header,
       cell.text AS text,
       cell.numeric_value AS numeric_value,
       cell.numeric_kind AS numeric_kind,
       cell.scale AS scale
ORDER BY cell.row_header, cell.col_header DESC""",
    """User: What is PepsiCo's FY2022 unadjusted EBITDA less capital expenditure in USD millions?
Cypher:
MATCH (ch:Chunk {doc_id: "PEPSICO_2022_10K.pdf"})-[:HAS_TABLE]->(:Table)-[:HAS_CELL]->(cell:TableCell)
WHERE cell.row_header IN ["Net cash provided by operating activities", "Capital spending", "Capital expenditures"]
  AND cell.col_header CONTAINS "2022"
RETURN ch.section_name AS section_name,
       cell.row_header AS row_header,
       cell.col_header AS col_header,
       cell.text AS text,
       cell.numeric_value AS numeric_value,
       cell.numeric_kind AS numeric_kind,
       cell.scale AS scale
ORDER BY cell.row_header""",
]


TEXT2CYPHER_PROMPT_CALCULATION = """
Task: Generate a Cypher statement for answering a financial-report calculation question over a Neo4j graph.

Schema:
{schema}

Examples:
{examples}

Rules:
- Return only Cypher. No prose.
- Only generate read-only Cypher. Do not use CREATE, MERGE, SET, DELETE, DETACH, REMOVE, DROP, LOAD CSV, CALL apoc.*, or database-management procedures.
- Allowed procedure calls are limited to read-only index/search procedures such as db.index.vector.queryNodes and db.index.fulltext.queryNodes.
- This is a calculation question, so retrieve every input needed for the formula instead of a single final value.
- Prefer TableCell evidence over Chunk text for calculation inputs whenever possible.
- For ratios, margins, averages, YoY changes, turnover, DPO, CCC, or CAGR, fetch all required numerator/denominator rows and all required years/periods.
- For statement-constrained questions, prefer primary statement sections and include section filters when the question names the statement.
- Use row_header aliases when the same metric may appear under multiple common labels.
- Do not use LIMIT 1 unless the question truly needs only one input row.
- Return the minimal structured evidence needed to compute the metric:
  ch.section_name AS section_name,
  cell.row_header AS row_header,
  cell.col_header AS col_header,
  cell.text AS text,
  cell.numeric_value AS numeric_value,
  cell.numeric_kind AS numeric_kind,
  cell.scale AS scale
- Order rows so the downstream answer model can easily compute from them, usually by row_header and col_header.

Question:
{query_text}

Cypher query:
"""


TEXT2CYPHER_EXAMPLES_NOVEL = [
    """User: If we exclude the impact of M&A, which segment has dragged down 3M's overall growth in 2022?
Cypher:
MATCH (ch:Chunk)-[:HAS_TABLE]->(:Table)-[:HAS_CELL]->(cell:TableCell)
WHERE toLower(ch.doc_id) STARTS WITH "3m"
  AND (
        toLower(ch.section_name) CONTAINS "business segment"
        OR toLower(ch.section_name) CONTAINS "performance by business segment"
        OR toLower(ch.section_name) CONTAINS "segment"
      )
  AND (
        toLower(cell.row_header) CONTAINS "consumer"
        OR toLower(cell.row_header) CONTAINS "safety and industrial"
        OR toLower(cell.row_header) CONTAINS "transportation and electronics"
        OR toLower(cell.row_header) CONTAINS "health care"
        OR toLower(cell.row_header) CONTAINS "organic"
      )
  AND (
        cell.col_header CONTAINS "2022"
        OR toLower(cell.col_header) CONTAINS "organic"
        OR toLower(cell.col_header) CONTAINS "sales change"
      )
RETURN ch.section_name AS section_name,
       cell.row_header AS row_header,
       cell.col_header AS col_header,
       cell.text AS text,
       cell.numeric_value AS numeric_value,
       cell.numeric_kind AS numeric_kind,
       cell.scale AS scale
ORDER BY ch.section_name, cell.row_header, cell.col_header""",
    """User: What was the key agenda of Amcor's 8-K filing dated July 1, 2022?
Cypher:
MATCH (ch:Chunk)
WHERE toLower(ch.doc_id) STARTS WITH "amcor"
  AND (
        ch.doc_id CONTAINS "2022_8K"
        OR ch.doc_id CONTAINS "2022-07-01"
      )
  AND (
        toLower(ch.text) CONTAINS "agenda"
        OR toLower(ch.text) CONTAINS "meeting"
        OR toLower(ch.text) CONTAINS "shareholder"
        OR toLower(ch.text) CONTAINS "vote"
        OR toLower(ch.text) CONTAINS "proposal"
        OR toLower(ch.section_name) CONTAINS "item"
      )
RETURN ch.section_name AS section_name,
       null AS row_header,
       null AS col_header,
       ch.text AS text,
       null AS numeric_value,
       null AS numeric_kind,
       null AS scale
LIMIT 12""",
    """User: Was American Express able to retain card members during 2022?
Cypher:
MATCH (ch:Chunk)
WHERE toLower(ch.doc_id) STARTS WITH "americanexpress"
  AND (
        ch.text CONTAINS "2022"
        OR ch.section_name CONTAINS "2022"
      )
  AND (
        toLower(ch.text) CONTAINS "card member"
        OR toLower(ch.text) CONTAINS "cardmember"
        OR toLower(ch.text) CONTAINS "retention"
        OR toLower(ch.text) CONTAINS "retain"
        OR toLower(ch.text) CONTAINS "engagement"
      )
RETURN ch.section_name AS section_name,
       null AS row_header,
       null AS col_header,
       ch.text AS text,
       null AS numeric_value,
       null AS numeric_kind,
       null AS scale
LIMIT 12""",
    """User: Does Boeing's business subject to cyclicality?
Cypher:
MATCH (ch:Chunk)
WHERE toLower(ch.doc_id) STARTS WITH "boeing"
  AND (
        toLower(ch.section_name) CONTAINS "risk"
        OR toLower(ch.section_name) CONTAINS "business"
        OR toLower(ch.text) CONTAINS "cyclical"
        OR toLower(ch.text) CONTAINS "cyclicality"
        OR toLower(ch.text) CONTAINS "economic conditions"
        OR toLower(ch.text) CONTAINS "commercial aircraft"
      )
RETURN ch.section_name AS section_name,
       null AS row_header,
       null AS col_header,
       ch.text AS text,
       null AS numeric_value,
       null AS numeric_kind,
       null AS scale
LIMIT 12""",
    """User: Does Adobe have an improving free cash flow conversion as of FY2022?
Cypher:
MATCH (ch:Chunk)-[:HAS_TABLE]->(:Table)-[:HAS_CELL]->(cell:TableCell)
WHERE toLower(ch.doc_id) STARTS WITH "adobe"
  AND (
        toLower(cell.row_header) CONTAINS "net cash provided by operating activities"
        OR toLower(cell.row_header) CONTAINS "cash flow from operating activities"
        OR toLower(cell.row_header) CONTAINS "capital expenditures"
        OR toLower(cell.row_header) CONTAINS "net income"
        OR toLower(cell.row_header) CONTAINS "revenue"
      )
  AND (
        cell.col_header CONTAINS "2022"
        OR cell.col_header CONTAINS "2021"
        OR cell.col_header CONTAINS "2020"
      )
RETURN ch.section_name AS section_name,
       cell.row_header AS row_header,
       cell.col_header AS col_header,
       cell.text AS text,
       cell.numeric_value AS numeric_value,
       cell.numeric_kind AS numeric_kind,
       cell.scale AS scale
ORDER BY cell.row_header, cell.col_header DESC""",
]


TEXT2CYPHER_PROMPT_NOVEL = """
Task: Generate a Cypher statement for retrieving evidence for an open-ended FinanceBench question over a Neo4j graph.

Schema:
{schema}

Examples:
{examples}

Rules:
- Return only Cypher. No prose.
- Only generate read-only Cypher. Do not use CREATE, MERGE, SET, DELETE, DETACH, REMOVE, DROP, LOAD CSV, CALL apoc.*, or database-management procedures.
- Allowed procedure calls are limited to read-only index/search procedures such as db.index.vector.queryNodes and db.index.fulltext.queryNodes.
- This is an open-ended financial question: retrieve evidence, not just a final answer.
- Use constraints from the question only when present. If the question has a company, year, quarter, filing type, date, section, or topic, include those constraints. Do not invent missing constraints.
- Prefer same-company evidence using doc_id prefixes when a company is identifiable.
- If the question mentions a specific filing type or date, use doc_id/date text when available.
- For segment, trend, growth, retention, dividend, free cash flow, margin, percentage, or comparison questions, retrieve both relevant narrative chunks and table cells when possible.
- For explanatory questions, retrieve concise chunks from risk factors, business overview, MD&A, notes, or the named section.
- Return evidence in this shape when possible:
  ch.section_name AS section_name,
  cell.row_header AS row_header,
  cell.col_header AS col_header,
  cell.text AS text,
  cell.numeric_value AS numeric_value,
  cell.numeric_kind AS numeric_kind,
  cell.scale AS scale
- If retrieving narrative chunks without table cells, return null for row_header/col_header/numeric fields and ch.text AS text.
- Use LIMIT 12-30 depending on whether the question needs multiple evidence rows. Avoid LIMIT 1.

Question:
{query_text}

Cypher query:
"""
