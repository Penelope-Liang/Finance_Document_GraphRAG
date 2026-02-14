"""
Prompt Templates for Medical Knowledge Graph Building and RAG

This file contains all customizable prompts used in the pipeline.
Import and use these prompts in build_kg.py and eval_rag.py.
"""

from neo4j_graphrag.rag.prompts import ERExtractionTemplate, RagTemplate, PromptTemplate


# =============================================================================
# ENTITY EXTRACTION PROMPTS (for KG building)
# =============================================================================

# Default extraction prompt (from neo4j-graphrag library)
DEFAULT_EXTRACTION_TEMPLATE = ERExtractionTemplate()

# Medical-specific extraction prompt
MEDICAL_EXTRACTION_TEMPLATE = ERExtractionTemplate(
    template="""
You are a medical knowledge extraction expert specializing in oncology and dermatology.
Your task is to extract medical entities and their relationships from clinical text.

Extract the entities (nodes) and specify their type from the following text.
Also extract the relationships between these nodes.

Return result as JSON using the following format:
{{"nodes": [ {{"id": "0", "label": "Disease", "properties": {{"name": "melanoma"}} }}],
"relationships": [{{"type": "AFFECTS", "start_node_id": "0", "end_node_id": "1", "properties": {{}} }}] }}

Use only the following node and relationship types (if provided):
{schema}

EXTRACTION GUIDELINES:
1. Focus on medical accuracy - only extract clearly stated facts
2. Capture disease-symptom, disease-treatment, and risk factor relationships
3. Include anatomical locations and cell types when mentioned
4. Preserve medical terminology exactly as written
5. Extract staging information (e.g., Stage I, II, III, IV)
6. Identify diagnostic tests and procedures

Assign a unique ID (string) to each node, and reuse it to define relationships.
Do respect the source and target node types for relationship and
the relationship direction.

Make sure you adhere to the following rules to produce valid JSON objects:
- Do not return any additional information other than the JSON in it.
- Omit any backticks around the JSON - simply output the JSON on its own.
- The JSON object must not wrapped into a list - it is its own JSON object.
- Property names must be enclosed in double quotes

Examples:
{examples}

Input text:

{text}
"""
)

# Few-shot examples for medical extraction
MEDICAL_EXTRACTION_EXAMPLES = """
Example 1:
Input: "Melanoma is a type of skin cancer that develops from melanocytes. UV radiation exposure is a major risk factor."
Output: {{"nodes": [
    {{"id": "0", "label": "Disease", "properties": {{"name": "Melanoma"}}}},
    {{"id": "1", "label": "Disease", "properties": {{"name": "skin cancer"}}}},
    {{"id": "2", "label": "Cell", "properties": {{"name": "melanocytes"}}}},
    {{"id": "3", "label": "RiskFactor", "properties": {{"name": "UV radiation exposure"}}}}
], "relationships": [
    {{"type": "TYPE_OF", "start_node_id": "0", "end_node_id": "1", "properties": {{}}}},
    {{"type": "DEVELOPS_FROM", "start_node_id": "0", "end_node_id": "2", "properties": {{}}}},
    {{"type": "RISK_FOR", "start_node_id": "3", "end_node_id": "0", "properties": {{}}}}
]}}

Example 2:
Input: "Basal cell carcinoma is treated with surgical excision. The 5-year survival rate is over 99%."
Output: {{"nodes": [
    {{"id": "0", "label": "Disease", "properties": {{"name": "Basal cell carcinoma"}}}},
    {{"id": "1", "label": "Treatment", "properties": {{"name": "surgical excision"}}}}
], "relationships": [
    {{"type": "TREATS", "start_node_id": "1", "end_node_id": "0", "properties": {{}}}}
]}}
"""


# =============================================================================
# RAG PROMPTS (for question answering)
# =============================================================================

# Default RAG prompt (from neo4j-graphrag library)
DEFAULT_RAG_TEMPLATE = RagTemplate()

# Medical QA prompt - concise answers
MEDICAL_RAG_TEMPLATE = RagTemplate(
    template="""You are a medical question answering assistant.

Based on the provided context, answer the question accurately and concisely.

Rules:
1. If the context contains the answer, cite the relevant evidence.
2. If the context is partially relevant, use it and clearly state what is supported vs. inferred.
3. If the context does not contain relevant information, say "The provided context does not contain sufficient information to answer this question."
4. Do not guess or use information not present in the context.

Context:
{context}

Question:
{query_text}
{examples}
Answer:
""",
    expected_inputs=["context", "query_text", "examples"],
)

# Medical QA prompt - detailed with evidence
MEDICAL_RAG_DETAILED_TEMPLATE = RagTemplate(
    template="""You are a medical expert assistant providing evidence-based answers.

Based on the provided context from medical literature, answer the question thoroughly.

Guidelines:
1. Provide a clear, direct answer first
2. Support your answer with specific evidence from the context
3. If multiple relevant facts exist, synthesize them coherently
4. Use medical terminology accurately
5. If the context is insufficient, clearly state the limitations

Context:
{context}

Question:
{query_text}
{examples}

Provide your answer in the following format:
Answer: [Your direct answer]
Evidence: [Supporting quotes or facts from context]
Confidence: [High/Medium/Low based on context quality]
""",
    expected_inputs=["context", "query_text", "examples"],
)

# Medical QA prompt - JSON output format
MEDICAL_RAG_JSON_TEMPLATE = RagTemplate(
    template="""You are a medical question answering assistant.

Answer the question using ONLY the provided context.

Context:
{context}

Question:
{query_text}
{examples}

Return your answer as JSON:
{{"answer": "your answer here", "evidence": ["quote1", "quote2"], "confidence": "high/medium/low"}}

JSON:
""",
    expected_inputs=["context", "query_text", "examples"],
)


# =============================================================================
# USAGE EXAMPLES
# =============================================================================
"""
# In build_kg.py:
from prompts import MEDICAL_EXTRACTION_TEMPLATE, MEDICAL_EXTRACTION_EXAMPLES
from neo4j_graphrag.kg_builder.components.entity_relation_extractor import LLMEntityRelationExtractor

extractor = LLMEntityRelationExtractor(
    llm=llm,
    prompt_template=MEDICAL_EXTRACTION_TEMPLATE,
)

# In eval_rag.py:
from prompts import MEDICAL_RAG_TEMPLATE

rag = GraphRAG(
    retriever=retriever,
    llm=llm,
    prompt_template=MEDICAL_RAG_TEMPLATE,
)
"""
