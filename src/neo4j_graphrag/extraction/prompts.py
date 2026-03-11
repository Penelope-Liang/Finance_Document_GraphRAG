"""Prompt templates for entity/relationship extraction."""

GRAPH_EXTRACTION_PROMPT = """
-Goal-
Given a text document and a list of entity types, identify matching entities and relationships among them.

-Steps-
1. Identify all entities. For each entity extract:
- entity_name: capitalized name
- entity_type: one of [{entity_types}]
- entity_description: concise description
Format:
("entity"<|><entity_name><|><entity_type><|><entity_description>)

2. Identify clearly related entity pairs. For each relation extract:
- source_entity
- target_entity
- relationship_description
- relationship_strength: numeric score
Format:
("relationship"<|><source_entity><|><target_entity><|><relationship_description><|><relationship_strength>)

3. Return all entities and relationships in one list separated by ##.
4. End with <|COMPLETE|>.

Entity_types: {entity_types}
Text: {input_text}
Output:
"""

CONTINUE_PROMPT = (
    "MANY entities and relationships were missed in the last extraction. "
    "Remember to ONLY emit entities matching the requested entity types. "
    "Add them below using the same format:\n"
)

LOOP_PROMPT = (
    "Are there still missing entities or relationships? "
    "Answer only Y or N.\n"
)
