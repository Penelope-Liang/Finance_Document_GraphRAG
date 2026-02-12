"""Use the JSONLLoader component to load JSONL (JSON Lines) files.

JSONL files contain one JSON object per line. This loader extracts content
from a specified field across all lines and combines them into a single document.
"""

import asyncio

from neo4j_graphrag.kg_builder.components.data_loader import JSONLLoader


async def main() -> None:
    # JSONLLoader extracts content from each line's specified field
    # and concatenates them into a single document

    # Example: For a JSONL file like:
    # {"question": "What is Python?", "answer": "A programming language", "doc_name": "faq.txt"}
    # {"question": "What is Neo4j?", "answer": "A graph database", "doc_name": "faq.txt"}

    loader = JSONLLoader(
        content_field="question",  # Field to extract as main content
        metadata_fields=["doc_name"],  # Fields to include as metadata (from first line)
        line_separator="\n\n",  # Separator between lines
    )

    # doc = await loader.run("data.jsonl")
    # print(f"Content: {doc.text}")
    # print(f"Metadata: {doc.document_info.metadata}")

    # For nested fields, use dot notation:
    # loader = JSONLLoader(content_field="data.content")

    print("JSONLLoader is ready to use!")
    print("Usage: doc = await loader.run('your_file.jsonl')")


if __name__ == "__main__":
    asyncio.run(main())
