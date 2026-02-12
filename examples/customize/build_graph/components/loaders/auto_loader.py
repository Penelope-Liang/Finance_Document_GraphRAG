"""Use the AutoLoader component to automatically detect and load various file formats.

Supported formats: PDF, JSON, JSONL, CSV, TXT, MD
"""

import asyncio

from neo4j_graphrag.kg_builder.components.data_loader import AutoLoader


async def main() -> None:
    # AutoLoader automatically detects file type and uses the appropriate loader
    loader = AutoLoader(
        json_content_field="question",  # For JSON/JSONL files
        json_metadata_fields=["doc_name", "answer"],  # Optional metadata fields
        csv_content_column="text",  # For CSV files
        csv_metadata_columns=["title"],  # Optional metadata columns
    )

    # Example 1: Load a text file
    # doc = await loader.run("document.txt")
    # print(f"Text file: {doc.text[:100]}...")

    # Example 2: Load a JSONL file
    # doc = await loader.run("data.jsonl")
    # print(f"JSONL file: {doc.text[:100]}...")
    # print(f"Metadata: {doc.document_info.metadata}")

    # Example 3: Load a CSV file
    # doc = await loader.run("data.csv")
    # print(f"CSV file: {doc.text[:100]}...")

    # Example 4: Load a PDF file (requires pypdf)
    # doc = await loader.run("document.pdf")
    # print(f"PDF file: {doc.text[:100]}...")

    print("AutoLoader is ready to use!")
    print(f"Supported formats: {', '.join(AutoLoader.SUPPORTED_EXTENSIONS.keys())}")


if __name__ == "__main__":
    asyncio.run(main())
