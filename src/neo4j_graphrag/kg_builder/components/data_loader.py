#  Copyright (c) "Neo4j"
#  Neo4j Sweden AB [https://neo4j.com]
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#      https://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Data loaders for various file formats."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import fsspec
from fsspec import AbstractFileSystem
from fsspec.implementations.local import LocalFileSystem

from neo4j_graphrag.exceptions import DataLoaderError
from neo4j_graphrag.kg_builder.components.pdf_loader import DataLoader, is_default_fs
from neo4j_graphrag.kg_builder.components.types import DocumentInfo, PdfDocument


class TextLoader(DataLoader):
    """Load plain text files (.txt, .md, etc.)."""

    async def run(
        self,
        filepath: Union[str, Path],
        metadata: Optional[Dict[str, str]] = None,
        fs: Optional[Union[AbstractFileSystem, str]] = None,
    ) -> PdfDocument:
        if not isinstance(filepath, str):
            filepath = str(filepath)
        if isinstance(fs, str):
            fs = fsspec.filesystem(fs)
        elif fs is None:
            fs = LocalFileSystem()

        try:
            with fs.open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception as e:
            raise DataLoaderError(f"Failed to load text file: {e}")

        return PdfDocument(
            text=text,
            document_info=DocumentInfo(
                path=filepath,
                metadata=self.get_document_metadata(text, metadata),
                document_type="text",
            ),
        )


class JSONLoader(DataLoader):
    """Load JSON files and extract text from specified field(s).

    Args:
        content_field: The field to use as document text. Can be a dot-separated
            path for nested fields (e.g., "data.content").
        metadata_fields: List of fields to include in metadata.
        flatten_lists: If True and content_field points to a list, join items
            with newlines.

    Example:
        >>> loader = JSONLoader(content_field="content", metadata_fields=["title", "author"])
        >>> doc = await loader.run("document.json")
    """

    def __init__(
        self,
        content_field: str = "text",
        metadata_fields: Optional[List[str]] = None,
        flatten_lists: bool = True,
    ) -> None:
        self.content_field = content_field
        self.metadata_fields = metadata_fields or []
        self.flatten_lists = flatten_lists

    def _get_nested_value(self, data: Dict[str, Any], path: str) -> Any:
        """Get a value from nested dict using dot notation."""
        keys = path.split(".")
        value = data
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return None
        return value

    def _extract_content(self, data: Dict[str, Any]) -> str:
        """Extract text content from JSON data."""
        content = self._get_nested_value(data, self.content_field)
        if content is None:
            raise DataLoaderError(
                f"Content field '{self.content_field}' not found in JSON"
            )
        if isinstance(content, list) and self.flatten_lists:
            return "\n".join(str(item) for item in content)
        return str(content)

    def _extract_metadata(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Extract metadata fields from JSON data."""
        metadata = {}
        for field in self.metadata_fields:
            value = self._get_nested_value(data, field)
            if value is not None:
                metadata[field] = str(value)
        return metadata

    async def run(
        self,
        filepath: Union[str, Path],
        metadata: Optional[Dict[str, str]] = None,
        fs: Optional[Union[AbstractFileSystem, str]] = None,
    ) -> PdfDocument:
        if not isinstance(filepath, str):
            filepath = str(filepath)
        if isinstance(fs, str):
            fs = fsspec.filesystem(fs)
        elif fs is None:
            fs = LocalFileSystem()

        try:
            with fs.open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise DataLoaderError(f"Invalid JSON file: {e}")
        except Exception as e:
            raise DataLoaderError(f"Failed to load JSON file: {e}")

        text = self._extract_content(data)
        extracted_metadata = self._extract_metadata(data)
        if metadata:
            extracted_metadata.update(metadata)

        return PdfDocument(
            text=text,
            document_info=DocumentInfo(
                path=filepath,
                metadata=self.get_document_metadata(text, extracted_metadata),
                document_type="json",
            ),
        )


class JSONLLoader(DataLoader):
    """Load JSONL files (one JSON object per line).

    Each line is parsed and the content fields are concatenated.

    Args:
        content_field: The field to use as document text from each line.
        metadata_fields: List of fields to include in metadata (from first line).
        line_separator: Separator between lines when concatenating.

    Example:
        >>> loader = JSONLLoader(content_field="question", metadata_fields=["doc_name"])
        >>> doc = await loader.run("data.jsonl")
    """

    def __init__(
        self,
        content_field: str = "text",
        metadata_fields: Optional[List[str]] = None,
        line_separator: str = "\n\n",
    ) -> None:
        self.content_field = content_field
        self.metadata_fields = metadata_fields or []
        self.line_separator = line_separator

    def _get_nested_value(self, data: Dict[str, Any], path: str) -> Any:
        """Get a value from nested dict using dot notation."""
        keys = path.split(".")
        value = data
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return None
        return value

    async def run(
        self,
        filepath: Union[str, Path],
        metadata: Optional[Dict[str, str]] = None,
        fs: Optional[Union[AbstractFileSystem, str]] = None,
    ) -> PdfDocument:
        if not isinstance(filepath, str):
            filepath = str(filepath)
        if isinstance(fs, str):
            fs = fsspec.filesystem(fs)
        elif fs is None:
            fs = LocalFileSystem()

        texts: List[str] = []
        extracted_metadata: Dict[str, str] = {}

        try:
            with fs.open(filepath, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError as e:
                        raise DataLoaderError(
                            f"Invalid JSON at line {i + 1}: {e}"
                        )

                    # Extract content
                    content = self._get_nested_value(data, self.content_field)
                    if content is not None:
                        texts.append(str(content))

                    # Extract metadata from first line only
                    if i == 0:
                        for field in self.metadata_fields:
                            value = self._get_nested_value(data, field)
                            if value is not None:
                                extracted_metadata[field] = str(value)

        except DataLoaderError:
            raise
        except Exception as e:
            raise DataLoaderError(f"Failed to load JSONL file: {e}")

        if not texts:
            raise DataLoaderError(
                f"No content found in field '{self.content_field}'"
            )

        text = self.line_separator.join(texts)
        if metadata:
            extracted_metadata.update(metadata)

        return PdfDocument(
            text=text,
            document_info=DocumentInfo(
                path=filepath,
                metadata=self.get_document_metadata(text, extracted_metadata),
                document_type="jsonl",
            ),
        )


class CSVLoader(DataLoader):
    """Load CSV files and extract text from specified column(s).

    Args:
        content_column: The column to use as document text.
        metadata_columns: List of columns to include in metadata.
        row_separator: Separator between rows when concatenating.

    Example:
        >>> loader = CSVLoader(content_column="text", metadata_columns=["title"])
        >>> doc = await loader.run("data.csv")
    """

    def __init__(
        self,
        content_column: str = "text",
        metadata_columns: Optional[List[str]] = None,
        row_separator: str = "\n\n",
    ) -> None:
        self.content_column = content_column
        self.metadata_columns = metadata_columns or []
        self.row_separator = row_separator

    async def run(
        self,
        filepath: Union[str, Path],
        metadata: Optional[Dict[str, str]] = None,
        fs: Optional[Union[AbstractFileSystem, str]] = None,
    ) -> PdfDocument:
        import csv

        if not isinstance(filepath, str):
            filepath = str(filepath)
        if isinstance(fs, str):
            fs = fsspec.filesystem(fs)
        elif fs is None:
            fs = LocalFileSystem()

        texts: List[str] = []
        extracted_metadata: Dict[str, str] = {}

        try:
            with fs.open(filepath, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    content = row.get(self.content_column)
                    if content:
                        texts.append(content)

                    # Extract metadata from first row only
                    if i == 0:
                        for col in self.metadata_columns:
                            value = row.get(col)
                            if value:
                                extracted_metadata[col] = value

        except Exception as e:
            raise DataLoaderError(f"Failed to load CSV file: {e}")

        if not texts:
            raise DataLoaderError(
                f"No content found in column '{self.content_column}'"
            )

        text = self.row_separator.join(texts)
        if metadata:
            extracted_metadata.update(metadata)

        return PdfDocument(
            text=text,
            document_info=DocumentInfo(
                path=filepath,
                metadata=self.get_document_metadata(text, extracted_metadata),
                document_type="csv",
            ),
        )


class AutoLoader(DataLoader):
    """Automatically detect file type and use appropriate loader.

    Supports: PDF, JSON, JSONL, CSV, TXT, MD

    Args:
        json_content_field: Field to extract text from in JSON/JSONL files.
        json_metadata_fields: Fields to include as metadata from JSON/JSONL.
        csv_content_column: Column to extract text from in CSV files.
        csv_metadata_columns: Columns to include as metadata from CSV.

    Example:
        >>> loader = AutoLoader(json_content_field="question")
        >>> # Automatically detects and processes any supported format
        >>> doc = await loader.run("data.pdf")
        >>> doc = await loader.run("data.jsonl")
        >>> doc = await loader.run("data.csv")
    """

    SUPPORTED_EXTENSIONS = {
        ".pdf": "pdf",
        ".json": "json",
        ".jsonl": "jsonl",
        ".ndjson": "jsonl",
        ".csv": "csv",
        ".txt": "text",
        ".md": "text",
        ".markdown": "text",
    }

    def __init__(
        self,
        json_content_field: str = "text",
        json_metadata_fields: Optional[List[str]] = None,
        csv_content_column: str = "text",
        csv_metadata_columns: Optional[List[str]] = None,
    ) -> None:
        self.json_content_field = json_content_field
        self.json_metadata_fields = json_metadata_fields or []
        self.csv_content_column = csv_content_column
        self.csv_metadata_columns = csv_metadata_columns or []

        # Initialize loaders
        self._text_loader = TextLoader()
        self._json_loader = JSONLoader(
            content_field=json_content_field,
            metadata_fields=json_metadata_fields,
        )
        self._jsonl_loader = JSONLLoader(
            content_field=json_content_field,
            metadata_fields=json_metadata_fields,
        )
        self._csv_loader = CSVLoader(
            content_column=csv_content_column,
            metadata_columns=csv_metadata_columns,
        )

    def _detect_type(self, filepath: str) -> str:
        """Detect file type from extension or content."""
        ext = Path(filepath).suffix.lower()

        if ext in self.SUPPORTED_EXTENSIONS:
            return self.SUPPORTED_EXTENSIONS[ext]

        # Unknown extension - try to detect from content
        raise DataLoaderError(
            f"Unsupported file type: '{ext}'. "
            f"Supported: {', '.join(self.SUPPORTED_EXTENSIONS.keys())}"
        )

    async def run(
        self,
        filepath: Union[str, Path],
        metadata: Optional[Dict[str, str]] = None,
        fs: Optional[Union[AbstractFileSystem, str]] = None,
    ) -> PdfDocument:
        if not isinstance(filepath, str):
            filepath = str(filepath)

        file_type = self._detect_type(filepath)

        if file_type == "pdf":
            # Import here to avoid circular import and make pypdf optional
            from neo4j_graphrag.kg_builder.components.pdf_loader import PdfLoader

            loader = PdfLoader()
            return await loader.run(filepath, metadata, fs)
        elif file_type == "json":
            return await self._json_loader.run(filepath, metadata, fs)
        elif file_type == "jsonl":
            return await self._jsonl_loader.run(filepath, metadata, fs)
        elif file_type == "csv":
            return await self._csv_loader.run(filepath, metadata, fs)
        elif file_type == "text":
            return await self._text_loader.run(filepath, metadata, fs)
        else:
            raise DataLoaderError(f"Unsupported file type: {file_type}")
