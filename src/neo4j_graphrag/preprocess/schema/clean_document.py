from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class CleanBlock:
    block_id: str
    type: str
    text: str
    page: Optional[int]
    section: str
    order: int
    html: Optional[str] = None
    grid: Optional[list[list[dict[str, Any]]]] = None
    bbox: Optional[list[list[float]]] = None
    attrs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CleanDocument:
    doc_id: str
    source_path: str
    parser_backend: str
    blocks: list[CleanBlock] = field(default_factory=list)
    attrs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "source_path": self.source_path,
            "parser_backend": self.parser_backend,
            "blocks": [block.to_dict() for block in self.blocks],
            "attrs": self.attrs,
        }
