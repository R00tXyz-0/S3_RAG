from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class PageClass(str, Enum):
    NATIVE_TEXT = "native_text"
    HYBRID = "hybrid"
    NO_TEXT = "no_text"


class ContentType(str, Enum):
    TEXT = "text"
    CODE = "code"


@dataclass
class Page:
    page_number: int
    source: str
    raw_text: str = ""
    cleaned_text: str = ""
    final_text: str = ""
    page_class: PageClass = PageClass.NATIVE_TEXT
    has_code: bool = False
    content_type: ContentType = ContentType.TEXT
    extraction_status: str = "pending"
    notes: List[str] = field(default_factory=list)

    def text_for_chunking(self) -> str:
        return self.final_text or self.cleaned_text or self.raw_text


@dataclass
class Chunk:
    chunk_index: int
    text: str
    source: str
    page_start: int
    page_end: int
    chapter: str
    slide_title: str
    content_type: ContentType
    has_code: bool
    token_count: int
    pages: List[int] = field(default_factory=list)
    section: Optional[str] = None
    subsection: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Document:
    source: str
    pages: List[Page] = field(default_factory=list)
    chunks: List[Chunk] = field(default_factory=list)


@dataclass
class LogicalGroup:
    chapter: str
    title: str
    page_numbers: List[int]
    doc_type: str = "slide_deck"
