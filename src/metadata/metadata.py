from __future__ import annotations

from typing import Any, Dict

from models.document import Chunk, ContentType


def compose_metadata(
    source: str,
    page_start: int,
    page_end: int,
    chapter: str,
    slide_title: str,
    content_type: ContentType,
    has_code: bool,
    chunk_index: int,
    pages: list,
    token_count: int,
    *,
    chunk_id: Optional[str] = None,
    section: Optional[str] = None,
    subsection: Optional[str] = None,
    content: Optional[str] = None,
    code_blocks: int = 0,
) -> Dict[str, Any]:
    """Assemble the required chunk metadata.

    Required keys (per spec): source, page_start, page_end, chapter, slide_title,
    content_type, has_code, chunk_index. Structure-aware fields (chunk_id, section,
    subsection, content, code_blocks) are added; missing values are stored as null
    rather than invented.
    """
    return {
        "chunk_id": chunk_id,
        "source": source,
        "page_start": page_start,
        "page_end": page_end,
        "chapter": chapter,
        "slide_title": slide_title,
        "section": section,
        "subsection": subsection,
        "content_type": content_type.value,
        "has_code": bool(has_code),
        "chunk_index": chunk_index,
        "token_count": token_count,
        "pages": list(pages),
        "content": content,
        "code_blocks": code_blocks,
    }


def chunk_summary(chunk: Chunk) -> Dict[str, Any]:
    return {
        "chunk_index": chunk.chunk_index,
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "chapter": chunk.chapter,
        "slide_title": chunk.slide_title,
        "content_type": chunk.content_type.value,
        "has_code": chunk.has_code,
        "token_count": chunk.token_count,
    }
