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
    has_visual_content: bool = False,
    vision_model: str = "",
    visual_processing_status: str = "",
) -> Dict[str, Any]:
    """Assemble the required chunk metadata.

    Required keys (per spec): source, page_start, page_end, chapter, slide_title,
    content_type, has_code, chunk_index. Vision fields are added when present.
    Note: the OCR-specific `is_ocr` field was intentionally removed in a prior cleanup;
    the analogous information for the rebuilt vision stage is captured by
    has_visual_content / visual_processing_status instead.
    """
    return {
        "source": source,
        "page_start": page_start,
        "page_end": page_end,
        "chapter": chapter,
        "slide_title": slide_title,
        "content_type": content_type.value,
        "has_code": bool(has_code),
        "chunk_index": chunk_index,
        "token_count": token_count,
        "pages": list(pages),
        "has_visual_content": bool(has_visual_content),
        "vision_model": vision_model,
        "visual_processing_status": visual_processing_status,
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
