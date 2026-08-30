from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

from models.document import Document, Page

from config import Config
from classification.page_classifier import classify_pages
from chunking.chunker import chunk_document, text_has_code
from cleaning.cleaner import TextCleaner
from extraction.extractor import extract_page_text, iter_pages
from structure.detector import StructureDetector
from validation.validator import validate


def run_pipeline(
    pdf_path: str,
    config: Config,
    processed_dir: Optional[str] = None,
) -> Tuple[Document, Dict]:
    """Native-text-only ingestion + chunking pipeline.

    PDF -> extract -> classify -> clean -> structure -> chunk -> validate -> report

    No images, OCR, or multimodal inference are used; only the extracted text layer.
    """
    processed_dir = processed_dir or config.processed_dir
    source = os.path.basename(pdf_path)
    stem = os.path.splitext(source)[0]

    # 1) Text extraction (native PDF text layer only)
    pages: List[Page] = []
    for idx, pypdf_page in iter_pages(pdf_path):
        page_num = idx + 1
        raw = extract_page_text(pypdf_page)
        pages.append(
            Page(
                page_number=page_num,
                source=source,
                raw_text=raw,
            )
        )
        if config.max_pages and page_num >= config.max_pages:
            break

    # 2) Classification (text quality only; used for reporting / coverage)
    classify_pages(pages, config)

    # 3) Cleaning (boilerplate learned from all pages)
    cleaner = TextCleaner(config)
    cleaner.fit([p.raw_text for p in pages])
    for page in pages:
        page.cleaned_text = cleaner.clean(page.raw_text)
        page.final_text = page.cleaned_text
        page.extraction_status = "extracted"
        page.has_code = text_has_code(
            page.final_text or page.cleaned_text, set(config.sql_keywords)
        )

    # 4) Structure detection (chapters + slide-title groups)
    document = Document(source=source, pages=pages)
    detector = StructureDetector(config)
    groups = detector.detect(document)

    # 5) Chunking
    document.chunks = chunk_document(document, groups, config)

    # 6) Validation
    validation = validate(document, config)

    # 7) Report
    report = build_report(source, pages, document.chunks, groups, validation, config)
    return document, report


def build_report(source, pages, chunks, groups, validation, config) -> Dict:
    def count(pred):
        return sum(1 for p in pages if pred(p))

    tokens = [c.token_count for c in chunks] or [0]
    code_chunks = sum(1 for c in chunks if c.has_code)
    code_blocks = sum(int(c.metadata.get("code_blocks", 0) or 0) for c in chunks)
    sections = {c.metadata.get("section") for c in chunks if c.metadata.get("section")}
    chapters = []
    for g in groups:
        if g.chapter not in chapters:
            chapters.append(g.chapter)

    return {
        "source": source,
        "total_pages": len(pages),
        "native_text_pages": count(lambda p: p.page_class.value == "native_text"),
        "hybrid_pages": count(lambda p: p.page_class.value == "hybrid"),
        "no_text_pages": count(lambda p: p.page_class.value == "no_text"),
        "chapters_detected": len(chapters),
        "logical_groups": len(groups),
        "chunks": len(chunks),
        "min_chunk_tokens": min(tokens),
        "max_chunk_tokens": max(tokens),
        "avg_chunk_tokens": round(sum(tokens) / len(tokens), 1),
        "chunks_below_min_tokens": len(validation.small_chunks),
        "chunks_with_code": code_chunks,
        "code_blocks_preserved": code_blocks,
        "sections_detected": len(sections),
        "suspicious_chunks": sorted(set(validation.suspicious)),
        "validation_ok": validation.ok,
        "validation_warnings": len(validation.warnings),
        "validation_errors": len(validation.errors),
        "chapters": chapters,
        "small_chunk_indices": validation.small_chunks,
    }
