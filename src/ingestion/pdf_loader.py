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
from vision.merger import merge_texts
from vision.model import GeminiVisionModel
from vision.renderer import PageRenderer
from vision.detector import VisualContentDetector
from vision.stage import VisionStage


def run_pipeline(
    pdf_path: str,
    config: Config,
    processed_dir: Optional[str] = None,
    vision_stage: Optional[VisionStage] = None,
) -> Tuple[Document, Dict]:
    """Ingestion + chunking pipeline with an optional Visual -> Text stage.

    PDF -> extract -> classify -> structure -> [Visual -> Text] -> merge -> clean -> chunk -> validate -> report

    The text-only pipeline is preserved: when the vision stage is disabled or fails,
    native text is never lost.
    """
    processed_dir = processed_dir or config.processed_dir
    source = os.path.basename(pdf_path)
    stem = os.path.splitext(source)[0]

    # 1) Text extraction (keep the pypdf page objects for visual-content detection)
    pages: List[Page] = []
    pypdf_pages: Dict[int, object] = {}
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
        pypdf_pages[page_num] = pypdf_page
        if config.max_pages and page_num >= config.max_pages:
            break

    # 2) Classification (text quality only; used for reporting / coverage)
    classify_pages(pages, config)

    # 3) Structure detection (on raw text, before cleaning) to drive chunk grouping
    #    and to give the vision model chapter/slide context.
    document = Document(source=source, pages=pages)
    detector = StructureDetector(config)
    groups = detector.detect(document)
    page_context: Dict[int, Tuple[str, str]] = {}
    for g in groups:
        for pnum in g.page_numbers:
            page_context[pnum] = (g.chapter, g.title)

    # 4) Visual -> Text stage (optional). Renders meaningful pages, infers text with the
    #    Gemini multimodal API, and stores it on the page. Native text is untouched on any failure.
    if vision_stage is None and config.vision.enabled:
        vision_stage = VisionStage(
            model=GeminiVisionModel(
                config.vision.model_name,
                timeout=config.vision.timeout,
                max_retries=config.vision.max_retries,
                temperature=config.vision.temperature,
            ),
            renderer=PageRenderer(dpi=config.vision.render_dpi),
            detector=VisualContentDetector(config.vision),
            config=config.vision,
            cache_dir=config.vision.cache_dir,
        )
    if vision_stage is not None:
        vision_stage.process(pdf_path, pages, pypdf_pages, page_context)

    # 5) Merge native + visual text, then clean (boilerplate learned from all pages).
    cleaner = TextCleaner(config)
    combined_texts = [merge_texts(p.raw_text, p.visual_text) for p in pages]
    cleaner.fit(combined_texts)
    for page, combined in zip(pages, combined_texts):
        page.cleaned_text = cleaner.clean(combined)
        page.final_text = page.cleaned_text
        page.extraction_status = "extracted"
        page.has_code = text_has_code(
            page.final_text or page.cleaned_text, set(config.sql_keywords)
        )

    # 6) Chunking
    document.chunks = chunk_document(document, groups, config)

    # 7) Validation
    validation = validate(document, config)

    # 8) Report
    report = build_report(source, pages, document.chunks, groups, validation, config)
    return document, report


def build_report(source, pages, chunks, groups, validation, config) -> Dict:
    def count(pred):
        return sum(1 for p in pages if pred(p))

    tokens = [c.token_count for c in chunks] or [0]
    code_chunks = sum(1 for c in chunks if c.has_code)
    vision_chunks = sum(1 for c in chunks if c.has_visual_content)
    chapters = []
    for g in groups:
        if g.chapter not in chapters:
            chapters.append(g.chapter)

    vision_model = next((p.vision_model for p in pages if p.vision_model), "")
    vision_statuses = {}
    for p in pages:
        if p.visual_processing_status:
            vision_statuses[p.visual_processing_status] = (
                vision_statuses.get(p.visual_processing_status, 0) + 1
            )

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
        "chunks_with_visual_content": vision_chunks,
        "visual_pages": count(lambda p: p.has_visual_content),
        "vision_model": vision_model,
        "vision_statuses": vision_statuses,
        "validation_ok": validation.ok,
        "validation_warnings": len(validation.warnings),
        "validation_errors": len(validation.errors),
        "chapters": chapters,
        "small_chunk_indices": validation.small_chunks,
    }


def run_vision_test_mode(
    pdf_path: str,
    config: Config,
    n_pages: int = 3,
    vision_stage: Optional[VisionStage] = None,
):
    """Process only N selected pages through the Visual -> Text stage (no full pipeline run).

    Picks the first N pages that contain meaningful visual content, sends them to Gemini
    (or a provided stage), caches results, and prints a small per-page report:
        page number | native text length | visual text length | processing status

    This is a smoke test for the vision stage; it does NOT run the full 288+ page job.
    Requires GEMINI_API_KEY to be set in the environment (or .env) to make real calls.
    """
    from dotenv import load_dotenv

    load_dotenv()

    source = os.path.basename(pdf_path)
    pages: List[Page] = []
    pypdf_pages: Dict[int, object] = {}
    for idx, pypdf_page in iter_pages(pdf_path):
        page_num = idx + 1
        pages.append(Page(page_number=page_num, source=source, raw_text=extract_page_text(pypdf_page)))
        pypdf_pages[page_num] = pypdf_page

    classify_pages(pages, config)
    detector = VisualContentDetector(config.vision)
    candidates = [p for p in pages if detector.has_meaningful_visual(p, pypdf_pages.get(p.page_number))]
    if len(candidates) > n_pages:
        candidates = candidates[:n_pages]

    if vision_stage is None:
        vision_stage = VisionStage(
            model=GeminiVisionModel(
                config.vision.model_name,
                timeout=config.vision.timeout,
                max_retries=config.vision.max_retries,
                temperature=config.vision.temperature,
            ),
            renderer=PageRenderer(dpi=config.vision.render_dpi),
            detector=detector,
            config=config.vision,
            cache_dir=config.vision.cache_dir,
        )

    vision_stage.process(
        pdf_path,
        candidates,
        {p.page_number: pypdf_pages[p.page_number] for p in candidates},
        {},
    )

    print("\n=== Vision test mode ({} pages) ===".format(len(candidates)))
    print(f"{'page':>5} | {'native_len':>10} | {'visual_len':>10} | status")
    print("-" * 45)
    for p in candidates:
        print(
            f"{p.page_number:>5} | {len(p.raw_text):>10} | {len(p.visual_text):>10} | {p.visual_processing_status}"
        )
    return candidates
