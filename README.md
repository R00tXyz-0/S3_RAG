# RAG Ingestion & Chunking — Cours Bases de Données Avancées (Oracle / PL/SQL)

## 1. Project purpose
A serious Retrieval-Augmented Generation ingestion layer for an *Advanced Database* course.
This repository implements the **first stage only**: turning two French course PDFs into
clean, structure-aware, validated chunks with rich metadata. Embeddings, vector search,
retrieval, reranking and LLM generation are **deliberately NOT implemented yet** (see roadmap).

## 2. The two course PDFs
- `data/raw/oracle/Cours_Oracle_Complet.pdf` — 133 pages, **PowerPoint exported to PDF**
  (one slide per page, bullets/diagrams/SQL, recurring footer, no continuous prose).
- `data/raw/plsql/PLSQL Version Finale (1).pdf` — 155 pages, also a **slide-style deck**
  (titles + page numbers + professor footer; body often flattened into images).

## 3. Why this is a RAG ingestion pipeline
Downstream RAG needs small, self-contained, well-described passages. Raw PDF pages are not
usable directly: ~31% of the Oracle PDF has no text layer, slides repeat titles, and footers
pollute the text. The pipeline normalises all of that into chunks ready for embedding later.

## 4. Why chunking is necessary
A 133-page slide deck cannot be embedded as one blob. Chunks must preserve meaning: keep a
definition with its explanation, never split a SQL statement, and keep related slides of the
same title together.

## 5. The actual document structures
- **Oracle**: `Chapitre 1` (pp.1–37) and `Chapitre 3` (pp.38–133). **There is no Chapitre 2**
  in the source — this is preserved, not invented. Slide titles repeat across consecutive pages
  (e.g. pp.18–26 "Structure des processus d'arrière-plan"), forming *logical groups*.
- **PL/SQL**: no `Chapitre` markers; the deck title `PL/SQL` is used as the top-level chapter and
  every slide title becomes its own logical group. The structure is **discovered from the PDF**,
  not hardcoded.

## 6. Oracle-specific structure-aware chunking
Hierarchy: **Chapter → slide-title group → page → content**. A slide-title group is a run of
consecutive pages sharing a (near-identical) title. Small pages in the same group are merged; a
group is only split when it exceeds the size limit. Target ≈ 550 tokens, overlap ≈ 64, hard max
≈ 800, min preferred ≈ 150. No overlap across chapter or slide-title-group boundaries.

## 7. PL/SQL adaptive structure detection
`StructureDetector` is generic: it detects chapters via configurable regexes and finds the real
heading per page (skipping recurring first-line "deck labels"). The same grouping logic serves
both slide decks and textbook-like documents, so PL/SQL is handled without special-casing.

## 8. Visual → Text processing — **rebuilt with Gemini**
The **Visual → Text** stage is a first-class part of the pipeline. Each PDF page is rendered with
PyMuPDF and, when it carries meaningful visual content, sent to the **Google Gemini** multimodal API
(`google-genai` SDK, model `gemini-3.6-flash` by default) to extract a clean textual description of
diagrams, tables, architecture figures and any SQL/PL/SQL code embedded in images. The extracted text
is **merged with the native text layer** (native first, visual second) before cleaning and chunking,
so image-only slides and hybrid slides become real chunks without changing the text chunking logic.

Key design points:
- **Selective inference.** Only `no_text` and `hybrid` pages, plus native-text pages that embed a
  large image (a diagram), are sent to Gemini. Plain text pages are skipped to avoid wasted calls.
- **Deduplication.** Identical rendered pages (same image hash) within a run are processed once.
- **Caching.** Successful per-page results are cached to `data/processed/vision/<source>/page_NNN.json`;
  a page is never re-processed if its result is already cached.
- **Production resilience.** Timeout, retry with exponential backoff, rate-limit (429) handling,
  API error handling, per-page logging, and graceful degradation. If the API key is missing, the
  network fails, or a page errors, native text is **never lost**; the failure is recorded in
  `visual_processing_status` and the page continues through the text pipeline.
- **Secure key handling.** The API key is read from `GEMINI_API_KEY` (optionally a `.env` file via
  `python-dotenv`). It is never hardcoded and never printed. `.env` is git-ignored.
- **Chapter/slide context.** The vision prompt receives the detected chapter and slide title so the
  model can label diagrams correctly.

## 9. Metadata
Every chunk carries: `source, page_start, page_end, chapter, slide_title, content_type, has_code,
chunk_index` (+ `token_count`, `pages`). Vision fields are added when present:
`has_visual_content, vision_model, visual_processing_status`. (The OCR-specific `is_ocr` field was
removed in an earlier cleanup; the analogous information for the vision stage is captured by
`has_visual_content` / `visual_processing_status`.) No fabricated `subsection` field.

## 11. Validation
`validation/validator.py` checks page continuity, extraction status, non-empty chunks, valid
metadata, page ranges, unique `chunk_index`, and code integrity. Chunks below `min_tokens` are
**flagged for review, never deleted**.

## 12. Current limitations
- The Visual → Text stage needs `google-genai`, `python-dotenv`, `pymupdf` and `Pillow`. If those are
  not installed, or the Gemini API key is missing / the request fails, the pipeline automatically
  degrades to the text-only behaviour: native text is preserved and `no_text` pages simply produce no
  chunk.
- Because the decks are slides (~60 tokens of text each), many chunks are below 150 tokens. This
  is expected for this document type and is flagged, not hidden. On prose-heavy sections chunks
  grow toward the target.
- Token count is a word-count approximation (good enough for pre-embedding sizing).
- Gemini inference is intentionally mocked in the test suite (a `generate_fn` is injected); running the
  real model only requires a `GEMINI_API_KEY` (no local GPU / model download).

## 13. Future roadmap (NOT implemented)
```
PDF
 ↓
Text extraction
 ↓
Page classification (text quality)
   ├── Native text ─────────────┐
   ├── Hybrid (title only) ─────┤
   └── No-text (image slide) ───┤
                                ↓
                       Visual → Text (render + Gemini)
                                ↓
                       Merge native + visual text
                                ↓
Cleaning
 ↓
Structure detection
 ↓
Structure-aware chunking
 ↓
Chunks + metadata
 ↓
[Future: Embeddings]
 ↓
[Future: Vector DB]
 ↓
[Future: Retrieval]
 ↓
[Future: Reranking]
 ↓
[Future: LLM]
```
Embeddings, vector database, retrieval, reranking, LLM generation, API and UI are explicitly
out of scope for this stage.

## How to run
```
pip install -r requirements.txt
python run_ingestion.py                 # processes both PDFs, writes reports + chunks JSONL
python -m pytest -q                     # unit + integration tests
```
Outputs: `data/processed/evaluation/report_*.json`, `data/processed/chunks/chunks_*.jsonl`.

## Architecture (modules)
```
src/
  extraction/extractor.py        raw text extraction per page
  classification/page_classifier.py  native / hybrid / no_text (text quality)
  cleaning/cleaner.py            boilerplate/footer removal
  structure/detector.py          chapters + slide-title/section groups
  chunking/chunker.py            structure-aware chunker (size/overlap/code guards)
  metadata/metadata.py           chunk metadata assembly
  validation/validator.py        page + chunk validation
  vision/prompt.py               vision model system prompt + context builder
  vision/renderer.py             PyMuPDF page renderer (lazy import)
  vision/detector.py             visual-content detection / filtering
   vision/model.py                GeminiVisionModel wrapper (mockable, injectable generate_fn)
  vision/merger.py               native + visual text merge
  vision/stage.py                per-page vision orchestration
  ingestion/pdf_loader.py        pipeline orchestrator
  ingestion/report.py            extraction/chunking report
  models/document.py             Page / Chunk / LogicalGroup dataclasses
  config/__init__.py             tunable Config (sizes, thresholds, regexes, VisionConfig)
```
