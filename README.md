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

## 8. Metadata
Every chunk carries: `source, page_start, page_end, chapter, slide_title, content_type, has_code,
chunk_index` (+ `token_count`, `pages`). No fabricated `subsection` field.

## 11. Validation
`validation/validator.py` checks page continuity, extraction status, non-empty chunks, valid
metadata, page ranges, unique `chunk_index`, and code integrity. Chunks below `min_tokens` are
**flagged for review, never deleted**.

## 12. Current limitations
- Because the decks are slides (~60 tokens of text each), many chunks are below 150 tokens. This
  is expected for this document type and is flagged, not hidden. On prose-heavy sections chunks
  grow toward the target.
- Token count is a word-count approximation (good enough for pre-embedding sizing).
- `no_text` / `hybrid` slides (no usable text layer) produce no chunk; only the native text layer
  is used.

## 13. Future roadmap (NOT implemented)
```
PDF
 ↓
Text extraction
 ↓
Page classification (text quality)
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
  ingestion/pdf_loader.py        pipeline orchestrator
  ingestion/report.py            extraction/chunking report
  models/document.py             Page / Chunk / LogicalGroup dataclasses
  config/__init__.py             tunable Config (sizes, thresholds, regexes)
```
