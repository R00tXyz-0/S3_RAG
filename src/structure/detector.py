from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from typing import Dict, List, Optional

from models.document import Document, LogicalGroup, Page

from config import Config, StructureConfig


def _norm_title(title: str) -> str:
    s = unicodedata.normalize("NFKD", title)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _titles_similar(a: str, b: str, threshold: float) -> bool:
    na, nb = _norm_title(a), _norm_title(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    return SequenceMatcher(None, na, nb).ratio() >= threshold


class StructureDetector:
    """Detect chapters and logical (slide-title / section) groups from the document itself.

    Works for both slide decks and textbook-like PDFs:
      * Chapters are detected with configurable regexes (Chapitre / Partie / ...).
      * The slide/section title is the most specific heading line on the page. On slide
        decks a recurring first-line "deck label" (e.g. "Administration Oracle") is skipped
        in favour of the real title line beneath it.
      * Consecutive pages sharing a (near-)identical title form one logical group.
    No page ranges are hardcoded; everything is derived from the content.
    """

    def __init__(self, config: Config):
        self.config: StructureConfig = config.structure
        self.classify = config.classify
        self._chapter_res = [
            re.compile(p, re.IGNORECASE) for p in self.config.chapter_patterns
        ]

    def _first_line_freq(self, pages: List[Page]) -> Dict[str, int]:
        freq: Dict[str, int] = {}
        for page in pages:
            src = page.cleaned_text or page.raw_text
            lines = [l.strip() for l in src.split("\n") if l.strip()]
            if not lines:
                continue
            fl = _norm_title(lines[0])
            if fl:
                freq[fl] = freq.get(fl, 0) + 1
        return freq

    def _detect_chapter(self, src: str) -> Optional[str]:
        for line in src.split("\n"):
            line = line.strip()
            if not line:
                continue
            for r in self._chapter_res:
                if r.search(line):
                    return line
        return None

    def _page_title(self, page: Page, first_line_freq: Dict[str, int]) -> str:
        src = page.cleaned_text or page.raw_text
        lines = [l.strip() for l in src.split("\n") if l.strip()]
        if not lines:
            return f"(page {page.page_number})"
        line1 = lines[0]
        if first_line_freq.get(_norm_title(line1), 0) >= self.classify.first_line_label_min_freq:
            # recurring deck label -> use the next meaningful line as the real title
            return lines[1] if len(lines) > 1 else line1
        return line1

    def detect(self, document: Document) -> List[LogicalGroup]:
        pages = document.pages
        if not pages:
            return []
        first_line_freq = self._first_line_freq(pages)

        # Per-page chapter + title
        chapter_per_page: List[Optional[str]] = []
        title_per_page: List[str] = []
        current_chapter: Optional[str] = None
        chapter_seen = False
        for page in pages:
            src = page.cleaned_text or page.raw_text
            ch = self._detect_chapter(src)
            if ch:
                current_chapter = ch
                chapter_seen = True
            chapter_per_page.append(current_chapter)
            title_per_page.append(self._page_title(page, first_line_freq))

        if not chapter_seen:
            # No explicit chapter marker: use the document's first title as the top level.
            fallback = title_per_page[0] if title_per_page else document.source
            current_chapter = fallback
            chapter_per_page = [fallback for _ in pages]

        # Build groups
        groups: List[LogicalGroup] = []
        doc_type = self._infer_doc_type(pages, first_line_freq)
        for idx, page in enumerate(pages):
            title = title_per_page[idx]
            chap = chapter_per_page[idx] or current_chapter or ""
            if groups:
                last = groups[-1]
                same_chapter = last.chapter == chap
                same_title = _titles_similar(last.title, title, self.config.title_similarity_threshold)
                if same_chapter and same_title:
                    last.page_numbers.append(page.page_number)
                    continue
            groups.append(
                LogicalGroup(chapter=chap, title=title, page_numbers=[page.page_number], doc_type=doc_type)
            )
        return groups

    def _infer_doc_type(self, pages: List[Page], first_line_freq: Dict[str, int]) -> str:
        label_pages = sum(
            1 for v in first_line_freq.values() if v >= self.classify.first_line_label_min_freq
        )
        if label_pages >= 3:
            return "slide_deck"
        return "document"
