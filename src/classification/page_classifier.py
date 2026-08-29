from __future__ import annotations

from typing import List

from models.document import Page, PageClass

from config import ClassifyConfig, Config


def classify_page(cleaned_len: int, config: Config) -> PageClass:
    """Classify a page from the quality of its extracted text (text-only pipeline).

      * NO_TEXT    : essentially no usable text layer.
      * HYBRID     : some text (usually a title) but the body is thin / likely non-textual.
      * NATIVE_TEXT: enough text that the page is usable as-is.

    This is driven entirely by extraction quality, never by hardcoded page ranges.
    """
    cfg: ClassifyConfig = config.classify
    if cleaned_len <= cfg.no_text_max_chars:
        return PageClass.NO_TEXT
    if cleaned_len <= cfg.hybrid_max_chars:
        return PageClass.HYBRID
    return PageClass.NATIVE_TEXT


def classify_pages(pages: List[Page], config: Config) -> None:
    for page in pages:
        page.page_class = classify_page(len(page.raw_text.strip()), config)
