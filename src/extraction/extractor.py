from __future__ import annotations

import pypdf
from typing import List, Tuple


def page_count(pdf_path: str) -> int:
    return len(pypdf.PdfReader(pdf_path).pages)


def extract_page_text(page) -> str:
    """Extract the text layer of a single pypdf page object."""
    try:
        return page.extract_text() or ""
    except Exception:
        return ""


def iter_pages(pdf_path: str):
    """Yield (index_0based, pypdf_page) for every page in the PDF."""
    reader = pypdf.PdfReader(pdf_path)
    for i, page in enumerate(reader.pages):
        yield i, page


def extract_all(pdf_path: str) -> List[str]:
    """Return the extracted text for every page (text-only pipeline)."""
    reader = pypdf.PdfReader(pdf_path)
    return [extract_page_text(page) for page in reader.pages]
