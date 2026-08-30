from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from models.document import Chunk, ContentType, Document, Page, PageClass

from config import Config
from chunking.chunker import count_tokens

VALID_CONTENT_TYPES = {ct.value for ct in ContentType}


@dataclass
class ValidationReport:
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    small_chunks: List[int] = field(default_factory=list)
    suspicious: List[int] = field(default_factory=list)
    ok: bool = True

    def add_error(self, msg: str):
        self.errors.append(msg)
        self.ok = False

    def add_warning(self, msg: str):
        self.warnings.append(msg)

    def to_dict(self) -> Dict:
        return {
            "ok": self.ok,
            "errors": self.errors,
            "warnings": self.warnings,
            "small_chunk_count": len(self.small_chunks),
            "small_chunk_indices": self.small_chunks,
            "suspicious_chunk_indices": self.suspicious,
        }


def validate(document: Document, config: Config) -> ValidationReport:
    report = ValidationReport()
    pages = document.pages

    # --- Page-level checks ---
    expected = list(range(1, len(pages) + 1))
    actual = [p.page_number for p in pages]
    if actual != expected:
        report.add_error(
            f"Page numbers not contiguous: expected 1..{len(pages)}, got {actual[:10]}..."
        )
    for p in pages:
        if p.extraction_status == "pending":
            report.add_error(f"Page {p.page_number}: extraction status never set")
        if p.page_class not in (
            PageClass.NATIVE_TEXT,
            PageClass.HYBRID,
            PageClass.NO_TEXT,
        ):
            report.add_error(f"Page {p.page_number}: invalid page_class {p.page_class}")

    # --- Chunk-level checks ---
    seen_indices = set()
    for c in document.chunks:
        if not c.text.strip():
            report.add_error(f"Chunk {c.chunk_index}: empty text")
        if c.chunk_index in seen_indices:
            report.add_error(f"Chunk {c.chunk_index}: duplicate chunk_index")
        seen_indices.add(c.chunk_index)
        if c.page_start > c.page_end:
            report.add_error(f"Chunk {c.chunk_index}: page_start > page_end")
        if c.source != document.source:
            report.add_error(f"Chunk {c.chunk_index}: source mismatch")
        if c.metadata.get("content_type") not in VALID_CONTENT_TYPES:
            report.add_error(f"Chunk {c.chunk_index}: invalid content_type")
        if c.token_count <= 0:
            report.add_error(f"Chunk {c.chunk_index}: non-positive token_count")
        if c.token_count < config.chunk.min_tokens:
            report.small_chunks.append(c.chunk_index)
            report.add_warning(
                f"Chunk {c.chunk_index}: below min_tokens ({c.token_count} < {config.chunk.min_tokens})"
            )
        if c.token_count < config.quality.small_chunk_tokens:
            report.add_warning(
                f"Chunk {c.chunk_index}: extremely small ({c.token_count} tokens)"
            )
            report.suspicious.append(c.chunk_index)
        if c.token_count > config.quality.large_chunk_tokens:
            report.add_warning(
                f"Chunk {c.chunk_index}: extremely large ({c.token_count} tokens)"
            )
            report.suspicious.append(c.chunk_index)
        if not c.metadata.get("source"):
            report.add_error(f"Chunk {c.chunk_index}: missing source metadata")
            report.suspicious.append(c.chunk_index)
        if c.has_code:
            _check_code_integrity(c, report)
            if _code_block_broken(c):
                report.add_warning(
                    f"Chunk {c.chunk_index}: possible broken/unbalanced code block"
                )
                report.suspicious.append(c.chunk_index)

    # Missing section info when sections exist somewhere in the document.
    sections_present = {
        c.metadata.get("section") for c in document.chunks if c.metadata.get("section")
    }
    if sections_present:
        for c in document.chunks:
            if not c.metadata.get("section"):
                report.add_warning(
                    f"Chunk {c.chunk_index}: missing section metadata"
                )
                report.suspicious.append(c.chunk_index)

    if not document.chunks:
        report.add_error("No chunks produced")

    return report


def _check_code_integrity(chunk: Chunk, report: ValidationReport) -> None:
    """Light heuristic: a code block in the chunk should end with ';' unless it is a
    fragment that continues on another page. Flag likely truncation only as a warning."""
    text = chunk.text
    for kw in ("CREATE", "SELECT", "INSERT", "UPDATE", "DELETE", "ALTER", "DROP", "GRANT", "REVOKE"):
        idx = text.upper().find(kw + " ")
        if idx != -1:
            segment = text[idx:]
            stripped = segment.rstrip()
            if not stripped.endswith(";") and not stripped.endswith("END"):
                if len(segment) < 200:
                    report.add_warning(
                        f"Chunk {chunk.chunk_index}: possible truncated code near '{kw}'"
                    )
            break


def _code_block_broken(chunk: Chunk) -> bool:
    """Heuristic: a chunk flagged as code should have balanced BEGIN/END if it uses them.

    A single BEGIN without a matching END (within a few of each other) strongly suggests
    the code block was split across chunks.
    """
    text = chunk.text.upper()
    begins = text.count("BEGIN")
    ends = text.count("END")
    if begins and ends and abs(begins - ends) <= 3 and begins != ends:
        return True
    return False
