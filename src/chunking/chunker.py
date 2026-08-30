from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from models.document import Chunk, ContentType, Document, LogicalGroup, Page

from config import Config
from metadata.metadata import compose_metadata

_TOKEN_RE = re.compile(r"\S+")


def count_tokens(text: str) -> int:
    """Approximate token count (word count). Good enough for pre-embedding chunk sizing."""
    return len(_TOKEN_RE.findall(text))


# --------------------------------------------------------------------------- #
# Code / list / heading detection (structure-aware block splitting)
# --------------------------------------------------------------------------- #

# PL/SQL / SQL tokens that, when starting a (non-prose) line, indicate code.
_CODE_START = {
    "BEGIN", "DECLARE", "CREATE", "ALTER", "DROP", "SELECT", "INSERT", "UPDATE",
    "DELETE", "MERGE", "TRUNCATE", "COMMENT", "GRANT", "REVOKE", "CURSOR", "LOOP",
    "END", "IF", "THEN", "ELSE", "ELSIF", "FOR", "WHILE", "TYPE", "PROCEDURE",
    "FUNCTION", "TRIGGER", "PACKAGE", "EXCEPTION", "RETURN", "EXIT", "VARRAY",
    "VARYING", "DBMS_OUTPUT", "PRAGMA", "RAISE", "WHEN", "CASE",
}

_LIST_RE = re.compile(r"^\s*([•\-*+\u2013\u2014]|\d+[.)])\s+\S")
_HEAD_NUM_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)[\.\)]?\s+\S")
_SPECIAL_RE = re.compile(
    r"^\s*(exemple|example|note|remarque|d\u00e9finition|definition)\b", re.IGNORECASE
)


def _is_code_line(line: str, keywords: set) -> bool:
    s = line.strip()
    if not s:
        return False
    first = s.split()[0].upper().strip("();")
    if first in keywords:
        return True
    if s.endswith(";") and any(k in s.upper() for k in keywords):
        return True
    return False


def _looks_like_code(line: str, keywords: set) -> bool:
    s = line.strip()
    if not s:
        return False
    if _is_code_line(s, keywords):
        return True
    if ":=" in s:
        return True
    if s.endswith(";") and len(s.split()) >= 2:
        return True
    first = s.split()[0].upper().strip("();")
    if first in _CODE_START:
        return True
    return False


def _is_list_item(line: str) -> bool:
    return bool(_LIST_RE.match(line))


def _is_special(line: str) -> Optional[str]:
    m = _SPECIAL_RE.match(line)
    if not m:
        return None
    word = m.group(1).lower()
    if word.startswith("exemple") or word.startswith("example"):
        return "example"
    if word.startswith("note") or word.startswith("remarque"):
        return "note"
    return "definition"


def _is_heading(line: str, next_line: Optional[str], keywords: set) -> bool:
    s = line.strip()
    if not s or len(s) > 90:
        return False
    # Numbered headings (e.g. "1. Variables", "2.3 Details") take priority over list items.
    if _HEAD_NUM_RE.match(s):
        return True
    if _looks_like_code(s, keywords) or _is_list_item(s):
        return False
    if s.count(".") >= 3:  # table-of-contents leader line
        return False
    if s.upper() == s and len(s) > 3:
        return True
    nxt = (next_line or "").strip()
    if nxt == "" or nxt.startswith("(p."):
        return True
    if len(s) <= 60 and s[0].isupper() and not s.endswith((".", ";", ":")):
        return True
    return False


# --------------------------------------------------------------------------- #
# Block model
# --------------------------------------------------------------------------- #


@dataclass
class Block:
    text: str
    kind: str  # heading | paragraph | code | list | table | definition | example | note
    level: int  # heading depth (1 = section, 2+ = subsection)
    page: int
    atomic: bool = False


def _classify_run(buf: List[str], first: str, nxt: Optional[str], keywords: set) -> Tuple[str, int]:
    special = _is_special(first)
    if special:
        return special, 0
    # Code and headings are detected before list so that a numbered section title
    # (e.g. "1. Variables") is not mistaken for an ordered list item.
    if _looks_like_code(first, keywords) or any(_looks_like_code(l, keywords) for l in buf):
        codeish = sum(1 for l in buf if _looks_like_code(l, keywords))
        if _looks_like_code(first, keywords) or (len(buf) >= 3 and codeish > len(buf) / 2):
            return "code", 0
    if _is_heading(first, nxt, keywords):
        level = 1
        m = _HEAD_NUM_RE.match(first)
        if m:
            level = len(m.group(1).split("."))
        return "heading", level
    if _is_list_item(first):
        return "list", 0
    if "|" in first and any("|" in l for l in buf):
        return "table", 0
    return "paragraph", 0


def split_into_blocks(text: str, page: int, keywords: set) -> List[Block]:
    """Split a page's text into typed, atomic-where-needed semantic blocks."""
    lines = text.split("\n")
    runs: List[List[str]] = []
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        buf = [line]
        i += 1
        while i < n and lines[i].strip():
            buf.append(lines[i])
            i += 1
        runs.append(buf)

    blocks: List[Block] = []
    for idx, buf in enumerate(runs):
        first = buf[0].strip()
        nxt = runs[idx + 1][0] if idx + 1 < len(runs) else None
        kind, level = _classify_run(buf, first, nxt, keywords)
        blocks.append(
            Block(
                text="\n".join(buf),
                kind=kind,
                level=level,
                page=page,
                atomic=kind in ("code", "table", "definition", "example", "note"),
            )
        )

    # Keep example/note/definition together with the explanation that follows.
    merged: List[Block] = []
    i = 0
    while i < len(blocks):
        b = blocks[i]
        if b.kind in ("example", "note", "definition"):
            j = i + 1
            while j < len(blocks) and blocks[j].kind == "paragraph":
                b = Block(
                    text=b.text + "\n" + blocks[j].text,
                    kind=b.kind,
                    level=b.level,
                    page=b.page,
                    atomic=True,
                )
                j += 1
            merged.append(b)
            i = j
        else:
            merged.append(b)
            i += 1
    return merged


# --------------------------------------------------------------------------- #
# Chunk assembly
# --------------------------------------------------------------------------- #


def text_has_code(text: str, keywords: set) -> bool:
    """True if any line of ``text`` looks like a SQL/PLSQL statement."""
    return any(_is_code_line(line, keywords) for line in text.split("\n") if line.strip())


def _content_type(has_code: bool, has_text: bool) -> ContentType:
    if has_code:
        return ContentType.CODE
    return ContentType.TEXT


def _split_overlong(block: Block, max_tokens: int) -> List[Block]:
    """Split a non-atomic block that exceeds the size limit.

    Prefers sentence boundaries; falls back to word-level splitting when a block has
    neither sentence punctuation nor newlines. Atomic blocks (code, tables,
    definitions, examples, notes) are never split.
    """
    if block.atomic or count_tokens(block.text) <= max_tokens:
        return [block]
    pieces = re.split(r"(?<=[.;])\s+|\n+", block.text)
    out: List[str] = []
    for p in pieces:
        if count_tokens(p) > max_tokens:
            words = p.split()
            for i in range(0, len(words), max_tokens):
                out.append(" ".join(words[i : i + max_tokens]))
        else:
            out.append(p)
    if len(out) <= 1:
        words = block.text.split()
        out = [" ".join(words[i : i + max_tokens]) for i in range(0, len(words), max_tokens)]
    return [Block(p, block.kind, block.level, block.page, block.atomic) for p in out] or [block]


def _assemble_chunks(
    blocks: List[Block],
    group: LogicalGroup,
    source: str,
    config: Config,
    start_index: int,
) -> List[Chunk]:
    cfg = config.chunk
    section_stack: Dict[int, str] = {}
    section: Optional[str] = None
    subsection: Optional[str] = None
    cur_blocks: List[Block] = []
    cur_tokens = 0
    built: List[Tuple[List[Block], Optional[str], Optional[str]]] = []

    def flush() -> None:
        nonlocal cur_blocks, cur_tokens
        if cur_blocks:
            built.append((cur_blocks, section, subsection))
        cur_blocks = []
        cur_tokens = 0

    for blk in blocks:
        if blk.kind == "heading":
            level = blk.level or 1
            # Use only the first line as the heading title (the rest may be body if the
            # heading and body were not separated by a blank line).
            title = blk.text.strip().split("\n", 1)[0].strip()[:120]
            section_stack[level] = title
            for lv in [k for k in section_stack if k > level]:
                del section_stack[lv]
            section = section_stack.get(1)
            subsection = section_stack.get(2) or section_stack.get(3)
            # Start a fresh chunk on a new major (level-1) heading once we have enough.
            if level == 1 and cur_blocks and cur_tokens >= cfg.min_tokens:
                flush()
            cur_blocks.append(blk)
            cur_tokens += count_tokens(blk.text)
            continue

        bt = count_tokens(blk.text)
        if cur_blocks and cur_tokens + bt > cfg.max_tokens and cur_tokens >= cfg.min_tokens:
            flush()
        cur_blocks.append(blk)
        cur_tokens += bt
    flush()

    built = _merge_small(built, cfg)
    built = _apply_overlap(built, cfg)

    chunks: List[Chunk] = []
    for i, (blks, sec, sub, overlap) in enumerate(built):
        body = "\n".join(b.text for b in blks)
        has_code = any(b.kind == "code" for b in blks)
        pages = sorted({b.page for b in blks})
        code_blocks = sum(1 for b in blks if b.kind == "code")
        token_count = count_tokens(body)

        # Hierarchy context (Title = chapter, Section, Subsection) for self-containment.
        ctx: List[str] = []
        if group.chapter:
            ctx.append(f"Title: {group.chapter}")
        if sec:
            ctx.append(f"Section: {sec}")
        if sub:
            ctx.append(f"Subsection: {sub}")
        content = ("\n".join(ctx) + "\n\n" + body) if ctx else body
        # Overlap: prepend trailing prose context from the previous chunk for continuity.
        if overlap:
            content = overlap + "\n\n" + content

        content_type = _content_type(has_code, bool(body.strip()))
        chunk_id = f"{source}::chunk-{start_index + i:04d}"
        chunk = Chunk(
            chunk_index=start_index + i,
            text=content,
            source=source,
            page_start=min(pages),
            page_end=max(pages),
            chapter=group.chapter,
            slide_title=group.title,
            content_type=content_type,
            has_code=has_code,
            token_count=count_tokens(content),
            pages=pages,
            section=sec,
            subsection=sub,
            metadata={},
        )
        chunk.metadata = compose_metadata(
            source=source,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            chapter=group.chapter,
            slide_title=group.title,
            content_type=content_type,
            has_code=has_code,
            chunk_index=chunk.chunk_index,
            pages=chunk.pages,
            token_count=chunk.token_count,
            chunk_id=chunk_id,
            section=sec,
            subsection=sub,
            content=content,
            code_blocks=code_blocks,
        )
        chunks.append(chunk)
    return chunks


def _merge_small(
    built: List[Tuple[List[Block], Optional[str], Optional[str]]], cfg
) -> List[Tuple[List[Block], Optional[str], Optional[str]]]:
    if not built:
        return built
    merged: List[Tuple[List[Block], Optional[str], Optional[str]]] = []
    for blks, sec, sub in built:
        t = sum(count_tokens(b.text) for b in blks)
        if t >= cfg.min_tokens:
            merged.append((blks, sec, sub))
            continue
        if merged:
            prev_blks, prev_sec, prev_sub = merged[-1]
            merged[-1] = (prev_blks + blks, prev_sec or sec, prev_sub or sub)
        else:
            merged.append((blks, sec, sub))
    if len(merged) >= 2:
        last_blks, last_sec, last_sub = merged[-1]
        if sum(count_tokens(b.text) for b in last_blks) < cfg.min_tokens:
            prev_blks, prev_sec, prev_sub = merged[-2]
            merged[-2] = (prev_blks + last_blks, prev_sec or last_sec, prev_sub or last_sub)
            merged.pop()
    return merged


def _apply_overlap(
    built: List[Tuple[List[Block], Optional[str], Optional[str]]], cfg
) -> List[Tuple[List[Block], Optional[str], Optional[str], str]]:
    if cfg.overlap_tokens <= 0 or len(built) < 2:
        return [(b, s, sub, "") for b, s, sub in built]
    ov = cfg.overlap_tokens
    out: List[Tuple[List[Block], Optional[str], Optional[str], str]] = []
    for i, (blks, sec, sub) in enumerate(built):
        if i == 0:
            out.append((blks, sec, sub, ""))
            continue
        prev_blks = built[i - 1][0]
        overlap = ""
        # Never carry a partial code block; only overlap trailing prose context.
        if prev_blks[-1].kind != "code":
            words = "\n".join(b.text for b in prev_blks).split()
            overlap = " ".join(words[-ov:]) if len(words) >= ov else " ".join(words)
        out.append((blks, sec, sub, overlap))
    return out


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #


def chunk_document(document: Document, groups: List[LogicalGroup], config: Config) -> List[Chunk]:
    """Structure-aware semantic chunking.

    The document structure (chapters / groups from the structure detector) is the
    top level; inside each group the page text is split into typed blocks (headings,
    paragraphs, code, lists, tables, definitions, examples, notes) and assembled into
    self-contained chunks that respect size guidance and never split atomic content
    (code blocks, tables, definitions, examples). A small overlap of trailing prose
    is carried into the next chunk for continuity.
    """
    pages_by_num: Dict[int, Page] = {p.page_number: p for p in document.pages}
    keywords = set(config.sql_keywords)
    chunks: List[Chunk] = []
    idx = 0
    for group in groups:
        blocks: List[Block] = []
        for pnum in group.page_numbers:
            page = pages_by_num.get(pnum)
            if page is None:
                continue
            txt = page.text_for_chunking()
            if not txt.strip():
                continue
            blocks.extend(split_into_blocks(txt, pnum, keywords))
        if not blocks:
            continue
        # Keep non-atomic blocks within the size limit (split at sentence boundaries).
        expanded: List[Block] = []
        for b in blocks:
            expanded.extend(_split_overlong(b, config.chunk.max_tokens))
        blocks = expanded
        group_chunks = _assemble_chunks(blocks, group, document.source, config, idx)
        chunks.extend(group_chunks)
        idx += len(group_chunks)
    return chunks
