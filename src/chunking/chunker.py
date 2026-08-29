from __future__ import annotations

import re
from typing import Dict, List, Tuple

from models.document import Chunk, ContentType, Document, LogicalGroup, Page

from config import Config
from metadata.metadata import compose_metadata

_TOKEN_RE = re.compile(r"\S+")


def count_tokens(text: str) -> int:
    """Approximate token count (word count). Good enough for pre-embedding chunk sizing."""
    return len(_TOKEN_RE.findall(text))


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


def _split_blocks(text: str, keywords: set) -> List[Tuple[str, bool]]:
    """Split page text into blocks, grouping consecutive code lines into one atomic block."""
    lines = text.split("\n")
    blocks: List[Tuple[str, bool]] = []
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if _is_code_line(line, keywords):
            buf = [line]
            i += 1
            while i < n and lines[i].strip():
                buf.append(lines[i])
                i += 1
            blocks.append(("\n".join(buf), True))
        else:
            blocks.append((line, False))
            i += 1
    return blocks


def _split_long_block(text: str, max_tokens: int) -> List[str]:
    parts = re.split(r"(?<=[.;])\s+|\n+", text)
    pieces: List[str] = []
    cur: List[str] = []
    cur_t = 0
    for p in parts:
        pt = count_tokens(p)
        if cur and cur_t + pt > max_tokens:
            pieces.append(" ".join(cur))
            cur, cur_t = [p], pt
        else:
            cur.append(p)
            cur_t += pt
    if cur:
        pieces.append(" ".join(cur))
    return pieces or [text]


def _block_tokens(btext: str) -> int:
    return count_tokens(btext)


def text_has_code(text: str, keywords: set) -> bool:
    """True if any line of ``text`` looks like a SQL/PLSQL statement."""
    return any(_is_code_line(line, keywords) for line in text.split("\n") if line.strip())


_PAGE_MAP: Dict[int, Page] = {}


def _set_page_map(pages: List[Page]) -> None:
    global _PAGE_MAP
    _PAGE_MAP = {p.page_number: p for p in pages}


def _content_type(has_code: bool, has_text: bool) -> ContentType:
    if has_code:
        return ContentType.CODE
    return ContentType.TEXT


def chunk_document(document: Document, groups: List[LogicalGroup], config: Config) -> List[Chunk]:
    _set_page_map(document.pages)
    pages_by_num: Dict[int, Page] = _PAGE_MAP
    keywords = set(config.sql_keywords)
    chunks: List[Chunk] = []
    idx = 0
    for group in groups:
        raw_chunks = _emit_raw_chunks(group, pages_by_num, config, keywords)
        merged = _merge_small(raw_chunks, config)
        chunks_for_group = _finalize(merged, group, document.source, config, idx)
        chunks.extend(chunks_for_group)
        idx += len(chunks_for_group)
    return chunks


def _emit_raw_chunks(
    group: LogicalGroup, pages_by_num: Dict[int, Page], config: Config, keywords: set
) -> List[Tuple[List[Tuple[str, bool, int]], set]]:
    emitted: List[Tuple[List[Tuple[str, bool, int]], set]] = []
    cur_blocks: List[Tuple[str, bool, int]] = []
    cur_tokens = 0
    cur_pages: set = set()

    def flush():
        nonlocal cur_blocks, cur_tokens, cur_pages
        if cur_blocks:
            emitted.append((cur_blocks, set(cur_pages)))
        cur_blocks = []
        cur_tokens = 0
        cur_pages = set()

    for pnum in group.page_numbers:
        page = pages_by_num[pnum]
        txt = page.text_for_chunking()
        # Text-only pipeline: pages with no extractable text contribute nothing.
        if not txt.strip():
            continue
        for btext, is_code in _split_blocks(txt, keywords):
            bt = _block_tokens(btext)
            if bt > config.chunk.max_tokens and not is_code:
                for piece in _split_long_block(btext, config.chunk.max_tokens):
                    emitted.append(([(piece, False, pnum)], {pnum}))
                continue
            if cur_blocks and cur_tokens + bt > config.chunk.max_tokens and cur_tokens >= config.chunk.min_tokens:
                flush()
            cur_blocks.append((btext, is_code, pnum))
            cur_tokens += bt
            cur_pages.add(pnum)
    flush()
    return emitted


def _merge_small(
    raw_chunks: List[Tuple[List[Tuple[str, bool, int]], set]], config: Config
) -> List[Tuple[List[Tuple[str, bool, int]], set]]:
    if not raw_chunks:
        return raw_chunks
    merged: List[Tuple[List[Tuple[str, bool, int]], set]] = []
    for blocks, pages in raw_chunks:
        t = sum(_block_tokens(b) for b, _, _ in blocks)
        if t >= config.chunk.min_tokens:
            merged.append((blocks, set(pages)))
            continue
        if merged:
            prev_blocks, prev_pages = merged[-1]
            merged[-1] = (prev_blocks + blocks, prev_pages | set(pages))
        else:
            merged.append((blocks, set(pages)))
    if len(merged) >= 2:
        last_b, last_p = merged[-1]
        if sum(_block_tokens(b) for b, _, _ in last_b) < config.chunk.min_tokens:
            prev_b, prev_p = merged[-2]
            merged[-2] = (prev_b + last_b, prev_p | last_p)
            merged.pop()
    return merged


def _finalize(
    merged: List[Tuple[List[Tuple[str, bool, int]], set]],
    group: LogicalGroup,
    source: str,
    config: Config,
    start_index: int,
) -> List[Chunk]:
    built: List[Tuple[str, int, set, bool]] = []
    for blocks, pages in merged:
        text = "\n".join(b for b, _, _ in blocks)
        has_code = any(is_code for _, is_code, _ in blocks)
        built.append((text, count_tokens(text), set(pages), has_code))

    ov = config.chunk.overlap_tokens
    for i in range(1, len(built)):
        prev_text = built[i - 1][0]
        prev_words = prev_text.split()
        overlap = prev_words[-ov:] if len(prev_words) >= ov else prev_words
        text, tok, pages, has_code = built[i]
        new_text = " ".join(overlap) + (" " + text if text else "")
        built[i] = (new_text, count_tokens(new_text), pages, has_code)

    chunks: List[Chunk] = []
    for i, (text, token_count, pages, has_code) in enumerate(built):
        pset = set(pages)
        content_type = _content_type(has_code, bool(text.strip()))
        has_visual = any(_PAGE_MAP[p].has_visual_content for p in pset)
        vision_model = next(
            (_PAGE_MAP[p].vision_model for p in pset if _PAGE_MAP[p].vision_model), ""
        )
        visual_status = next(
            (_PAGE_MAP[p].visual_processing_status for p in pset if _PAGE_MAP[p].visual_processing_status),
            "",
        )
        chunk = Chunk(
            chunk_index=start_index + i,
            text=text,
            source=source,
            page_start=min(pset),
            page_end=max(pset),
            chapter=group.chapter,
            slide_title=group.title,
            content_type=content_type,
            has_code=has_code,
            token_count=token_count,
            pages=sorted(pset),
            has_visual_content=has_visual,
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
            token_count=token_count,
            has_visual_content=has_visual,
            vision_model=vision_model,
            visual_processing_status=visual_status,
        )
        chunks.append(chunk)
    return chunks
