from __future__ import annotations

from models.document import ContentType, Document, LogicalGroup, Page, PageClass
from config import Config
from chunking.chunker import chunk_document, count_tokens


def make_page(num, text, cls=PageClass.NATIVE_TEXT, source="t.pdf"):
    p = Page(page_number=num, source=source, raw_text=text)
    p.final_text = text
    p.cleaned_text = text
    p.page_class = cls
    return p


def test_count_tokens_basic():
    assert count_tokens("un deux trois") == 3
    assert count_tokens("") == 0


def test_single_page_chunk():
    pages = [make_page(1, "hello world")]
    doc = Document(source="t.pdf", pages=pages)
    groups = [LogicalGroup(chapter="C1", title="T", page_numbers=[1])]
    chunks = chunk_document(doc, groups, Config.default())
    assert len(chunks) == 1
    assert chunks[0].page_start == 1 and chunks[0].page_end == 1
    assert chunks[0].metadata["source"] == "t.pdf"


def test_code_not_split_mid_statement():
    big = "CREATE TABLESPACE ts\nDATAFILE 'a.dbf' SIZE 10M\nEXTENT MANAGEMENT LOCAL;"
    pages = [make_page(1, big)]
    doc = Document(source="t.pdf", pages=pages)
    groups = [LogicalGroup(chapter="C", title="T", page_numbers=[1])]
    chunks = chunk_document(doc, groups, Config.default())
    assert len(chunks) == 1
    assert chunks[0].has_code is True
    assert chunks[0].text.strip().endswith(";")


def test_overlap_between_chunks():
    cfg = Config.default()
    cfg.chunk.target_tokens = 300
    cfg.chunk.max_tokens = 400
    cfg.chunk.overlap_tokens = 20
    p1 = " ".join(f"w{i}" for i in range(300))
    p2 = " ".join(f"x{i}" for i in range(300))
    pages = [make_page(1, p1), make_page(2, p2)]
    doc = Document(source="t.pdf", pages=pages)
    groups = [LogicalGroup(chapter="C", title="T", page_numbers=[1, 2])]
    chunks = chunk_document(doc, groups, cfg)
    assert len(chunks) >= 2
    last_words = chunks[0].text.split()[-20:]
    assert all(w in chunks[1].text.split() for w in last_words)


def test_small_chunks_merged_within_group():
    cfg = Config.default()
    pages = [make_page(1, "petit a"), make_page(2, "petit b"), make_page(3, "petit c")]
    doc = Document(source="t.pdf", pages=pages)
    groups = [LogicalGroup(chapter="C", title="T", page_numbers=[1, 2, 3])]
    chunks = chunk_document(doc, groups, cfg)
    # all below min_tokens but in same group -> merged into one
    assert len(chunks) == 1
    assert chunks[0].pages == [1, 2, 3]


def test_no_cross_group_merge():
    cfg = Config.default()
    p1 = make_page(1, "groupe un contenu court")
    p2 = make_page(2, "groupe deux contenu court")
    doc = Document(source="t.pdf", pages=[p1, p2])
    groups = [
        LogicalGroup(chapter="C", title="A", page_numbers=[1]),
        LogicalGroup(chapter="C", title="B", page_numbers=[2]),
    ]
    chunks = chunk_document(doc, groups, cfg)
    assert len(chunks) == 2
    assert chunks[0].pages == [1]
    assert chunks[1].pages == [2]


def test_native_text_page_is_text_content_type():
    p = make_page(1, "vrai contenu textuel ici")
    doc = Document(source="t.pdf", pages=[p])
    groups = [LogicalGroup(chapter="C", title="T", page_numbers=[1])]
    chunks = chunk_document(doc, groups, Config.default())
    assert chunks[0].content_type == ContentType.TEXT


def test_code_page_is_code_content_type():
    p = make_page(1, "SELECT * FROM emp;")
    doc = Document(source="t.pdf", pages=[p])
    groups = [LogicalGroup(chapter="C", title="T", page_numbers=[1])]
    chunks = chunk_document(doc, groups, Config.default())
    assert chunks[0].content_type == ContentType.CODE


def test_page_without_text_produces_no_chunk():
    # Text-only pipeline: a page with no extractable text yields no chunk,
    # but the Page record itself is preserved by the orchestrator.
    p = make_page(1, "", cls=PageClass.NO_TEXT)
    doc = Document(source="t.pdf", pages=[p])
    groups = [LogicalGroup(chapter="C", title="D", page_numbers=[1])]
    chunks = chunk_document(doc, groups, Config.default())
    assert chunks == []
