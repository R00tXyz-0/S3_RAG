from __future__ import annotations

from models.document import (
    Chunk,
    ContentType,
    Document,
    LogicalGroup,
    Page,
    PageClass,
)
from config import Config
from chunking.chunker import chunk_document, count_tokens
from validation.validator import validate


def make_page(num, text, cls=PageClass.NATIVE_TEXT, source="t.pdf"):
    p = Page(page_number=num, source=source, raw_text=text)
    p.final_text = text
    p.cleaned_text = text
    p.page_class = cls
    p.extraction_status = "extracted"
    return p


def run(pages, text, groups, cfg=None):
    cfg = cfg or Config.default()
    doc = Document(source="t.pdf", pages=pages)
    chunks = chunk_document(doc, groups, cfg)
    return doc, chunks


def test_chunks_are_generated():
    pages = [make_page(1, "Introduction au PL/SQL. C'est un langage procedurale.")]
    groups = [LogicalGroup(chapter="PL/SQL", title="Intro", page_numbers=[1])]
    doc, chunks = run(pages, "", groups)
    assert chunks, "expected at least one chunk"
    assert all(c.text.strip() for c in chunks)


def test_chunk_metadata_present():
    pages = [make_page(1, "Contenu textuel simple.")]
    groups = [LogicalGroup(chapter="PL/SQL", title="T", page_numbers=[1])]
    doc, chunks = run(pages, "", groups)
    for c in chunks:
        for key in ("chunk_id", "source", "page_start", "page_end",
                    "section", "subsection", "content_type", "has_code",
                    "chunk_index", "token_count", "pages", "content"):
            assert key in c.metadata, f"missing metadata key {key}"
        assert c.metadata["chunk_id"]
        assert c.metadata["source"] == "t.pdf"
        assert c.metadata["content"] == c.text


def test_source_and_page_information_preserved():
    pages = [make_page(1, "page une."), make_page(2, "page deux.")]
    groups = [LogicalGroup(chapter="PL/SQL", title="T", page_numbers=[1, 2])]
    doc, chunks = run(pages, "", groups)
    all_pages = {p for c in chunks for p in c.pages}
    assert all_pages <= {1, 2}
    for c in chunks:
        assert c.page_start <= c.page_end
        assert c.metadata["page_start"] == c.page_start
        assert c.metadata["page_end"] == c.page_end


def test_headings_sections_preserved():
    text = (
        "1. Variables\n"
        "Les variables se declarent avec un nom et un type.\n"
        "2. Constantes\n"
        "Une constante ne change pas de valeur."
    )
    pages = [make_page(1, text)]
    groups = [LogicalGroup(chapter="PL/SQL", title="T", page_numbers=[1])]
    doc, chunks = run(pages, "", groups)
    joined = "\n".join(c.text for c in chunks)
    # Heading text survives and a section label is attached.
    assert "Variables" in joined
    assert any(c.metadata.get("section") for c in chunks), "expected a detected section"
    assert any("Section:" in c.text for c in chunks), "expected Section: context in content"


def test_code_block_not_broken():
    code = (
        "DECLARE\n"
        "  x NUMBER;\n"
        "BEGIN\n"
        "  SELECT a INTO x FROM t WHERE id = 1;\n"
        "  IF x > 0 THEN\n"
        "    DBMS_OUTPUT.PUT_LINE('ok');\n"
        "  END IF;\n"
        "END;"
    )
    pages = [make_page(1, code)]
    groups = [LogicalGroup(chapter="PL/SQL", title="T", page_numbers=[1])]
    doc, chunks = run(pages, "", groups)
    # The whole code block must appear intact in a single chunk.
    assert any(code in c.text for c in chunks), "code block was split"
    assert any(c.text.strip().endswith("END;") for c in chunks)


def test_chunk_size_reasonable():
    # One very long paragraph must be split so no chunk is absurdly large.
    cfg = Config.default()
    big = " ".join(f"mot{i}" for i in range(1500))
    pages = [make_page(1, big)]
    groups = [LogicalGroup(chapter="PL/SQL", title="T", page_numbers=[1])]
    doc, chunks = run(pages, "", groups, cfg)
    assert len(chunks) >= 2
    for c in chunks:
        # allow headroom for the overlap + hierarchy-context prefix
        assert c.token_count <= cfg.chunk.max_tokens + cfg.chunk.overlap_tokens + 150


def test_overlap_works():
    cfg = Config.default()
    cfg.chunk.target_tokens = 300
    cfg.chunk.max_tokens = 400
    cfg.chunk.overlap_tokens = 20
    p1 = " ".join(f"w{i}" for i in range(300))
    p2 = " ".join(f"x{i}" for i in range(300))
    pages = [make_page(1, p1), make_page(2, p2)]
    groups = [LogicalGroup(chapter="PL/SQL", title="T", page_numbers=[1, 2])]
    doc, chunks = run(pages, "", groups, cfg)
    assert len(chunks) >= 2
    last_words = chunks[0].text.split()[-20:]
    assert all(w in chunks[1].text.split() for w in last_words)


def test_no_content_lost_during_chunking():
    text1 = "Premiere partie du cours sur les variables et les constantes en PL/SQL."
    text2 = "Seconde partie qui parle des boucles et du controle de flux avec LOOP."
    text3 = "Troisieme partie concernant les curseurs explicites et implicites."
    pages = [make_page(1, text1), make_page(2, text2), make_page(3, text3)]
    groups = [LogicalGroup(chapter="PL/SQL", title="T", page_numbers=[1, 2, 3])]
    doc, chunks = run(pages, "", groups)
    joined = "\n".join(c.text for c in chunks).lower()
    for text in (text1, text2, text3):
        for word in text.lower().split():
            assert word in joined, f"lost word: {word}"


def test_validator_flags_extremely_small_and_missing_source():
    CT = ContentType.TEXT
    pages = [Page(page_number=1, source="s.pdf", raw_text="x", final_text="x",
                 extraction_status="extracted")]
    good = Chunk(0, "texte valide suffisamment long pour depasser le minimum", "s.pdf",
                 1, 1, "C", "T", CT, False, 60, pages=[1], metadata={"source": "s.pdf"})
    tiny = Chunk(1, "x", "s.pdf", 2, 2, "C", "T", CT, False, 1, pages=[2],
                 metadata={"source": "s.pdf"})
    nosrc = Chunk(2, "autre texte valide assez long lui aussi", "", 3, 3, "C", "T",
                  CT, False, 60, pages=[3], metadata={"source": ""})
    doc = Document(source="s.pdf", pages=pages, chunks=[good, tiny, nosrc])
    rep = validate(doc, Config.default())
    assert 1 in rep.suspicious  # extremely small
    assert 2 in rep.suspicious  # missing source metadata
    assert any("missing source" in e for e in rep.errors)


def test_validator_flags_missing_section_when_sections_exist():
    CT = ContentType.TEXT
    pages = [Page(page_number=1, source="s.pdf", raw_text="x", final_text="x",
                 extraction_status="extracted")]
    with_sec = Chunk(0, "texte avec section", "s.pdf", 1, 1, "C", "T", CT, False, 40,
                     pages=[1], metadata={"source": "s.pdf", "section": "Variables"})
    without_sec = Chunk(1, "texte sans section", "s.pdf", 2, 2, "C", "T", CT, False, 40,
                        pages=[2], metadata={"source": "s.pdf", "section": None})
    doc = Document(source="s.pdf", pages=pages, chunks=[with_sec, without_sec])
    rep = validate(doc, Config.default())
    assert 1 in rep.suspicious
    assert any("missing section" in w for w in rep.warnings)
