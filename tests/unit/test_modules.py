from __future__ import annotations

from models.document import Document, Page, PageClass
from config import Config
from classification.page_classifier import classify_page
from cleaning.cleaner import TextCleaner
from structure.detector import StructureDetector
from metadata.metadata import compose_metadata
from validation.validator import validate


def test_classify_thresholds():
    cfg = Config.default()
    assert classify_page(0, cfg) == PageClass.NO_TEXT
    assert classify_page(10, cfg) == PageClass.NO_TEXT
    assert classify_page(50, cfg) == PageClass.HYBRID
    assert classify_page(200, cfg) == PageClass.NATIVE_TEXT
    assert classify_page(400, cfg) == PageClass.NATIVE_TEXT


def test_cleaner_removes_boilerplate_and_footer():
    cfg = Config.default()
    c = TextCleaner(cfg)
    # "footer line" appears on every page (boilerplate); "real content" on only 1/10.
    texts = ["footer line\nreal content"] + ["footer line\nother real" for _ in range(9)]
    c.fit(texts)
    out = c.clean("footer line\nreal content")
    assert "footer line" not in out
    assert "real content" in out


def test_cleaner_removes_email_and_pagenum():
    cfg = Config.default()
    c = TextCleaner(cfg)
    c.fit(["x", "y", "z"])
    assert c.clean("a.amine@usms.ma") == ""
    assert c.clean("50") == ""
    assert "keep" in c.clean("keep this line")


def test_detector_groups_consecutive_titles_and_chapters():
    cfg = Config.default()
    pages = [
        Page(page_number=1, source="t.pdf", raw_text="Chapitre 1 : X", cleaned_text="Chapitre 1 : X"),
        Page(page_number=2, source="t.pdf", raw_text="Title A", cleaned_text="Title A"),
        Page(page_number=3, source="t.pdf", raw_text="Title A", cleaned_text="Title A"),
        Page(page_number=4, source="t.pdf", raw_text="Title B", cleaned_text="Title B"),
    ]
    doc = Document(source="t.pdf", pages=pages)
    groups = StructureDetector(cfg).detect(doc)
    assert groups[0].chapter.startswith("Chapitre 1")
    titles = [g.title for g in groups]
    assert titles.count("Title A") == 1  # pages 2-3 merged
    assert len(groups) == 3


def test_metadata_keys():
    m = compose_metadata(
        source="s.pdf", page_start=1, page_end=2, chapter="C", slide_title="T",
        content_type=__import__("models.document", fromlist=["ContentType"]).ContentType.TEXT,
        has_code=False, chunk_index=0, pages=[1, 2], token_count=10,
    )
    for k in ["source", "page_start", "page_end", "chapter", "slide_title",
              "content_type", "has_code", "chunk_index"]:
        assert k in m
    assert "subsection" not in m
    assert "is_ocr" not in m


def test_validator_flags_empty_chunk():
    from models.document import Chunk

    CT = __import__("models.document", fromlist=["ContentType"]).ContentType.TEXT
    good = Chunk(0, "texte valide", "s.pdf", 1, 1, "C", "T", CT, False, 10, pages=[1], metadata={})
    bad = Chunk(1, "", "s.pdf", 2, 2, "C", "T", CT, False, 0, pages=[2], metadata={})
    doc = Document(source="s.pdf", pages=[], chunks=[good, bad])
    report = validate(doc, Config.default())
    assert report.ok is False
    assert any("empty text" in e for e in report.errors)
