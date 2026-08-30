from __future__ import annotations

from pathlib import Path

import pytest

from config import Config, load_config
from ingestion.pdf_loader import run_pipeline

ROOT = Path(__file__).resolve().parents[2]
ORACLE = ROOT / "data" / "raw" / "oracle" / "Cours_Oracle_Complet.pdf"
PLSQL = ROOT / "data" / "raw" / "plsql" / "PLSQL Version Finale (1).pdf"


def _covered(chunks):
    s = set()
    for c in chunks:
        s.update(c.pages)
    return s


@pytest.mark.skipif(not ORACLE.exists(), reason="Oracle PDF not present")
def test_oracle_pipeline(tmp_path):
    cfg = Config.default()
    doc, report = run_pipeline(str(ORACLE), cfg, str(tmp_path))
    # Every PDF page is recorded as a Page (text-only pipeline keeps page continuity).
    assert len(doc.pages) == 133
    # No fabricated Chapter 2: exactly Chapitre 1 and Chapitre 3 detected.
    assert report["chapters_detected"] == 2
    # Pages that carry extractable text are represented as chunks.
    covered = _covered(doc.chunks)
    text_pages = [p.page_number for p in doc.pages if p.text_for_chunking().strip()]
    assert all(p in covered for p in text_pages)
    assert report["validation_errors"] == 0


@pytest.mark.skipif(not PLSQL.exists(), reason="PL/SQL PDF not present")
def test_plsql_pipeline(tmp_path):
    cfg = Config.default()
    doc, report = run_pipeline(str(PLSQL), cfg, str(tmp_path))
    assert len(doc.pages) == 155
    covered = _covered(doc.chunks)
    text_pages = [p.page_number for p in doc.pages if p.text_for_chunking().strip()]
    assert all(p in covered for p in text_pages)
    assert report["validation_errors"] == 0


@pytest.mark.skipif(not ORACLE.exists(), reason="Oracle PDF not present")
def test_oracle_chapter_one_and_three_detection(tmp_path):
    cfg = Config.default()
    doc, report = run_pipeline(str(ORACLE), cfg, str(tmp_path))
    chapters = " ".join(report["chapters"])
    assert "Chapitre 1" in chapters
    assert "Chapitre 3" in chapters
    assert "Chapitre 2" not in chapters
