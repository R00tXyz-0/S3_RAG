from __future__ import annotations

from pathlib import Path

import pytest

from config import Config, load_config
from ingestion.pdf_loader import run_pipeline
from tests.unit.test_vision import FakeRenderer
from vision.detector import VisualContentDetector
from vision.model import GeminiVisionModel
from vision.stage import VisionStage

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
    cfg.vision.enabled = False  # keep this integrity test offline/fast
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
    cfg.vision.enabled = False  # keep this integrity test offline/fast
    doc, report = run_pipeline(str(PLSQL), cfg, str(tmp_path))
    assert len(doc.pages) == 155
    covered = _covered(doc.chunks)
    text_pages = [p.page_number for p in doc.pages if p.text_for_chunking().strip()]
    assert all(p in covered for p in text_pages)
    assert report["validation_errors"] == 0


@pytest.mark.skipif(not ORACLE.exists(), reason="Oracle PDF not present")
def test_oracle_chapter_one_and_three_detection(tmp_path):
    cfg = Config.default()
    cfg.vision.enabled = False  # keep this integrity test offline/fast
    doc, report = run_pipeline(str(ORACLE), cfg, str(tmp_path))
    chapters = " ".join(report["chapters"])
    assert "Chapitre 1" in chapters
    assert "Chapitre 3" in chapters
    assert "Chapitre 2" not in chapters


@pytest.mark.skipif(not ORACLE.exists(), reason="Oracle PDF not present")
def test_oracle_pipeline_with_vision_stage(tmp_path):
    """End-to-end run with a fake (mocked) vision model injected.

    This exercises the real merge/clean/chunk path with vision output while avoiding
    any dependency on transformers/torch/PyMuPDF at test time.
    """
    cfg = Config.default()
    calls = []

    def fake_generate(image, prompt):
        calls.append(prompt)
        return "VISUAL: architecture of the Oracle instance."

    stage = VisionStage(
        model=GeminiVisionModel("Qwen/fake", generate_fn=fake_generate),
        renderer=FakeRenderer(mode="new"),
        detector=VisualContentDetector(cfg.vision),
        config=cfg.vision,
        cache_dir=str(tmp_path),
    )
    doc, report = run_pipeline(str(ORACLE), cfg, str(tmp_path), vision_stage=stage)
    # No page loses its native text; vision output is additive.
    assert report["validation_errors"] == 0
    assert report["chunks_with_visual_content"] > 0
    assert report["vision_model"] == "Qwen/fake"
    assert any("Oracle" in c.text for c in doc.chunks)
    assert any("VISUAL" in c.text for c in doc.chunks)
    # The vision prompt received chapter/slide context.
    assert any("Chapitre" in p for p in calls)
