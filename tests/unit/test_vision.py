from __future__ import annotations

from models.document import ContentType, Document, LogicalGroup, Page, PageClass

from chunking.chunker import chunk_document, text_has_code
from config import Config
from vision.detector import VisualContentDetector
from vision.merger import merge_texts
from vision.model import GeminiVisionModel
from vision.stage import VisionStage


class FakeRenderer:
    def __init__(self, mode="new"):
        self.mode = mode
        self._same = None

    def render(self, pdf_path, page_index):
        if self.mode == "raise":
            raise RuntimeError("render failed")
        if self.mode == "same":
            if self._same is None:
                from PIL import Image

                self._same = Image.new("RGB", (10, 10))
            return self._same
        from PIL import Image

        return Image.new("RGB", (10, 10))


class FakeDetector:
    def __init__(self, process=True):
        self.process = process
        self.build_calls = []

    def has_meaningful_visual(self, page, pypdf_page):
        return self.process

    def build_prompt(self, source, chapter, slide_title):
        self.build_calls.append((source, chapter, slide_title))
        return f"PROMPT|{chapter}|{slide_title}"


def _page(page_number, page_class=PageClass.NATIVE_TEXT, raw_text="native text"):
    return Page(page_number=page_number, source="S.pdf", raw_text=raw_text, page_class=page_class)


def _run_stage(pages, detector, model, renderer, config=None, page_context=None):
    import tempfile

    cfg = (config or Config.default()).vision
    # Isolate the cache per call so tests don't contaminate each other via the repo cache dir.
    stage = VisionStage(
        model=model, renderer=renderer, detector=detector, config=cfg, cache_dir=tempfile.mkdtemp()
    )
    ctx = page_context or {p.page_number: ("Chap", f"Slide {p.page_number}") for p in pages}
    pypdf_pages = {p.page_number: None for p in pages}
    stage.process("S.pdf", pages, pypdf_pages, ctx)
    return pages


def test_native_skipped_when_no_visual():
    page = _page(1)
    det = FakeDetector(process=False)
    fn = lambda img, prompt: "VISUAL"
    _run_stage([page], det, GeminiVisionModel("m", generate_fn=fn), FakeRenderer())
    assert page.visual_processing_status == "skipped_no_visual"
    assert page.has_visual_content is False
    assert page.visual_text == ""
    assert merge_texts(page.raw_text, page.visual_text) == "native text"


def test_image_only_page_generated():
    page = _page(2, page_class=PageClass.NO_TEXT, raw_text="")
    det = FakeDetector(process=True)
    fn = lambda img, prompt: "diagram description"
    _run_stage([page], det, GeminiVisionModel("m", generate_fn=fn), FakeRenderer())
    assert page.visual_processing_status == "generated"
    assert page.has_visual_content is True
    assert page.visual_text == "diagram description"
    assert page.vision_model == "m"
    assert merge_texts(page.raw_text, page.visual_text) == "diagram description"


def test_hybrid_page_merged_native_then_visual():
    page = _page(3, page_class=PageClass.HYBRID, raw_text="title only")
    det = FakeDetector(process=True)
    fn = lambda img, prompt: "body from image"
    _run_stage([page], det, GeminiVisionModel("m", generate_fn=fn), FakeRenderer())
    assert page.has_visual_content is True
    assert merge_texts(page.raw_text, page.visual_text) == "title only\n\nbody from image"


def test_detector_flags_native_page_with_large_image():
    from PIL import Image

    class FakeImg:
        def __init__(self, w, h):
            self.image = Image.new("RGB", (w, h))

    class FakePdfPage:
        images = [FakeImg(200, 200)]  # 40000 px > default min_image_area 20000

    det = VisualContentDetector(Config.default().vision)
    page = _page(4, page_class=PageClass.NATIVE_TEXT, raw_text="x")
    assert det.has_meaningful_visual(page, FakePdfPage()) is True
    # A tiny decorative icon should not trigger inference.
    class FakePdfPageSmall:
        images = [FakeImg(20, 20)]

    assert det.has_meaningful_visual(page, FakePdfPageSmall()) is False


def test_sql_in_visual_text_sets_has_code():
    native = "Some explanation"
    visual = "CREATE TABLE etudiant (id NUMBER);"
    combined = merge_texts(native, visual)
    assert text_has_code(combined, set(Config.default().sql_keywords)) is True


def test_duplicate_visual_page_skipped():
    pages = [_page(1, PageClass.NO_TEXT, ""), _page(2, PageClass.NO_TEXT, "")]
    det = FakeDetector(process=True)
    calls = []
    fn = lambda img, prompt: (calls.append(1) or "x")
    _run_stage(pages, det, GeminiVisionModel("m", generate_fn=fn), FakeRenderer(mode="same"))
    assert pages[0].visual_processing_status == "generated"
    assert pages[1].visual_processing_status == "duplicate"
    assert len(calls) == 1


def test_model_error_preserves_native_text():
    page = _page(1, PageClass.NO_TEXT, raw_text="native")
    det = FakeDetector(process=True)
    fn = lambda img, prompt: (_ for _ in ()).throw(RuntimeError("boom"))
    _run_stage([page], det, GeminiVisionModel("m", generate_fn=fn), FakeRenderer())
    assert page.visual_processing_status == "model_error"
    assert page.has_visual_content is False
    assert page.visual_text == ""
    assert merge_texts(page.raw_text, page.visual_text) == "native"


def test_render_error_preserves_native_text():
    page = _page(1, PageClass.NO_TEXT, raw_text="native")
    det = FakeDetector(process=True)
    fn = lambda img, prompt: "x"
    _run_stage([page], det, GeminiVisionModel("m", generate_fn=fn), FakeRenderer(mode="raise"))
    assert page.visual_processing_status == "render_failed"
    assert page.has_visual_content is False
    assert merge_texts(page.raw_text, page.visual_text) == "native"


def test_merge_order_and_empty_handling():
    assert merge_texts("A", "B") == "A\n\nB"
    assert merge_texts("A", "") == "A"
    assert merge_texts("", "B") == "B"
    assert merge_texts("", "") == ""


def test_vision_metadata_propagates_to_chunks():
    pages = [
        _page(1, PageClass.NATIVE_TEXT, raw_text="intro"),
        _page(2, PageClass.NO_TEXT, raw_text=""),
    ]
    det = FakeDetector(process=True)
    fn = lambda img, prompt: "visual chunk"
    _run_stage(pages, det, GeminiVisionModel("Qwen/fake", generate_fn=fn), FakeRenderer())
    for p in pages:
        p.final_text = merge_texts(p.raw_text, p.visual_text)
    group = LogicalGroup(chapter="Chapitre 1", title="Intro", page_numbers=[1, 2])
    doc = Document(source="S.pdf", pages=pages)
    chunks = chunk_document(doc, [group], Config.default())
    assert len(chunks) == 1
    c = chunks[0]
    assert c.has_visual_content is True
    assert c.metadata["has_visual_content"] is True
    assert c.metadata["vision_model"] == "Qwen/fake"
    assert c.metadata["visual_processing_status"] == "generated"
    assert c.chapter == "Chapitre 1"
    assert c.slide_title == "Intro"


def test_vision_prompt_includes_chapter_and_slide_context():
    page = _page(5, PageClass.NO_TEXT, "")
    det = FakeDetector(process=True)
    _run_stage(
        [page],
        det,
        GeminiVisionModel("m", generate_fn=lambda img, prompt: "x"),
        FakeRenderer(),
        page_context={5: ("Chapitre 3", "Architecture")},
    )
    assert det.build_calls == [("S.pdf", "Chapitre 3", "Architecture")]
