from __future__ import annotations

import hashlib
import io
import json
import logging
import os
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("vision.stage")


def hash_image(image):
    """Stable hash of a rendered page image for deduplication."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return hashlib.md5(buf.getvalue()).hexdigest()


class VisionStage:
    """Per-page orchestration: detect meaningful visual content, render, dedupe, infer, cache.

    On any failure (render or inference) the page keeps its native text and the failure is
    recorded in visual_processing_status so the caller can continue safely. Successful visual
    results are cached to disk so a page is never re-processed if its result is already cached.
    """

    def __init__(
        self,
        model,
        renderer,
        detector,
        config,
        cache_dir: Optional[str] = None,
    ):
        self.model = model
        self.renderer = renderer
        self.detector = detector
        self.config = config
        self.cache_dir = cache_dir or getattr(config, "cache_dir", None) or "data/processed/vision"

    def _cache_path(self, source: str, page_number: int):
        from pathlib import Path

        base = Path(self.cache_dir) / os.path.splitext(source)[0]
        base.mkdir(parents=True, exist_ok=True)
        return base / f"page_{page_number:03d}.json"

    def _load_cache(self, path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("status") == "success" and str(data.get("visual_text", "")).strip():
                return data
        except Exception:
            return None
        return None

    def _save_cache(self, path, source, page_number, model_name, visual_text):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "source": source,
                    "page": page_number,
                    "model": model_name,
                    "visual_text": visual_text,
                    "status": "success",
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    def process(
        self,
        pdf_path: str,
        pages: List,
        pypdf_pages: Dict[int, object],
        page_context: Dict[int, Tuple[str, str]],
    ) -> None:
        seen_hashes = set()
        for page in pages:
            source = page.source
            chapter, slide_title = page_context.get(page.page_number, ("", ""))

            # 1) Use cache if a successful result already exists (never re-process).
            cache_path = self._cache_path(source, page.page_number)
            cached = self._load_cache(cache_path)
            if cached is not None:
                page.visual_text = cached["visual_text"]
                page.has_visual_content = True
                page.visual_processing_status = "cached"
                page.vision_model = cached.get("model", self.model.name)
                logger.info("page %d: cache hit (skip Gemini)", page.page_number)
                continue

            # 2) Skip pages without meaningful visual content (don't spam the API).
            if not self.detector.has_meaningful_visual(page, pypdf_pages.get(page.page_number)):
                page.visual_processing_status = "skipped_no_visual"
                page.has_visual_content = False
                continue

            # 3) Render the page to an image.
            try:
                image = self.renderer.render(pdf_path, page.page_number - 1)
            except Exception:
                page.visual_processing_status = "render_failed"
                page.has_visual_content = False
                logger.error("page %d: render_failed", page.page_number)
                continue

            # 4) Deduplicate identical rendered pages within the run.
            if self.config.dedup:
                image_hash = hash_image(image)
                if image_hash in seen_hashes:
                    page.visual_processing_status = "duplicate"
                    page.has_visual_content = False
                    continue
                seen_hashes.add(image_hash)

            # 5) Infer visual text via Gemini.
            try:
                prompt = self.detector.build_prompt(source, chapter, slide_title)
                generated = self.model.generate(image, prompt).strip()
            except Exception:
                page.visual_processing_status = "model_error"
                page.has_visual_content = False
                logger.error("page %d: model_error", page.page_number)
                continue

            if generated:
                page.visual_text = generated
                page.has_visual_content = True
                page.visual_processing_status = "generated"
                page.vision_model = self.model.name
                self._save_cache(cache_path, source, page.page_number, self.model.name, generated)
                logger.info(
                    "page %d: success (native=%d visual=%d)",
                    page.page_number,
                    len(page.raw_text),
                    len(generated),
                )
            else:
                page.visual_processing_status = "empty"
                page.has_visual_content = False
                logger.warning("page %d: empty response", page.page_number)
