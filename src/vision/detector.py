from models.document import PageClass

from vision.prompt import build_vision_prompt


class VisualContentDetector:
    """Decide whether a page carries meaningful visual content worth sending to the vision model.

    Goals:
    - Do NOT blindly send every page to the model.
    - Skip pages that are plain native text with no diagram.
    - Still process image-only (NO_TEXT) and body-image (HYBRID) pages.
    - Process native-text pages that embed a large image (a diagram), while ignoring
      tiny decorative icons.
    """

    def __init__(self, config):
        self.config = config

    def _max_embedded_image_area(self, pypdf_page):
        """Return the largest embedded image area (width*height) on the page, or 0."""
        if pypdf_page is None:
            return 0
        try:
            max_area = 0
            for image in pypdf_page.images:
                try:
                    pil = image.image
                    if pil is None:
                        continue
                    area = pil.width * pil.height
                except Exception:
                    area = 0
                if area > max_area:
                    max_area = area
            return max_area
        except Exception:
            return 0

    def has_meaningful_visual(self, page, pypdf_page):
        if page.page_class in (PageClass.NO_TEXT, PageClass.HYBRID):
            return True
        if self._max_embedded_image_area(pypdf_page) >= self.config.min_image_area:
            return True
        return False

    def build_prompt(self, source, chapter, slide_title):
        return build_vision_prompt(source, chapter, slide_title)
