class RendererUnavailable(Exception):
    pass


class PageRenderer:
    """Render a PDF page to a PIL image using PyMuPDF (pymupdf).

    The heavy dependency (pymupdf) is imported lazily so that simply importing this
    module never requires PyMuPDF to be installed. When rendering is not possible,
    methods raise RendererUnavailable which the caller is expected to handle so that
    the native-text path is preserved.
    """

    def __init__(self, dpi=150):
        self.dpi = dpi
        self._pymupdf = None

    def _ensure_pymupdf(self):
        if self._pymupdf is None:
            try:
                import pymupdf  # PyMuPDF (the `fitz` alias is deprecated)
            except Exception as exc:  # pragma: no cover - depends on environment
                raise RendererUnavailable(
                    "PyMuPDF is not installed; visual rendering is unavailable."
                ) from exc
            self._pymupdf = pymupdf
        return self._pymupdf

    def render(self, pdf_path, page_index):
        """Render a 0-based page index of pdf_path and return a PIL.Image."""
        pymupdf = self._ensure_pymupdf()
        doc = pymupdf.open(pdf_path)
        try:
            page = doc.load_page(page_index)
            pix = page.get_pixmap(dpi=self.dpi)
            from io import BytesIO

            from PIL import Image

            return Image.open(BytesIO(pix.tobytes("png")))
        finally:
            doc.close()
