from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Set

from config import CleanConfig, Config


def _norm_line(line: str) -> str:
    s = unicodedata.normalize("NFKD", line)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


class TextCleaner:
    """Remove recurring header/footer boilerplate without touching real content.

    Two mechanisms, both generic (no document-specific ranges):
      1. Frequency: a normalized line present on >= ``boilerplate_fraction`` of pages
         is treated as boilerplate and removed everywhere.
      2. Regex: configurable footer/signature patterns (emails, professor signature, ...).
    """

    def __init__(self, config: Config):
        self.config: CleanConfig = config.clean
        self.boilerplate: Set[str] = set()
        self._email_re = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
        self._footer_res = [re.compile(p, re.IGNORECASE) for p in self.config.footer_patterns]
        self._num_re = re.compile(r"^[\d\s./-]+$")

    def fit(self, page_texts: List[str]) -> None:
        n = max(1, len(page_texts))
        counts: Dict[str, int] = {}
        for text in page_texts:
            seen = set()
            for line in text.split("\n"):
                nl = _norm_line(line)
                if nl and nl not in seen:
                    seen.add(nl)
                    counts[nl] = counts.get(nl, 0) + 1
        for line, c in counts.items():
            if c / n >= self.config.boilerplate_fraction and len(line) <= 120:
                self.boilerplate.add(line)

    def clean(self, text: str) -> str:
        if not text:
            return ""
        out_lines: List[str] = []
        for line in text.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue
            nl = _norm_line(stripped)
            if nl in self.boilerplate:
                continue
            if self.config.remove_emails and self._email_re.search(stripped):
                continue
            if any(r.search(stripped) for r in self._footer_res):
                continue
            # Drop standalone page numbers (e.g. "50", "2") that survive on their own line.
            if self._num_re.match(stripped) and len(stripped) <= 6:
                continue
            out_lines.append(stripped)
        cleaned = "\n".join(out_lines)
        cleaned = cleaned.replace("\xa0", " ")
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()
