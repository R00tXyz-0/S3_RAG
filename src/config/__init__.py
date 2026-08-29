from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from vision.prompt import VISION_PROMPT


@dataclass
class ChunkConfig:
    target_tokens: int = 550
    overlap_tokens: int = 64
    max_tokens: int = 800
    min_tokens: int = 150


@dataclass
class ClassifyConfig:
    # A page whose cleaned text is shorter than this has no usable text layer.
    no_text_max_chars: int = 20
    # A page shorter than this (but with some text) is treated as title-only / hybrid.
    hybrid_max_chars: int = 120
    # A first line that repeats as the first line on >= this many pages is a deck label,
    # not the real slide title (used to find the true slide title on slide decks).
    first_line_label_min_freq: int = 4


@dataclass
class CleanConfig:
    # A normalized line present on at least this fraction of pages is boilerplate (header/footer).
    boilerplate_fraction: float = 0.25
    remove_emails: bool = True
    # Regexes (case-insensitive) matching recurring footer/signature lines to drop.
    footer_patterns: List[str] = field(
        default_factory=lambda: [
            r"A\.?Amine",
            r"ABDELLAH",
            r"Pr\.?\s*Dr\.?\s*Ing",
            r"Ann[eé]e universitaire",
            r"usms\.ma",
        ]
    )


@dataclass
class StructureConfig:
    # Lines matching these (case-insensitive) start a new chapter.
    chapter_patterns: List[str] = field(
        default_factory=lambda: [
            r"^\s*chapitre\s+\d+",
            r"^\s*chapter\s+\d+",
            r"^\s*partie\s+\d+",
            r"^\s*part\s+\d+",
        ]
    )
    # Normalized-title similarity at/above this merges consecutive pages into one group.
    title_similarity_threshold: float = 0.9


@dataclass
class VisionConfig:
    enabled: bool = True
    model_name: str = "gemini-3.6-flash"
    render_dpi: int = 150
    # Embedded image area (width*height in px) at/above which a native-text page is
    # considered to contain a meaningful diagram worth sending to the vision model.
    min_image_area: int = 20000
    # Skip identical rendered pages (same image hash) within a document run.
    dedup: bool = True
    max_new_tokens: int = 1024
    # Gemini API production behavior.
    timeout: int = 120
    max_retries: int = 4
    temperature: float = 0.3
    # Directory where successful per-page visual results are cached as JSON.
    cache_dir: str = "data/processed/vision"
    prompt_template: str = VISION_PROMPT


@dataclass
class Config:
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    classify: ClassifyConfig = field(default_factory=ClassifyConfig)
    clean: CleanConfig = field(default_factory=CleanConfig)
    structure: StructureConfig = field(default_factory=StructureConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    sql_keywords: List[str] = field(
        default_factory=lambda: [
            "CREATE",
            "ALTER",
            "DROP",
            "GRANT",
            "REVOKE",
            "SELECT",
            "INSERT",
            "UPDATE",
            "DELETE",
            "MERGE",
            "TRUNCATE",
            "COMMENT",
            "BEGIN",
            "DECLARE",
            "CURSOR",
            "LOOP",
            "COMMIT",
            "ROLLBACK",
        ]
    )
    # 0 = process all pages (used for quick tests).
    max_pages: int = 0
    processed_dir: str = "data/processed"

    @staticmethod
    def default() -> "Config":
        return Config()


def load_config(path: Optional[str] = None) -> Config:
    """Load configuration. YAML support can be added later; defaults are used for now."""
    return Config.default()
