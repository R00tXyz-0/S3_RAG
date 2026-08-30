from __future__ import annotations

import json
import os
from typing import Dict, List

from models.document import Chunk

from config import Config


def print_report(report: Dict, chunks: List[Chunk], config: Config) -> None:
    print("=" * 70)
    print(f"INGESTION REPORT — {report['source']}")
    print("=" * 70)
    rows = [
        ("total pages", report["total_pages"]),
        ("native-text pages", report["native_text_pages"]),
        ("hybrid pages", report["hybrid_pages"]),
        ("no-text pages", report["no_text_pages"]),
        ("chapters detected", report["chapters_detected"]),
        ("logical groups", report["logical_groups"]),
        ("chunks", report["chunks"]),
        ("min chunk tokens", report["min_chunk_tokens"]),
        ("max chunk tokens", report["max_chunk_tokens"]),
        ("avg chunk tokens", report["avg_chunk_tokens"]),
        ("chunks below min tokens", report["chunks_below_min_tokens"]),
        ("chunks with code", report["chunks_with_code"]),
        ("code blocks preserved", report["code_blocks_preserved"]),
        ("sections detected", report["sections_detected"]),
        ("suspicious chunks", len(report["suspicious_chunks"])),
        ("validation ok", report["validation_ok"]),
        ("validation warnings", report["validation_warnings"]),
        ("validation errors", report["validation_errors"]),
    ]
    for k, v in rows:
        print(f"  {k:<26}: {v}")
    print("-" * 70)
    print("  chapters:")
    for ch in report["chapters"]:
        print(f"    - {ch}")
    print("=" * 70)


def save_report(report: Dict, chunks: List[Chunk], config: Config, stem: str) -> Dict[str, str]:
    out_dir = os.path.join(config.processed_dir, "evaluation")
    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, f"report_{stem}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    chunks_dir = os.path.join(config.processed_dir, "chunks")
    os.makedirs(chunks_dir, exist_ok=True)
    chunks_path = os.path.join(chunks_dir, f"chunks_{stem}.jsonl")
    with open(chunks_path, "w", encoding="utf-8") as f:
        for c in chunks:
            rec = {
                "chunk_index": c.chunk_index,
                "text": c.text,
                "metadata": c.metadata,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return {"report": report_path, "chunks": chunks_path}
