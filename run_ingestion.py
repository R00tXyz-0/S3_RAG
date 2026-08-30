from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from config import load_config
from ingestion.pdf_loader import run_pipeline
from ingestion.report import print_report, save_report

ROOT = os.path.dirname(__file__)
RAW = os.path.join(ROOT, "data", "raw")

PDFS = [
    ("oracle", "Cours_Oracle_Complet.pdf"),
    ("plsql", "PLSQL Version Finale (1).pdf"),
]


def main() -> None:
    config = load_config()
    config.processed_dir = os.path.join(ROOT, "data", "processed")

    for sub, fname in PDFS:
        path = os.path.join(RAW, sub, fname)
        if not os.path.exists(path):
            print(f"[skip] {path} not found")
            continue
        print(f"\n>>> Processing {fname} ...")
        document, report = run_pipeline(path, config, config.processed_dir)
        stem = os.path.splitext(fname)[0]
        paths = save_report(report, document.chunks, config, stem)
        print_report(report, document.chunks, config)
        print(f"    saved: {paths['report']}")
        print(f"    saved: {paths['chunks']}")


if __name__ == "__main__":
    main()
