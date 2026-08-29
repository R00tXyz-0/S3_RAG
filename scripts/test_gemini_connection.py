"""Minimal Gemini API connectivity test.

Intent: verify that the Gemini API is reachable and that the configured API key
authenticates, using the simplest possible text-only request. This script does NOT
touch the RAG pipeline, PDF extraction, PyMuPDF, Pillow, or the VisionStage.

Run with: .venv/Scripts/python.exe scripts/test_gemini_connection.py
"""

from __future__ import annotations

import os
import time

from dotenv import load_dotenv


def main() -> int:
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
    key_detected = bool(api_key)
    print(f"API key detected: {'yes' if key_detected else 'no'}")
    print("model: gemini-3.6-flash")

    if not key_detected:
        print("ERROR: GEMINI_API_KEY is not set (environment or .env). Cannot authenticate.")
        return 2

    from google import genai
    from google.genai import types

    model_name = "gemini-3.6-flash"
    timeout_seconds = 30

    try:
        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=timeout_seconds),
        )
        print(f"request started (timeout={timeout_seconds}s)")

        start = time.time()
        response = client.models.generate_content(
            model=model_name,
            contents="Reply with exactly: GEMINI_OK",
        )
        elapsed = time.time() - start
        print("response received")
        print(f"elapsed time: {elapsed:.2f}s")

        text = getattr(response, "text", None)
        print(f"response text: {text!r}")

        if text and "GEMINI_OK" in text:
            print("RESULT: OK")
            return 0
        print("RESULT: unexpected response (but API reachable)")
        return 0
    except Exception as exc:  # surface the original error clearly, with no retries
        elapsed = time.time() - start if "start" in dir() else None
        print(f"EXCEPTION: {type(exc).__name__}: {exc}")
        if elapsed is not None:
            print(f"elapsed time: {elapsed:.2f}s")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
