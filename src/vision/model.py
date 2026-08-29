from __future__ import annotations

import io
import logging
import os
import re
import time
from typing import Optional

logger = logging.getLogger("vision.model")

ENV_API_KEY = "GEMINI_API_KEY"


class VisionModelError(Exception):
    """Raised when the Gemini vision model cannot produce a result after retries."""


class GeminiVisionModel:
    """Multimodal vision wrapper backed by the Google Gemini API (google-genai SDK).

    The API key is read from the GEMINI_API_KEY environment variable (optionally loaded
    from a .env file via python-dotenv). The key is NEVER stored on the instance in a way
    that gets logged, and is NEVER printed. A generate_fn can be injected for testing so
    that no network call is made.
    """

    def __init__(
        self,
        model_name: str = "gemini-3.6-flash",
        api_key: Optional[str] = None,
        generate_fn=None,
        timeout: int = 120,
        max_retries: int = 4,
        temperature: float = 0.3,
    ):
        self.name = model_name
        self._api_key = api_key
        self._generate_fn = generate_fn
        self.timeout = timeout
        self.max_retries = max_retries
        self.temperature = temperature
        self._client = None

    def _get_api_key(self) -> Optional[str]:
        return self._api_key or os.getenv(ENV_API_KEY)

    def _ensure_client(self):
        if self._client is None:
            from google import genai
            from google.genai import types

            key = self._get_api_key()
            if not key:
                raise VisionModelError(
                    f"{ENV_API_KEY} is not set; cannot call the Gemini API."
                )
            # HttpOptions.timeout is expressed in MILLISECONDS, while VisionConfig.timeout is
            # in seconds. Convert and enforce the API's minimum 10s deadline so the request is
            # never rejected with "Manually set deadline 1s is too short".
            timeout_ms = max(int(self.timeout * 1000), 10_000)
            self._client = genai.Client(
                api_key=key,
                http_options=types.HttpOptions(
                    timeout=timeout_ms,
                    # Disable the SDK's built-in retries (default retries 408/429/5xx) so that
                    # OUR explicit policy is the only one applied: no retry on 4xx/429, retry on
                    # 5xx and network/timeout errors.
                    retry_options=types.HttpRetryOptions(attempts=1),
                ),
            )
        return self._client

    def _api_status(self, exc) -> Optional[int]:
        # google-genai's ClientError often omits status_code; recover it from the body/message
        # so our 4xx/5xx retry policy is enforced correctly in production.
        status = getattr(exc, "status_code", None)
        if status is not None:
            return status
        msg = str(exc)
        m = re.search(r"'code'\s*:\s*(\d{3})", msg)
        if m:
            return int(m.group(1))
        m = re.match(r"^\s*(\d{3})\s+[A-Z_]+", msg)
        if m:
            return int(m.group(1))
        return None

    def generate(self, image, prompt: str) -> str:
        if self._generate_fn is not None:
            return self._generate_fn(image, prompt)
        return self._generate_with_retry(image, prompt)

    # --- diagnostics (never include secrets) ---

    def _image_info(self, image) -> dict:
        info: dict = {}
        try:
            info["format"] = getattr(image, "format", None) or "unknown"
            info["mode"] = getattr(image, "mode", None)
            info["width"], info["height"] = image.size
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            info["bytes"] = len(buf.getvalue())
        except Exception:
            pass
        return info

    def _redact(self, text: str) -> str:
        # Belt-and-suspenders: never leak the API key if it ever ends up in an error string.
        key = self._api_key
        if key and len(key) > 4 and key in text:
            text = text.replace(key, "***REDACTED***")
        return text

    def _phase_for(self, exc) -> str:
        name = type(exc).__name__.lower()
        from google.genai import errors

        if isinstance(exc, errors.APIError):
            return "Gemini responded with an API error (after sending the image)"
        if "readtimeout" in name:
            return "waiting for the Gemini response (ReadTimeout)"
        if "writetimeout" in name:
            return "uploading/sending the image (WriteTimeout)"
        if "connecttimeout" in name or "connectionerror" in name or name.startswith("connect"):
            return "connecting/sending the image (connection error)"
        return "unknown phase (during the API call)"

    def _log_diagnostic(self, exc, status, img_info, phase, retryable):
        msg = self._redact(str(exc))
        level = logger.error if not retryable else logger.warning
        level(
            "Gemini vision failure | model=%s | exception=%s | http_status=%s | retryable=%s | "
            "phase=%s | image=%s | message=%s",
            self.name,
            type(exc).__name__,
            status,
            retryable,
            phase,
            img_info,
            msg[:800],
        )

    def _generate_with_retry(self, image, prompt: str) -> str:
        from google.genai import errors

        img_info = self._image_info(image)
        last_exc: Optional[BaseException] = None
        for attempt in range(self.max_retries):
            try:
                client = self._ensure_client()
                from google.genai import types

                config = types.GenerateContentConfig(temperature=self.temperature)
                response = client.models.generate_content(
                    model=self.name,
                    contents=[prompt, image],
                    config=config,
                )
                return response.text or ""
            except VisionModelError:
                # Configuration / missing-key error: never retryable.
                raise
            except errors.APIError as exc:
                status = self._api_status(exc)
                phase = self._phase_for(exc)
                if status is not None and 400 <= status < 500:
                    # Client errors (e.g. 429 RESOURCE_EXHAUSTED / daily free-tier quota,
                    # 401 auth, 403, 404 model-not-found) are NOT retryable: retrying will
                    # not help and only wastes quota/time. Report and propagate immediately.
                    self._log_diagnostic(exc, status, img_info, phase, retryable=False)
                    raise
                # 5xx server errors or unknown status: retry with backoff.
                last_exc = exc
                self._log_diagnostic(exc, status, img_info, phase, retryable=True)
                time.sleep((2 ** attempt) * 5)
            except Exception as exc:  # pragma: no cover - network/unknown
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                phase = self._phase_for(exc)
                last_exc = exc
                self._log_diagnostic(exc, None, img_info, phase, retryable=True)
                time.sleep((2 ** attempt) * 2)
        detail = self._redact(str(last_exc))
        logger.error(
            "Gemini vision ultimately failed | model=%s | exception=%s | http_status=%s | image=%s | message=%s",
            self.name,
            type(last_exc).__name__,
            getattr(last_exc, "status_code", None),
            img_info,
            detail[:800],
        )
        raise VisionModelError(
            f"Gemini generation failed after {self.max_retries} attempts: "
            f"{type(last_exc).__name__}: {detail[:300]}"
        )
