import time
from dataclasses import dataclass

import httpx

from app.core.config import settings


RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


@dataclass(frozen=True)
class GeminiResponse:
    payload: dict
    model: str
    attempts: int


def gemini_model_candidates(primary: str | None = None) -> list[str]:
    values = [primary or settings.gemini_model]
    values.extend(
        item.strip()
        for item in settings.gemini_fallback_models.split(",")
        if item.strip()
    )
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def generate_content(
    *,
    parts: list[dict],
    response_mime_type: str = "application/json",
    max_output_tokens: int = 4096,
    timeout_seconds: float | None = None,
    thinking_level: str | None = None,
    primary_model: str | None = None,
) -> GeminiResponse:
    """Call Gemini with bounded retry + stable-model failover.

    Paid/final rendering must not silently degrade because of a transient 429/5xx.
    We retry each configured model with exponential backoff and then move to the
    next stable fallback model.
    """
    if not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured.")

    attempts_per_model = max(1, int(settings.gemini_max_attempts_per_model))
    base_delay = max(0.0, float(settings.gemini_retry_backoff_seconds))
    timeout = httpx.Timeout(timeout_seconds or settings.gemini_timeout_seconds)
    level = (thinking_level or settings.gemini_thinking_level).strip().lower()
    if level not in {"low", "medium", "high"}:
        level = "low"

    last_error = "Gemini request failed."
    total_attempts = 0

    for model in gemini_model_candidates(primary_model):
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent"
        )
        for attempt in range(attempts_per_model):
            total_attempts += 1
            try:
                response = httpx.post(
                    url,
                    params={"key": settings.gemini_api_key},
                    json={
                        "contents": [{"parts": parts}],
                        "generationConfig": {
                            "responseMimeType": response_mime_type,
                            "maxOutputTokens": max_output_tokens,
                            "thinkingConfig": {"thinkingLevel": level},
                        },
                    },
                    timeout=timeout,
                )
            except httpx.TimeoutException:
                last_error = f"Gemini {model} request timed out."
                retryable = True
            except httpx.HTTPError as exc:
                # HTTP exception strings may contain the request URL/API key.
                last_error = f"Gemini {model} connection failed ({type(exc).__name__})."
                retryable = True
            else:
                if response.status_code == 200:
                    return GeminiResponse(
                        payload=response.json(),
                        model=model,
                        attempts=total_attempts,
                    )

                safe_detail = ""
                try:
                    error_payload = response.json().get("error") or {}
                    safe_detail = str(error_payload.get("status") or error_payload.get("message") or "")
                except Exception:
                    safe_detail = ""
                suffix = f" ({safe_detail[:180]})" if safe_detail else ""
                last_error = f"Gemini {model} returned HTTP {response.status_code}{suffix}"
                retryable = response.status_code in RETRYABLE_STATUS_CODES

            if not retryable:
                break
            if attempt < attempts_per_model - 1 and base_delay > 0:
                time.sleep(base_delay * (2**attempt))

    raise RuntimeError(last_error)
