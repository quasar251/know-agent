"""OCR for image-only PDF pages.

The PDF parser (pdf.py) hands us pages with no extractable text layer — i.e.
scanned or photographed pages (e.g. the 90 MB 学生手册). We render such a page
to a PNG and OCR it. Two providers sit behind one interface, picked by
OCR_PROVIDER:

  rapidocr       local ONNX model — offline, no API key, no per-call cost.
                 Needs `pip install rapidocr-onnxruntime`.
  siliconflow    vision LLM over the OpenAI /chat/completions surface; the page
                 image is inlined as a data: URI. Reuses the SiliconFlow key.
                 `openai-compat` uses the same protocol with a explicit URL.

Config resolution mirrors embedding / reranker: the env-level config is
synthesized from OCR_* settings (see settings.py). It returns None when OCR is
disabled, so callers keep a single code path (no cfg → no OCR).
"""
from __future__ import annotations

import base64
import threading
from typing import Any

import httpx
import structlog

from src.settings import get_settings

log = structlog.get_logger()

# Presets supply base_url/model/protocol; explicit OCR_* values always win.
PROVIDER_PRESETS: dict[str, dict[str, str]] = {
    "rapidocr": {"protocol": "local"},
    "siliconflow": {
        "protocol": "vision",
        "base_url": "https://api.siliconflow.cn/v1",
        "model": "Qwen/Qwen2.5-VL-32B-Instruct",
    },
    "openai-compat": {"protocol": "vision", "base_url": "", "model": ""},
}

# Keep the vision prompt terse & literal: we want a faithful transcription, not
# a summary or commentary, so downstream chunking/embedding sees raw policy text.
_OCR_PROMPT = (
    "请逐字识别这张页面图片中的全部文字，按原有的阅读顺序和段落换行输出。"
    "只输出识别到的文字本身，不要添加任何解释、标题、页码或 Markdown 代码块标记。"
    "如果页面没有任何文字，输出空字符串。"
)


class OcrUnavailable(RuntimeError):
    """OCR is enabled but its provider cannot run (missing dep / config)."""


def resolve_ocr_config() -> dict[str, Any] | None:
    """Synthesize the OCR config from settings, or None when OCR is off."""
    s = get_settings()
    if not s.ocr_enabled:
        return None
    preset = PROVIDER_PRESETS.get(s.ocr_provider)
    if preset is None:
        raise OcrUnavailable(
            f"unknown OCR_PROVIDER '{s.ocr_provider}'; expected one of {sorted(PROVIDER_PRESETS)}"
        )
    return {
        "provider": s.ocr_provider,
        "protocol": preset["protocol"],
        "base_url": (s.ocr_base_url or preset.get("base_url", "")).rstrip("/"),
        "api_key": s.ocr_api_key,
        "model": s.ocr_model or preset.get("model", ""),
        "dpi": s.ocr_dpi,
        "max_pages": s.ocr_max_pages,
    }


def ensure_available(cfg: dict[str, Any]) -> None:
    """Raise OcrUnavailable if the configured provider can't run at all.

    Called lazily on the first page that actually needs OCR, so born-digital
    PDFs are never affected by an OCR misconfiguration.
    """
    if cfg["protocol"] == "local":
        try:
            import rapidocr_onnxruntime  # noqa: F401
        except ImportError as exc:  # pragma: no cover - depends on install
            raise OcrUnavailable(
                "OCR_ENABLED=true 且 OCR_PROVIDER=rapidocr，但未安装本地 OCR 依赖。"
                "请执行 `pip install rapidocr-onnxruntime`，"
                "或改用 OCR_PROVIDER=siliconflow（视觉模型 API）。"
            ) from exc
    elif not (cfg["base_url"] and cfg["model"] and cfg["api_key"]):
        raise OcrUnavailable(
            f"OCR provider '{cfg['provider']}' 缺少 base_url / model / api_key，"
            "请在 .env 中配置 OCR_BASE_URL / OCR_MODEL / OCR_API_KEY。"
        )


# ---------------------------------------------------------------------------
# Local provider (RapidOCR)
# ---------------------------------------------------------------------------
# Model load is expensive (hundreds of ms + tens of MB); keep one per process.
_local_lock = threading.Lock()
_local_engine: Any = None


def _get_local_engine() -> Any:
    global _local_engine
    if _local_engine is None:
        with _local_lock:
            if _local_engine is None:
                from rapidocr_onnxruntime import RapidOCR

                _local_engine = RapidOCR()
    return _local_engine


def _ocr_local(png_bytes: bytes, cfg: dict[str, Any]) -> str:
    engine = _get_local_engine()
    result, _elapsed = engine(png_bytes)
    if not result:
        return ""
    lines = [str(item[1]) for item in result if len(item) > 1 and item[1]]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Vision provider (SiliconFlow / any OpenAI-compatible chat endpoint)
# ---------------------------------------------------------------------------
def _ocr_vision(png_bytes: bytes, cfg: dict[str, Any]) -> str:
    data_uri = "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")
    payload: dict[str, Any] = {
        "model": cfg["model"],
        "temperature": 0,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _OCR_PROMPT},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }
        ],
    }
    headers = {"Content-Type": "application/json"}
    if cfg["api_key"]:
        headers["Authorization"] = f"Bearer {cfg['api_key']}"
    # Sync client: parsers run inside a worker thread / background task, and a
    # per-page timeout keeps one slow page from stalling the whole document.
    with httpx.Client(timeout=120.0) as client:
        resp = client.post(f"{cfg['base_url']}/chat/completions", headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
    choices = data.get("choices") if isinstance(data, dict) else None
    if not choices:
        return ""
    content = (choices[0].get("message") or {}).get("content")
    if isinstance(content, list):  # some gateways return content parts
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return (content or "").strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def ocr_image(png_bytes: bytes, cfg: dict[str, Any]) -> str:
    """OCR a single rendered page image. May raise OcrUnavailable."""
    if cfg["protocol"] == "local":
        return _ocr_local(png_bytes, cfg)
    return _ocr_vision(png_bytes, cfg)