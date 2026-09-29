"""PDF parser — PyMuPDF (fitz).

PyMuPDF is fastest of the Python options and gets text layout right for most
born-digital PDFs. Scanned PDFs have no text layer, so when OCR is enabled
(see ocr.py) their pages are rendered to images and OCR'd instead.
"""
from __future__ import annotations

from typing import Any

import pymupdf
import structlog

from .ocr import OcrUnavailable, ensure_available, ocr_image, resolve_ocr_config

log = structlog.get_logger()


def parse_pdf(content: bytes) -> str:
    """Extract concatenated page text from PDF bytes.

    Pages are separated by a double newline so the chunker can treat them as
    natural paragraph boundaries.

    Born-digital pages use the embedded text layer. Pages with no text fall
    back to OCR when OCR is enabled; when it's off they're still skipped, so
    behaviour is unchanged for the common case.
    """
    cfg = resolve_ocr_config()

    doc = pymupdf.open(stream=content, filetype="pdf")
    try:
        pages: list[str] = []
        ocr_attempts = 0
        ocr_hits = 0
        max_pages = cfg["max_pages"] if cfg is not None else 0
        for page in doc:
            text = (page.get_text("text") or "").strip()  # type: ignore[attr-defined]
            if not text and cfg is not None:
                if max_pages and ocr_attempts >= max_pages:
                    continue
                ocr_attempts += 1
                text = _ocr_page(page, cfg)
                if text:
                    ocr_hits += 1
            if text:
                pages.append(text)
        # A non-empty render pass that recognized nothing is almost always a
        # misconfiguration (bad key, wrong model) — surface it instead of
        # silently producing an "empty content after parsing" ingest failure.
        if ocr_attempts and not ocr_hits:
            raise RuntimeError(
                f"OCR 对 {ocr_attempts} 个无文本页面均未识别出文字，"
                f"请检查 OCR_PROVIDER={cfg['provider']} 的配置或凭证。"
            )
        return "\n\n".join(pages)
    finally:
        doc.close()


def _ocr_page(page: Any, cfg: dict[str, Any]) -> str:
    """Render one page to PNG and OCR it. Returns '' on a per-page failure."""
    ensure_available(cfg)
    try:
        pix = page.get_pixmap(dpi=cfg["dpi"])
        png = pix.tobytes("png")
    except Exception:  # noqa: BLE001
        log.exception("ocr_render_failed", page=getattr(page, "number", None))
        return ""
    try:
        return ocr_image(png, cfg).strip()
    except OcrUnavailable:
        raise
    except Exception:  # noqa: BLE001
        log.exception("ocr_failed", page=getattr(page, "number", None))
        return ""