"""L3 — output filter. Redact PII patterns."""
from __future__ import annotations

import re

PHONE = re.compile(r"\b1[3-9]\d{9}\b")
ID_CARD = re.compile(r"\b\d{17}[\dXx]\b")


ID_CARD_LEN = 18
# Longest PII pattern is an 18-char ID card. Holding that many trailing chars
# back while streaming guarantees a pattern is never split across two emitted
# chunks. A trailing partial pattern prefix (e.g. "1", "138", "1234...X") is
# held back too so it can't be emitted before the match completes.
_HOLDBACK = ID_CARD_LEN
_PARTIAL_TAIL = re.compile(r"\b1[3-9]\d{0,9}$|\b\d{1,17}[Xx]?$")


def redact_pii(text: str) -> str:
    text = PHONE.sub("[手机号已隐藏]", text)
    text = ID_CARD.sub("[身份证已隐藏]", text)
    return text


class StreamRedactor:
    """Incrementally redact PII across a token stream.

    ``feed`` returns the redacted-safe prefix of everything received so far
    (empty until enough text has accumulated); ``flush`` returns the redacted
    remainder once the stream ends. Together they emit the same text as
    ``redact_pii(full_text)`` but split across calls, without ever splitting a
    phone / ID-card pattern across two emitted chunks.
    """

    def __init__(self) -> None:
        self._buf = ""

    def feed(self, delta: str) -> str:
        self._buf += delta
        cut = len(self._buf) - _HOLDBACK
        if cut <= 0:
            return ""
        m = _PARTIAL_TAIL.search(self._buf[:cut])
        if m:
            cut = m.start()
        if cut <= 0:
            return ""
        out, self._buf = self._buf[:cut], self._buf[cut:]
        return redact_pii(out)

    def flush(self) -> str:
        out, self._buf = redact_pii(self._buf), ""
        return out
