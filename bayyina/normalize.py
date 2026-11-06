"""Arabic normalization and Unicode tokenization for Bayyina."""

from __future__ import annotations

import re

_DELETE = dict.fromkeys([*(range(0x064B, 0x0653)), 0x0670, 0x0640])
_FOLD = str.maketrans({
    "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ة": "ه",
    **{chr(0x0660 + i): str(i) for i in range(10)},
    **{chr(0x06F0 + i): str(i) for i in range(10)},
})
_WORDS = re.compile(r"\w+", flags=re.UNICODE)


def normalize_ar(text: str) -> str:
    """Normalize only the code points specified by the DSC task."""
    cleaned = text.translate(_DELETE).translate(_FOLD).lower()
    return " ".join(cleaned.split())


def tokenize(text: str) -> list[str]:
    """Tokenize normalized text, keeping mixed Latin/digit words intact."""
    return _WORDS.findall(normalize_ar(text))
