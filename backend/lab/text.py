"""Text normalisation shared by uploads, folders and ground-truth file names."""

from __future__ import annotations

import unicodedata


def nfc(text: str) -> str:
    """The composed (NFC) form: macOS may send decomposed Hangul (NFD), which compares and sorts
    differently from the same name typed elsewhere."""
    return unicodedata.normalize("NFC", text)
