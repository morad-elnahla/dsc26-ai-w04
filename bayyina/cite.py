"""Exact-source citation verification and presentation."""

from __future__ import annotations

from bayyina.types import Hit


def verify_citation(quote: str, document: str, start: int, end: int) -> bool:
    if not (0 <= start < end <= len(document)):
        return False
    normal_quote = " ".join(quote.split())
    if not normal_quote:
        return False
    return normal_quote in " ".join(document[start:end].split())


def format_citation(hit: Hit, max_quote: int = 160) -> str:
    quote = " ".join(hit.text.split())
    if len(quote) > max_quote:
        prefix = quote[:max_quote]
        last_space = prefix.rfind(" ")
        if last_space >= 0:
            prefix = prefix[:last_space]
        quote = prefix + "…"
    return f'[{hit.doc_id}:{hit.start}-{hit.end}] "{quote}"'
