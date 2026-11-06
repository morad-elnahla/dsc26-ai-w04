"""Structure-aware chunking with offsets into the unmodified source document."""

from __future__ import annotations

from bayyina.types import Chunk

_SENTENCE_ENDS = frozenset(".!?؟۔\n")


def _cut_offset(text: str, start: int, max_chars: int) -> int:
    """Return the exclusive cut offset, before trimming trailing whitespace."""
    window = text[start : start + max_chars]
    first = max_chars // 2
    # All priority rules use the *last* suitable marker in their own group.
    paragraphs = [c for c in range(max(2, first), max_chars + 1)
                  if window[c - 2 : c] == "\n\n"]
    if paragraphs:
        return start + paragraphs[-1]
    sentences = [c for c in range(max(1, first), max_chars + 1)
                 if window[c - 1] in _SENTENCE_ENDS]
    if sentences:
        return start + sentences[-1]
    spaces = [c for c in range(max(1, first), max_chars + 1)
              if window[c - 1].isspace()]
    if spaces:
        return start + spaces[-1]
    return start + max_chars


def _next_start(text: str, previous_start: int, previous_end: int,
                overlap: int) -> int:
    """Find an allowed next start within the overlap, or the next word."""
    lower = max(previous_start + 1, previous_end - overlap)
    mid_word_cut = (previous_end < len(text)
                    and not text[previous_end - 1].isspace()
                    and not text[previous_end].isspace())
    if mid_word_cut:
        return lower

    # If the cut was at a word boundary, choose a word boundary for the next
    # chunk as well. This also prevents duplicating a mid-word fragment.
    for pos in range(lower, previous_end + 1):
        if (pos < len(text) and not text[pos].isspace()
                and (pos == 0 or text[pos - 1].isspace())):
            return pos

    pos = previous_end
    while pos < len(text) and text[pos].isspace():
        pos += 1
    return pos


def chunk_text(doc_id: str, text: str, max_chars: int = 500,
               overlap: int = 50) -> list[Chunk]:
    """Split into source-identical, contiguous/overlapping text spans."""
    if max_chars <= 0 or overlap < 0 or overlap >= max_chars:
        raise ValueError("Require max_chars > 0 and 0 <= overlap < max_chars")
    if not text.strip():
        return []

    chunks: list[Chunk] = []
    start = 0
    while start < len(text):
        while start < len(text) and text[start].isspace():
            start += 1
        if start >= len(text):
            break

        if len(text) - start <= max_chars:
            raw_end = len(text)
        else:
            raw_end = _cut_offset(text, start, max_chars)
        end = raw_end
        while end > start and text[end - 1].isspace():
            end -= 1

        # Whitespace can fill the entire selected cut window for tiny sizes.
        if end <= start:
            end = start + 1

        chunks.append(Chunk(
            doc_id=doc_id,
            chunk_id=f"{doc_id}#{len(chunks)}",
            start=start,
            end=end,
            text=text[start:end],
        ))
        if not text[raw_end:].strip():
            break
        start = _next_start(text, start, end, overlap)

    return chunks
