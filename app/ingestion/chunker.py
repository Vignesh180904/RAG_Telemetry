"""
Chunking
========
Splits text into overlapping chunks of at most `chunk_size` characters.

Changes from the original sliding window:
* Boundary-aware: the cut is moved back to the nearest paragraph / sentence / word break inside
  the last 30% of the window, so chunks rarely end mid-word or mid-sentence.
* Deterministic IDs: `sha1(doc_id, page, index)` instead of random UUIDs, so re-ingesting the
  same document overwrites its chunks instead of duplicating them.
* Page numbers are carried through for citations.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, List, Optional

_BREAKS = ("\n\n", ". ", "? ", "! ", "\n", "; ", ", ", " ")


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    chunk_index: int
    start_char: int
    end_char: int
    page: Optional[int] = None
    doc_id: str = ""


def chunk_id(doc_id: str, page: Optional[int], index: int) -> str:
    return hashlib.sha1(f"{doc_id}|{page}|{index}".encode()).hexdigest()[:24]


def _cut_point(text: str, start: int, end: int, chunk_size: int) -> int:
    if end >= len(text):
        return end
    floor = start + int(chunk_size * 0.7)
    window = text[floor:end]
    for br in _BREAKS:
        pos = window.rfind(br)
        if pos != -1:
            return floor + pos + len(br)
    return end


def chunk_text(text: str, source: str, chunk_size: int = 500, chunk_overlap: int = 50,
               page: Optional[int] = None, doc_id: str = "", start_index: int = 0) -> List[Chunk]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    doc_id = doc_id or source
    chunks: List[Chunk] = []
    start, index, n = 0, start_index, len(text)
    while start < n:
        end = _cut_point(text, start, min(start + chunk_size, n), chunk_size)
        piece = text[start:end].strip()
        if piece:
            chunks.append(Chunk(id=chunk_id(doc_id, page, index), text=piece, source=source,
                                chunk_index=index, start_char=start, end_char=end, page=page, doc_id=doc_id))
            index += 1
        if end >= n:
            break
        nxt = max(end - chunk_overlap, start + 1)  # always make progress
        # start the overlap on a word boundary so the next chunk doesn't begin mid-word
        sp = text.find(" ", nxt, end)
        start = sp + 1 if sp != -1 and sp + 1 < end else nxt
    return chunks


def chunk_pages(pages: Iterable, source: str, doc_id: str, chunk_size: int, chunk_overlap: int) -> List[Chunk]:
    """Chunk a list of loader.Page objects; chunk_index keeps counting across pages."""
    out: List[Chunk] = []
    for p in pages:
        out.extend(chunk_text(p.text, source, chunk_size, chunk_overlap, page=p.page,
                              doc_id=doc_id, start_index=len(out)))
    return out


def chunk_documents(documents: List[dict], chunk_size: int = 500, chunk_overlap: int = 50) -> List[Chunk]:
    """Original API: chunk [{"source", "text"}]."""
    out: List[Chunk] = []
    for doc in documents:
        out.extend(chunk_text(doc["text"], doc["source"], chunk_size, chunk_overlap))
    return out
