"""
Document loading
================
Extracts text from PDF, DOCX and TXT while keeping page numbers (PDF) so answers can cite
"document name + page". Every failure is turned into a `DocumentLoadError` with a short,
user-safe message (no file system paths).

    file -> load_pages() -> [Page(text, page_number)] -> clean_text() -> chunker
"""

from __future__ import annotations

import os
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


class DocumentLoadError(ValueError):
    """Raised for unsupported, empty or unreadable documents."""


@dataclass
class Page:
    text: str
    page: Optional[int]  # 1-based for PDFs; None for formats without pages


def _check_signature(path: str, ext: str) -> None:
    """Cheap content sniffing so a renamed .exe is not parsed as a PDF/DOCX."""
    with open(path, "rb") as f:
        head = f.read(8)
    if ext == ".pdf" and not head.startswith(b"%PDF"):
        raise DocumentLoadError("File does not look like a valid PDF.")
    if ext == ".docx" and not zipfile.is_zipfile(path):
        raise DocumentLoadError("File does not look like a valid DOCX.")


def load_pdf_pages(file_path: str) -> List[Page]:
    from pypdf import PdfReader

    try:
        reader = PdfReader(file_path)
        if reader.is_encrypted:
            raise DocumentLoadError("Encrypted PDFs are not supported.")
        return [Page(text=p.extract_text() or "", page=i) for i, p in enumerate(reader.pages, start=1)]
    except DocumentLoadError:
        raise
    except Exception as e:
        raise DocumentLoadError(f"Could not read PDF ({type(e).__name__}).") from e


def load_docx_pages(file_path: str) -> List[Page]:
    from docx import Document as DocxDocument

    try:
        doc = DocxDocument(file_path)
    except Exception as e:
        raise DocumentLoadError(f"Could not read DOCX ({type(e).__name__}).") from e
    parts = [p.text for p in doc.paragraphs]
    for table in doc.tables:  # tables are common in specs/policies and were ignored before
        for row in table.rows:
            parts.append(" | ".join(c.text.strip() for c in row.cells))
    return [Page(text="\n".join(parts), page=None)]


def load_txt_pages(file_path: str) -> List[Page]:
    with open(file_path, "rb") as f:
        raw = f.read()
    for enc in ("utf-8-sig", "utf-16", "cp1252"):
        try:
            return [Page(text=raw.decode(enc), page=None)]
        except UnicodeDecodeError:
            continue
    return [Page(text=raw.decode("utf-8", errors="ignore"), page=None)]


def load_pages(file_path: str) -> List[Page]:
    ext = Path(file_path).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise DocumentLoadError(f"Unsupported file type '{ext}'. Supported: .pdf, .docx, .txt")
    if os.path.getsize(file_path) == 0:
        raise DocumentLoadError("File is empty.")
    _check_signature(file_path, ext)
    loader = {".pdf": load_pdf_pages, ".docx": load_docx_pages, ".txt": load_txt_pages}[ext]
    pages = [Page(text=clean_text(p.text), page=p.page) for p in loader(file_path)]
    pages = [p for p in pages if p.text]
    if not pages:
        raise DocumentLoadError("No extractable text (scanned PDFs need OCR, which is not included).")
    return pages


# ---------------------------------------------------------------- backward-compatible helpers
def load_document(file_path: str) -> str:
    """Original API: whole-document text."""
    return "\n\n".join(p.text for p in load_pages(file_path))


def clean_text(text: str) -> str:
    """Remove NULs/control chars and collapse whitespace while keeping paragraph breaks."""
    text = text.replace("\x00", "")
    text = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f]", " ", text)
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def load_all_documents(directory: str) -> List[dict]:
    """Original API kept for scripts: [{"source", "text"}] for every supported file."""
    results = []
    for fname in sorted(os.listdir(directory)):
        if Path(fname).suffix.lower() in SUPPORTED_EXTENSIONS:
            try:
                results.append({"source": fname, "text": load_document(os.path.join(directory, fname))})
            except DocumentLoadError:
                continue
    return results
