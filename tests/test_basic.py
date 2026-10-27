"""
Basic tests to verify core logic works.
Run with: pytest tests/
"""

from app.ingestion.chunker import chunk_text
from app.ingestion.loader import clean_text


def test_chunk_text_respects_size_and_overlap():
    text = "A" * 1000
    chunks = chunk_text(text, source="test.txt", chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text) <= 200


def test_chunk_text_no_chunks_for_empty_string():
    chunks = chunk_text("", source="test.txt", chunk_size=100, chunk_overlap=10)
    assert chunks == []


def test_chunk_overlap_must_be_smaller_than_chunk_size():
    try:
        chunk_text("hello world", source="t.txt", chunk_size=10, chunk_overlap=10)
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_clean_text_collapses_whitespace():
    dirty = "Hello    world\n\n\n\nThis   is   messy.\n\n"
    cleaned = clean_text(dirty)
    assert "    " not in cleaned
    assert "\n\n\n" not in cleaned


def test_clean_text_strips_null_bytes():
    dirty = "Hello\x00World"
    cleaned = clean_text(dirty)
    assert "\x00" not in cleaned
