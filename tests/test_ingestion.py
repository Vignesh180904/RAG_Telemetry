"""Ingestion: loaders, metadata, chunking, deterministic IDs, duplicate/re-index safety, error handling."""

import shutil

import pytest

from app.ingestion.chunker import chunk_pages, chunk_text
from app.ingestion.loader import DocumentLoadError, Page, load_pages
from app.ingestion.pipeline import IngestionService, document_id_for
from app.retrieval.vector_store import EmbeddingMismatchError, VectorStore


def _svc(tmp_path, emb, name="ingest_test"):
    store = VectorStore(str(tmp_path / "db"), name, emb.model_id)
    return IngestionService(emb, store, chunk_size=300, chunk_overlap=40), store


# ---------------------------------------------------------------- chunking
def test_chunk_ids_are_deterministic():
    a = chunk_text("Hello world. " * 100, "a.txt", 200, 20, doc_id="d1")
    b = chunk_text("Hello world. " * 100, "a.txt", 200, 20, doc_id="d1")
    assert [c.id for c in a] == [c.id for c in b]
    assert len({c.id for c in a}) == len(a)


def test_chunks_prefer_sentence_boundaries():
    text = " ".join(f"Sentence number {i} is here." for i in range(60))
    for c in chunk_text(text, "s.txt", 200, 30)[:-1]:
        assert c.text.endswith("."), c.text[-20:]


def test_chunk_pages_keep_page_numbers_and_continuous_index():
    pages = [Page("Page one text. " * 30, 1), Page("Page two text. " * 30, 2)]
    chunks = chunk_pages(pages, "x.pdf", "doc", 200, 20)
    assert {c.page for c in chunks} == {1, 2}
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_large_overlap_still_terminates():
    chunks = chunk_text("word " * 500, "w.txt", chunk_size=100, chunk_overlap=95)
    assert chunks and all(len(c.text) <= 100 for c in chunks)


# ---------------------------------------------------------------- loaders
def test_pdf_pages_are_extracted(corpus_dir):
    pages = load_pages(str(corpus_dir / "backup_guide.pdf"))
    assert [p.page for p in pages] == [1, 2]
    assert "every 6 hours" in pages[1].text


def test_docx_paragraphs_and_tables(corpus_dir):
    text = load_pages(str(corpus_dir / "travel_policy.docx"))[0].text
    assert "INR 1,500" in text and "Economy class" in text


def test_empty_file_rejected(tmp_path):
    p = tmp_path / "empty.txt"
    p.write_text("")
    with pytest.raises(DocumentLoadError, match="empty"):
        load_pages(str(p))


def test_whitespace_only_file_rejected(tmp_path):
    p = tmp_path / "blank.txt"
    p.write_text("   \n\n  \t ")
    with pytest.raises(DocumentLoadError, match="No extractable text"):
        load_pages(str(p))


def test_unsupported_extension_rejected(tmp_path):
    p = tmp_path / "x.exe"
    p.write_bytes(b"MZ...")
    with pytest.raises(DocumentLoadError, match="Unsupported"):
        load_pages(str(p))


@pytest.mark.parametrize("name,content", [("fake.pdf", b"not a pdf at all"), ("fake.docx", b"plain text")])
def test_corrupt_or_renamed_files_rejected(tmp_path, name, content):
    p = tmp_path / name
    p.write_bytes(content)
    with pytest.raises(DocumentLoadError):
        load_pages(str(p))


def test_truncated_pdf_rejected(tmp_path, corpus_dir):
    data = (corpus_dir / "backup_guide.pdf").read_bytes()
    p = tmp_path / "broken.pdf"
    p.write_bytes(data[:200])
    with pytest.raises(DocumentLoadError):
        load_pages(str(p))


# ---------------------------------------------------------------- pipeline
def test_ingest_stores_metadata(tmp_path, hash_embedder, corpus_dir):
    svc, store = _svc(tmp_path, hash_embedder)
    r = svc.ingest_file(str(corpus_dir / "backup_guide.pdf"))
    assert r.status == "indexed" and r.pages == 2 and r.chunks >= 2
    metas = store.get_all()["metadatas"]
    assert {m["page"] for m in metas} == {1, 2}
    assert all(m["doc_id"] == document_id_for("backup_guide.pdf") and m["source"] == "backup_guide.pdf" for m in metas)


def test_duplicate_ingestion_does_not_duplicate_chunks(tmp_path, hash_embedder, corpus_dir):
    svc, store = _svc(tmp_path, hash_embedder)
    first = svc.ingest_file(str(corpus_dir / "product_faq.txt"))
    n = store.count()
    second = svc.ingest_file(str(corpus_dir / "product_faq.txt"))
    assert first.status == "indexed" and second.status == "unchanged" and store.count() == n


def test_same_content_under_new_name_is_not_embedded_twice(tmp_path, hash_embedder, corpus_dir):
    svc, store = _svc(tmp_path, hash_embedder)
    svc.ingest_file(str(corpus_dir / "product_faq.txt"))
    n = store.count()
    copy = tmp_path / "faq_copy.txt"
    shutil.copy(corpus_dir / "product_faq.txt", copy)
    r = svc.ingest_file(str(copy))
    assert r.status == "duplicate" and r.duplicate_of == "product_faq.txt" and store.count() == n


def test_changed_file_is_reindexed_not_appended(tmp_path, hash_embedder):
    svc, store = _svc(tmp_path, hash_embedder)
    p = tmp_path / "notes.txt"
    p.write_text("Alpha fact one. " * 40)
    svc.ingest_file(str(p))
    p.write_text("Beta fact two.")
    r = svc.ingest_file(str(p))
    texts = store.get_all()["documents"]
    assert r.status == "reindexed" and texts == ["Beta fact two."]


def test_legacy_random_id_chunks_are_replaced_on_reupload(tmp_path, hash_embedder):
    svc, store = _svc(tmp_path, hash_embedder)
    store.upsert(["legacy-uuid-1"], [hash_embedder.embed_text("old")], ["old text"], [{"source": "notes.txt", "chunk_index": 0}])
    p = tmp_path / "notes.txt"
    p.write_text("New text for notes.")
    r = svc.ingest_file(str(p))
    assert r.status == "reindexed" and "legacy-uuid-1" not in store.get_all()["ids"]


def test_ingest_directory_reports_errors_without_stopping(tmp_path, hash_embedder):
    svc, store = _svc(tmp_path, hash_embedder)
    d = tmp_path / "in"
    d.mkdir()
    (d / "good.txt").write_text("Some useful content here.")
    (d / "empty.txt").write_text("")
    (d / "image.png").write_bytes(b"\x89PNG")
    res = {r.source: r.status for r in svc.ingest_directory(str(d))}
    assert res == {"empty.txt": "error", "good.txt": "indexed", "image.png": "error"}


def test_delete_document(tmp_path, hash_embedder, corpus_dir):
    svc, store = _svc(tmp_path, hash_embedder)
    r = svc.ingest_file(str(corpus_dir / "product_faq.txt"))
    assert svc.delete(r.document_id) == r.chunks and store.count() == 0


def test_embedding_model_mismatch_is_refused_and_data_kept(tmp_path, hash_embedder, corpus_dir):
    svc, store = _svc(tmp_path, hash_embedder)
    svc.ingest_file(str(corpus_dir / "product_faq.txt"))
    n = store.count()
    with pytest.raises(EmbeddingMismatchError):
        VectorStore(str(tmp_path / "db"), "ingest_test", "ollama:all-minilm")
    assert VectorStore(str(tmp_path / "db"), "ingest_test", hash_embedder.model_id).count() == n
