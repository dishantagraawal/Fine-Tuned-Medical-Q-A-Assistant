from app.services.rag import chunk_text


def test_chunk_text_overlap():
    text = "word " * 300
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)


def test_chunk_empty():
    assert chunk_text("   ", 100, 10) == []
