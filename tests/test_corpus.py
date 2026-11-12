from belanjawan.corpus import chunk_page


def test_chunks_overlap_when_no_break_is_found():
    text = "word " * 240  # 1,200 chars, no newline or sentence end
    chunks = chunk_page(text, size=400, overlap=100)
    assert len(chunks) >= 3
    assert chunks[0][-60:] in chunks[1]


def test_prefers_cutting_at_a_newline():
    text = "A" * 350 + "\n" + "B" * 400
    assert chunk_page(text, size=400, overlap=50)[0] == "A" * 350


def test_falls_back_to_sentence_end():
    text = "x" * 300 + ". " + "y" * 300
    assert chunk_page(text, size=400, overlap=50)[0] == "x" * 300 + "."


def test_drops_tiny_fragments():
    assert chunk_page("  12  ") == []
    assert all(len(c) >= 40 for c in chunk_page("Long sentence here. " * 60))


def test_whole_text_is_covered():
    text = "".join(f"Line {i} of the budget speech.\n" for i in range(100))
    joined = " ".join(chunk_page(text))
    assert "Line 0 " in joined and "Line 99 " in joined
