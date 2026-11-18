"""Extraction, chunking and relevance prefilter tests."""

from __future__ import annotations

import pytest

from app.services.chunk import chunk_text, estimate_tokens
from app.services.extract import ExtractionError, extract_text
from app.services.relevance import score_passages, select_relevant, tokenize

LONG_PARAGRAPH = (
    "HTTPS encrypts data in transit between the browser and the server. "
    "It protects against passive eavesdropping on untrusted networks. "
    "It does not prevent cross-site scripting or SQL injection attacks. "
) * 6


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def test_extracts_visible_text() -> None:
    html = "<html><body><p>HTTPS encrypts traffic in transit between the browser and the server over the network.</p></body></html>"
    result = extract_text(html)
    assert "HTTPS encrypts traffic in transit" in result.text


def test_removes_scripts_and_styles() -> None:
    html = """
    <html><head>
      <style>.hidden { display: none; }</style>
      <script>var secret = "do not keep me";</script>
    </head><body>
      <p>This is the readable paragraph that should survive extraction intact after all the noise is stripped away.</p>
    </body></html>
    """
    result = extract_text(html)
    assert "do not keep me" not in result.text
    assert "display: none" not in result.text
    assert "readable paragraph" in result.text


def test_removes_navigation_and_footer() -> None:
    html = """
    <html><body>
      <nav><a href="/">Home</a><a href="/about">About</a></nav>
      <main><p>The main content paragraph lives here and is comfortably long enough to survive the minimum length check.</p></main>
      <footer><p>Copyright notice boilerplate text goes here.</p></footer>
    </body></html>
    """
    result = extract_text(html)
    assert "main content paragraph" in result.text
    assert "Copyright notice" not in result.text


def test_extracts_title() -> None:
    html = "<html><head><title>How HTTPS Works</title></head><body><p>" + LONG_PARAGRAPH + "</p></body></html>"
    assert extract_text(html).title == "How HTTPS Works"


def test_rejects_empty_document() -> None:
    with pytest.raises(ExtractionError):
        extract_text("")


def test_rejects_too_short_content() -> None:
    with pytest.raises(ExtractionError, match="too short"):
        extract_text("<html><body><p>tiny</p></body></html>")


def test_plain_text_content_type() -> None:
    result = extract_text(LONG_PARAGRAPH, content_type="text/plain")
    assert "HTTPS encrypts" in result.text


def test_plain_text_too_short_rejected() -> None:
    with pytest.raises(ExtractionError):
        extract_text("short", content_type="text/plain")


def test_whitespace_is_normalized() -> None:
    html = "<html><body><p>A   paragraph\n\nwith    irregular\twhitespace that should be normalized by the extractor.</p></body></html>"
    result = extract_text(html)
    assert "  " not in result.text.replace("\n", "")


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def test_chunking_is_deterministic() -> None:
    a = chunk_text(LONG_PARAGRAPH, max_tokens=50)
    b = chunk_text(LONG_PARAGRAPH, max_tokens=50)
    assert [(c.chunk_index, c.text, c.quote_start, c.quote_end) for c in a] == [
        (c.chunk_index, c.text, c.quote_start, c.quote_end) for c in b
    ]


def test_chunk_indices_are_sequential_from_zero() -> None:
    chunks = chunk_text(LONG_PARAGRAPH, max_tokens=50)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_offsets_match_the_source_text() -> None:
    """quote_start/quote_end must slice back to exactly the chunk text."""
    chunks = chunk_text(LONG_PARAGRAPH, max_tokens=50)
    for chunk in chunks:
        assert LONG_PARAGRAPH[chunk.quote_start:chunk.quote_end] == chunk.text


def test_no_empty_chunks() -> None:
    for chunk in chunk_text(LONG_PARAGRAPH, max_tokens=50):
        assert chunk.text.strip()


def test_chunks_respect_token_budget() -> None:
    chunks = chunk_text(LONG_PARAGRAPH, max_tokens=40)
    for chunk in chunks:
        # Allow one sentence of slack: the budget is enforced at sentence
        # boundaries, so a chunk can be at most one sentence over.
        assert chunk.token_count <= 40 + estimate_tokens(
            "It does not prevent cross-site scripting or SQL injection attacks."
        )


def test_empty_input_yields_no_chunks() -> None:
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_short_text_is_one_chunk() -> None:
    chunks = chunk_text("A single short sentence.", max_tokens=400)
    assert len(chunks) == 1
    assert chunks[0].quote_start == 0


def test_overlap_repeats_boundary_sentence() -> None:
    """With overlap enabled, consecutive chunks share a character range.

    Overlap only applies when a chunk holds more than one sentence. With a
    budget of one sentence per chunk, overlapping would make no progress, so
    the chunker correctly suppresses it.
    """
    chunks = chunk_text(LONG_PARAGRAPH, max_tokens=120, overlap_sentences=1)
    assert len(chunks) > 1
    assert chunks[1].quote_start < chunks[0].quote_end


def test_overlap_suppressed_when_chunks_are_single_sentence() -> None:
    """A one-sentence budget must not produce overlapping chunks."""
    chunks = chunk_text(LONG_PARAGRAPH, max_tokens=30, overlap_sentences=1)
    assert len(chunks) > 1
    for previous, current in zip(chunks, chunks[1:]):
        assert current.quote_start >= previous.quote_end


def test_no_overlap_when_disabled() -> None:
    chunks = chunk_text(LONG_PARAGRAPH, max_tokens=30, overlap_sentences=0)
    for previous, current in zip(chunks, chunks[1:]):
        assert current.quote_start >= previous.quote_end


def test_very_long_single_sentence_is_hard_split() -> None:
    """A sentence over budget must still be chunked, not dropped."""
    long_sentence = "word " * 500 + "end."
    chunks = chunk_text(long_sentence, max_tokens=50)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.text.strip()
        assert long_sentence[chunk.quote_start:chunk.quote_end] == chunk.text


# ---------------------------------------------------------------------------
# Relevance prefilter
# ---------------------------------------------------------------------------

def test_tokenize_removes_stopwords() -> None:
    terms = tokenize("The HTTPS is a protocol for the web")
    assert "the" not in terms
    assert "is" not in terms
    assert "https" in terms


def test_tokenize_stems_inflections() -> None:
    assert tokenize("encrypts") == tokenize("encrypted")


def test_relevant_passage_scores_above_irrelevant() -> None:
    scores = score_passages(
        "HTTPS encrypts data in transit",
        [
            "HTTPS encrypts data in transit between browser and server.",
            "The recipe calls for two cups of flour and a pinch of salt.",
        ],
    )
    assert scores[0] > scores[1]


def test_irrelevant_passage_removed() -> None:
    passages = [
        "HTTPS encrypts data in transit between the browser and the server.",
        "A recipe for chocolate cake with butter and sugar and eggs.",
    ]
    kept = select_relevant("HTTPS encrypts data in transit", passages)
    assert 0 in kept
    assert 1 not in kept


def test_keeps_at_least_min_keep() -> None:
    """The floor applies among passages that actually share a content term."""
    passages = [
        "HTTPS encrypts traffic in transit between browser and server.",
        "Transport encryption protects the channel with confidentiality.",
        "TLS provides confidentiality for the encrypted connection.",
    ]
    kept = select_relevant("HTTPS encrypts data", passages, min_keep=2)
    assert len(kept) >= 2


def test_floor_never_resurrects_zero_score_passages() -> None:
    """A passage sharing no content term is dropped even below the floor.

    Keeping a passage that mentions nothing in the subclaim would waste Stage 3
    calls and, worse, invite a spurious INSUFFICIENT link.
    """
    passages = [
        "HTTPS encrypts traffic in transit.",
        "A recipe for chocolate cake with butter and sugar.",
        "Notes on bicycle maintenance and tyre pressure.",
    ]
    kept = select_relevant("HTTPS encrypts data", passages, min_keep=3)
    assert kept == [0]


def test_zero_scores_are_dropped() -> None:
    passages = [
        "Completely unrelated text about gardening and soil.",
        "More unrelated text about bicycles and mountains.",
    ]
    kept = select_relevant("HTTPS encrypts data in transit", passages)
    assert kept == []


def test_empty_passage_list_returns_empty() -> None:
    assert select_relevant("anything", []) == []


def test_prefilter_does_not_classify_relationships() -> None:
    """The prefilter returns indices only: no SUPPORTS/CONTRADICTS anywhere."""
    kept = select_relevant(
        "HTTPS encrypts data",
        ["HTTPS encrypts data in transit between browser and server."],
    )
    assert all(isinstance(i, int) for i in kept)

