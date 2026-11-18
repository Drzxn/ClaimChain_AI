"""Deterministic text chunking into passages.

Chunks are built from the *normalized extracted text*, so ``quote_start`` and
``quote_end`` are real character offsets into the text that produced them. That
matters: Stage 3 validates that every quoted span is a verbatim substring of
its passage, and Stage 6 shows the user the quote. Offsets that do not
correspond to the stored text make that impossible.

The chunker is deterministic: the same input always produces the same chunks
with the same indices and offsets. There is no randomness and no model.

Strategy: accumulate sentences until the token budget would be exceeded, then
emit a chunk. Consecutive chunks overlap by a small number of trailing
sentences so a claim spanning a boundary is not lost. Sentence splitting is
regex-based and deliberately simple; it does not need to be linguistically
perfect, only stable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Chunk budget in approximate tokens. One token ~= 4 characters is the usual
# English heuristic; it is used only to size chunks, never for billing.
DEFAULT_CHUNK_TOKENS = 400
DEFAULT_OVERLAP_SENTENCES = 1
CHARS_PER_TOKEN = 4

# Split after ., ! or ? followed by whitespace, keeping the delimiter.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True, slots=True)
class Chunk:
    """One passage, with offsets into the source text."""

    chunk_index: int
    text: str
    quote_start: int
    quote_end: int
    token_count: int


def estimate_tokens(text: str) -> int:
    """Approximate token count. Always at least 1 for non-empty text."""
    if not text:
        return 0
    return max(1, (len(text) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN)


def _split_sentences(text: str) -> list[tuple[str, int, int]]:
    """Split into ``(sentence, start, end)`` triples with exact offsets."""
    pieces: list[tuple[str, int, int]] = []
    cursor = 0

    for match in _SENTENCE_RE.finditer(text):
        end = match.start()
        sentence = text[cursor:end]
        if sentence.strip():
            pieces.append((sentence, cursor, end))
        cursor = match.end()

    tail = text[cursor:]
    if tail.strip():
        pieces.append((tail, cursor, len(text)))

    # A document with no sentence punctuation is one piece.
    if not pieces and text.strip():
        pieces.append((text, 0, len(text)))

    return pieces


def chunk_text(
    text: str,
    *,
    max_tokens: int = DEFAULT_CHUNK_TOKENS,
    overlap_sentences: int = DEFAULT_OVERLAP_SENTENCES,
) -> list[Chunk]:
    """Split normalized text into overlapping, offset-accurate chunks.

    Args:
        text: normalized extracted text.
        max_tokens: approximate token budget per chunk.
        overlap_sentences: trailing sentences repeated at the start of the next
            chunk. 0 disables overlap.

    Returns:
        Chunks in document order with ``chunk_index`` starting at 0. Empty
        input yields an empty list; no empty chunk is ever produced.
    """
    if not text or not text.strip():
        return []

    max_chars = max_tokens * CHARS_PER_TOKEN
    sentences = _split_sentences(text)

    chunks: list[Chunk] = []
    index = 0
    start_sentence = 0

    while start_sentence < len(sentences):
        end_sentence = start_sentence
        char_start = sentences[start_sentence][1]
        char_end = char_start

        # Grow the window while it fits the budget.
        while end_sentence < len(sentences):
            _, _, sentence_end = sentences[end_sentence]
            candidate_len = sentence_end - char_start
            if candidate_len > max_chars and end_sentence > start_sentence:
                break
            char_end = sentence_end
            end_sentence += 1

        # A single sentence longer than the budget is hard-split so no chunk
        # exceeds the cap and no text is silently dropped.
        if end_sentence == start_sentence + 1 and (char_end - char_start) > max_chars:
            piece_start = char_start
            while piece_start < char_end:
                piece_end = min(piece_start + max_chars, char_end)
                piece = text[piece_start:piece_end]
                if piece.strip():
                    chunks.append(
                        Chunk(
                            chunk_index=index,
                            text=piece,
                            quote_start=piece_start,
                            quote_end=piece_end,
                            token_count=estimate_tokens(piece),
                        )
                    )
                    index += 1
                piece_start = piece_end
            start_sentence = end_sentence
            continue

        piece = text[char_start:char_end]
        if piece.strip():
            chunks.append(
                Chunk(
                    chunk_index=index,
                    text=piece,
                    quote_start=char_start,
                    quote_end=char_end,
                    token_count=estimate_tokens(piece),
                )
            )
            index += 1

        if end_sentence >= len(sentences):
            break

        # Step forward, backing up for overlap so a boundary-spanning claim
        # still appears whole in at least one chunk.
        next_start = end_sentence - overlap_sentences
        start_sentence = max(next_start, start_sentence + 1)

    return chunks
