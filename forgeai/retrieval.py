from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable

from .documents import Chunk


STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for", "from", "how",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "our", "the", "their", "to", "was",
    "what", "when", "where", "which", "who", "why", "with", "you", "your",
}


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z0-9]+(?:[.-][a-zA-Z0-9]+)*", text.lower())
    return [token for token in tokens if token not in STOP_WORDS and len(token) > 1]


class BM25Retriever:
    def __init__(self, chunks: Iterable[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = list(chunks)
        self.k1 = k1
        self.b = b
        self.term_frequencies = [Counter(tokenize(chunk.text)) for chunk in self.chunks]
        self.lengths = [len(tokens) for tokens in self.term_frequencies]
        self.average_length = sum(self.lengths) / len(self.lengths) if self.lengths else 1.0
        document_frequency: Counter[str] = Counter()
        for frequencies in self.term_frequencies:
            document_frequency.update(frequencies.keys())
        count = len(self.chunks)
        self.idf = {
            term: math.log(1 + (count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def search(self, query: str, top_k: int = 4, min_score: float = 0.0) -> list[tuple[Chunk, float]]:
        query_terms = tokenize(query)
        scores: list[tuple[Chunk, float]] = []
        for chunk, frequencies, length in zip(self.chunks, self.term_frequencies, self.lengths):
            score = 0.0
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                denominator = frequency + self.k1 * (1 - self.b + self.b * length / max(self.average_length, 1))
                score += self.idf.get(term, 0.0) * frequency * (self.k1 + 1) / denominator
            lowered = query.lower()
            if lowered and lowered in chunk.text.lower():
                score += 2.0
            scores.append((chunk, score))
        scores.sort(key=lambda item: (-item[1], item[0].chunk_id))
        return [item for item in scores[:top_k] if item[1] >= min_score]
