"""TF-IDF sobre os chunks. O recuperador é o MESMO para as três segmentações,
então a única variável do experimento é como o documento foi cortado.
"""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from chunkers import Chunk


class TfidfRetriever:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self._vec = TfidfVectorizer(ngram_range=(1, 2), strip_accents="unicode")
        self._matrix = self._vec.fit_transform(c.indexavel for c in chunks)

    def search(self, query: str, top_k: int) -> list[Chunk]:
        scores = cosine_similarity(self._vec.transform([query]), self._matrix).ravel()
        order = scores.argsort()[::-1][:top_k]
        return [self.chunks[i] for i in order]
