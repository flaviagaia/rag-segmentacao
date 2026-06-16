"""Avaliação da segmentação.

Duas métricas que separam achar de entregar:
- recall@3        : algum chunk do dispositivo certo aparece no top-3 (achou a área)?
- integridade@1   : o chunk do topo CONTÉM o trecho-resposta inteiro (entregou a
                    resposta sem cortar)? É aqui que o chunk fixo falha.
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path


def load_queries(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def _norm(t: str) -> str:
    t = unicodedata.normalize("NFKD", t.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return " ".join(t.split())


def recall_at(retriever, queries: list[dict], k: int = 3) -> float:
    hits = 0
    for item in queries:
        chunks = retriever.search(item["q"], k)
        if any(c.fonte == item["fonte"] for c in chunks):
            hits += 1
    return hits / len(queries)


def span_integrity(retriever, queries: list[dict]) -> float:
    """Fração das consultas em que o chunk do TOPO contém o trecho-resposta inteiro."""
    ok = 0
    for item in queries:
        top = retriever.search(item["q"], 1)[0]
        if _norm(item["span"]) in _norm(top.original):
            ok += 1
    return ok / len(queries)
