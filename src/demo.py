"""Demo: três segmentações sobre a mesma LGPD, mesmo recuperador (~1s).

    python src/demo.py
"""

from __future__ import annotations

from pathlib import Path

from chunkers import chunk_fixo, chunk_late, chunk_por_dispositivo, load_provisions
from evaluate import load_queries, recall_at, span_integrity
from retriever import TfidfRetriever

ROOT = Path(__file__).parent.parent
FIXO_SIZE = 120


def main() -> None:
    provs = load_provisions(ROOT / "data" / "lei_lgpd.json")
    queries = load_queries(ROOT / "data" / "consultas.json")

    estrategias = {
        f"fixo ({FIXO_SIZE} chars)": chunk_fixo(provs, size=FIXO_SIZE, overlap=20),
        "por dispositivo": chunk_por_dispositivo(provs),
        "late chunking": chunk_late(provs),
    }

    print("=" * 64)
    print("Segmentação: como cortar o documento muda a resposta (LGPD)")
    print("=" * 64)
    print(f"{'estratégia':<22}{'nº chunks':>10}{'recall@3':>11}{'integridade@1':>16}")
    print("-" * 64)
    for nome, chunks in estrategias.items():
        r = TfidfRetriever(chunks)
        print(f"{nome:<22}{len(chunks):>10}{recall_at(r, queries):>11.2f}"
              f"{span_integrity(r, queries):>16.2f}")

    print("\n  recall@3 = achou a área certa. integridade@1 = o chunk do topo traz o")
    print("  trecho-resposta INTEIRO. O chunk fixo às vezes acha mas entrega cortado;")
    print("  por dispositivo e late chunking preservam a unidade de sentido.")


if __name__ == "__main__":
    main()
