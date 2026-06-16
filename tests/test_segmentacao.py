"""A lição vira invariante: chunk fixo corta a resposta; estrutura preserva."""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from chunkers import (  # noqa: E402
    chunk_fixo,
    chunk_late,
    chunk_por_dispositivo,
    load_provisions,
)
from evaluate import load_queries, recall_at, span_integrity  # noqa: E402
from retriever import TfidfRetriever  # noqa: E402

provs = load_provisions(ROOT / "data" / "lei_lgpd.json")
queries = load_queries(ROOT / "data" / "consultas.json")

R_FIXO = TfidfRetriever(chunk_fixo(provs, size=120, overlap=20))
R_DISP = TfidfRetriever(chunk_por_dispositivo(provs))
R_LATE = TfidfRetriever(chunk_late(provs))


def test_chunk_fixo_corta_a_resposta():
    """O chunk fixo quase nunca entrega o trecho-resposta inteiro no topo."""
    assert span_integrity(R_FIXO, queries) <= 0.2


def test_por_dispositivo_preserva_a_resposta():
    """Por dispositivo entrega o trecho inteiro na grande maioria."""
    assert span_integrity(R_DISP, queries) >= 0.85


def test_fixo_perde_para_estrutura():
    """Estrutura supera o fixo em integridade e em recall."""
    assert span_integrity(R_DISP, queries) > span_integrity(R_FIXO, queries)
    assert recall_at(R_DISP, queries) > recall_at(R_FIXO, queries)


def test_late_nao_regride_a_integridade():
    """Late chunking mantém a unidade de sentido (não pior que por dispositivo)."""
    assert span_integrity(R_LATE, queries) >= span_integrity(R_DISP, queries)


def test_por_dispositivo_nao_corta_o_texto():
    """Cada chunk por dispositivo contém o texto íntegro do seu dispositivo."""
    by_id = {p["id"]: p["texto"] for p in provs}
    for c in chunk_por_dispositivo(provs):
        assert by_id[c.fonte] in c.original


def test_fixo_gera_mais_chunks_que_dispositivos():
    assert len(chunk_fixo(provs, 120, 20)) > len(chunk_por_dispositivo(provs))


def test_late_carrega_contexto_do_pai():
    """No late chunking, o chunk de um inciso inclui o caput do artigo na representação."""
    late = {c.fonte: c for c in chunk_late(provs)}
    inciso = late["art7_II"]
    assert "Art. 7" in inciso.indexavel and "hipóteses" in inciso.indexavel
