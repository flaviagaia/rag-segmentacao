"""Três estratégias de segmentação do documento antes de indexar.

A segmentação é uma decisão de PRÉ-PROCESSAMENTO que muda o que o recuperador
consegue achar. Comparamos:

- fixo            : janela deslizante por caracteres, com sobreposição. Ignora a
                    estrutura e pode CORTAR a resposta no meio.
- por_dispositivo : um chunk por unidade jurídica (artigo/inciso). Respeita a
                    unidade de sentido, então a resposta nunca é cortada.
- late_chunking   : o chunk por dispositivo, mas sua REPRESENTAÇÃO carrega o
                    contexto do documento (aqui, o caput do artigo). Proxy offline
                    do late chunking (Günther et al., 2024), que no original opera
                    nos embeddings de token com um modelo de contexto longo.

Saída comum: lista de Chunk(id, texto_indexavel, fonte_id, texto_original).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Chunk:
    id: str
    indexavel: str       # texto usado na indexação/recuperação
    fonte: str           # id do dispositivo de origem
    original: str        # texto do dispositivo (para checar a resposta)


def load_provisions(path: Path) -> list[dict]:
    """Dispositivos como {id, rotulo, texto, parent}."""
    return json.loads(path.read_text(encoding="utf-8"))["nodes"]


def full_document(provs: list[dict]) -> str:
    return "\n".join(f"{p['rotulo']}: {p['texto']}" for p in provs)


def _provision_of_offset(provs: list[dict], doc: str, start: int) -> str:
    """Descobre a qual dispositivo pertence o início de uma janela (por offset)."""
    pos, atual = 0, provs[0]["id"]
    for p in provs:
        linha = f"{p['rotulo']}: {p['texto']}\n"
        if pos <= start < pos + len(linha):
            return p["id"]
        pos += len(linha)
        atual = p["id"]
    return atual


# ----------------------------------------------------------------- fixo
def chunk_fixo(provs: list[dict], size: int = 120, overlap: int = 20) -> list[Chunk]:
    doc = full_document(provs)
    step = max(size - overlap, 1)
    chunks = []
    for i, start in enumerate(range(0, len(doc), step)):
        seg = doc[start:start + size]
        if seg.strip():
            chunks.append(Chunk(f"fixo::{i}", seg, _provision_of_offset(provs, doc, start), seg))
        if start + size >= len(doc):
            break
    return chunks


# --------------------------------------------------------- por dispositivo
def chunk_por_dispositivo(provs: list[dict]) -> list[Chunk]:
    out = []
    for p in provs:
        txt = f"{p['rotulo']}: {p['texto']}"
        out.append(Chunk(p["id"], txt, p["id"], p["texto"]))
    return out


# ------------------------------------------------------------ late chunking
def chunk_late(provs: list[dict]) -> list[Chunk]:
    by_id = {p["id"]: p for p in provs}
    out = []
    for p in provs:
        pai = by_id.get(p["parent"]) if p.get("parent") else None
        contexto = f"{pai['rotulo']}: {pai['texto']} " if pai else ""
        # a representação carrega o contexto do documento; o texto-resposta é o do dispositivo
        indexavel = f"{contexto}{p['rotulo']}: {p['texto']}"
        out.append(Chunk(p["id"], indexavel, p["id"], p["texto"]))
    return out
