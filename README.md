# Segmentação para RAG: chunk fixo vs por dispositivo vs late chunking

[🇧🇷 Português](#-português) · [🇺🇸 English](#-english)

Python 3.10+ · scikit-learn · 100% offline, sem API key · MIT License

Dados públicos: LGPD (Lei nº 13.709/2018), arts. 6º e 7º.

> **Em uma frase:** a forma como você corta o documento antes de indexar decide se
> o RAG entrega a resposta inteira ou um pedaço dela. O chunk de tamanho fixo
> entregou **0%** dos trechos-resposta íntegros no topo; cortar por unidade de
> sentido (artigo/inciso) entregou **88%**.

---

## 🇧🇷 Português

### O problema
Antes de recuperar, o documento é fatiado em chunks. A escolha mais comum, **chunk
de tamanho fixo** (janela de N caracteres com sobreposição), ignora a estrutura e
corta a resposta no meio: a janela que casa com a pergunta pode não conter o trecho
que de fato responde. Em texto jurídico, a unidade de sentido é o dispositivo
(artigo, parágrafo, inciso), então faz sentido cortar por ali.

### As três estratégias
```
fixo            : janela deslizante de N chars (overlap). Rápido, mas corta a resposta.
por dispositivo : um chunk por artigo/inciso. Respeita a unidade de sentido.
late chunking   : o chunk por dispositivo cuja REPRESENTAÇÃO carrega o contexto do
                  documento (aqui, o caput do artigo). Proxy offline do late chunking
                  de Günther et al. (2024), que no original pondera embeddings de
                  token de um modelo de contexto longo.
```
O recuperador (TF-IDF) é o **mesmo** para as três; a única variável é o corte.

### Resultado real deste repositório
8 consultas com trecho-resposta (gabarito), métricas que separam *achar* de *entregar*:

| estratégia | nº chunks | recall@3 | integridade@1 |
| ---------- | --------- | -------- | ------------- |
| fixo (120 chars) | 37 | 0.50 | **0.00** |
| por dispositivo | 22 | 0.88 | **0.88** |
| late chunking | 22 | 0.88 | 0.88 |

- **recall@3** = achou a área certa. **integridade@1** = o chunk do topo traz o
  trecho-resposta **inteiro**.
- O chunk fixo até acha às vezes (recall 0.50), mas entrega **cortado** (integridade
  0.00): a resposta fica partida entre janelas.
- Cortar por dispositivo preserva a resposta. Late chunking preserva igual e, além
  disso, carrega o contexto do documento na representação do chunk.

### Sobre o late chunking (honestidade)
No nível lexical (TF-IDF), o late chunking empata com o "por dispositivo": os dois
preservam a unidade. O ganho próprio do late chunking, fazer o **embedding** do
chunk carregar o contexto do documento, aparece com um modelo neural de contexto
longo (o método original). Por isso ele entra aqui como a evolução natural do corte
por estrutura, e a representação contextualizada já fica montada para o modo com
embeddings reais.

### Como explicar em 30 segundos
Cortar documento por tamanho fixo é como rasgar uma página no meio da frase: você
acha a página, mas a resposta ficou partida. Cortar por artigo/inciso é destacar a
frase inteira. Late chunking é destacar a frase sem esquecer de que capítulo ela veio.

### Execução
```
pip install -r requirements.txt
python src/demo.py        # as três estratégias, com números reais
pytest tests/ -v          # 7 testes
```

### Estrutura
```
data/lei_lgpd.json   # dispositivos (arts. 6º e 7º)
data/consultas.json  # consultas + trecho-resposta (gabarito)
src/chunkers.py      # fixo, por dispositivo, late chunking
src/retriever.py     # TF-IDF (igual para as três)
src/evaluate.py      # recall@3 e integridade do trecho-resposta
src/demo.py          # a comparação
```

### Limitações honestas
Corpus pequeno; recuperador lexical (TF-IDF). O efeito do chunk fixo depende do
tamanho da janela (aqui 120 chars) em relação ao tamanho do dispositivo. O benefício
pleno do late chunking exige embeddings de contexto longo. O objetivo é isolar **por
que** o corte por estrutura preserva a resposta que o corte fixo parte.

### Referências científicas (crédito aos autores)
- Günther et al. (2024). Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models. Jina AI, arXiv:2409.04701.
- Lewis et al. (2020). Retrieval-Augmented Generation. NeurIPS. · Gao et al. (2024). RAG: A Survey. arXiv:2312.10997.
- LGPD (Lei nº 13.709/2018): ato oficial de domínio público (Planalto/gov.br).

Reimplementação didática e offline; crédito às autoras e aos autores originais.

---

## 🇺🇸 English

**In one line:** how you split the document before indexing decides whether the RAG
returns the whole answer or a fragment. Fixed-size chunking delivered **0%** of the
answer spans intact at the top; cutting by unit of meaning (article/inciso) delivered
**88%**.

### The three strategies
Fixed-size sliding window; per-provision (structure-aware); late chunking (the
per-provision chunk whose representation carries the document context — an offline
proxy of Günther et al., 2024). Same TF-IDF retriever for all three.

### Real result
| strategy | chunks | recall@3 | span integrity@1 |
| -------- | ------ | -------- | ---------------- |
| fixed (120 chars) | 37 | 0.50 | 0.00 |
| per provision | 22 | 0.88 | 0.88 |
| late chunking | 22 | 0.88 | 0.88 |

Fixed-size sometimes finds the area (recall 0.50) but delivers it **cut** (integrity
0.00). Structure-aware cutting preserves the answer. At the lexical level late
chunking ties per-provision; its own gain (document context inside the chunk's
embedding) shows with a long-context neural model.

### Running
```
pip install -r requirements.txt
python src/demo.py
pytest tests/ -v          # 7 tests
```

### References
Günther et al. (2024), Late Chunking; Lewis et al. (2020), RAG; Gao et al. (2024), survey.

---

Part of my LinkedIn series on RAG efficiency → [Flávia Gaia](https://www.linkedin.com/in/flavia-gaia/)
