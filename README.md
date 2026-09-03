# Qdrant Demo Projects

![python](https://img.shields.io/badge/Python-3.11+-blue)
![qdrant](https://img.shields.io/badge/Qdrant-1.19-e11d48)
![license](https://img.shields.io/badge/License-MIT-1c1917)
![uv](https://img.shields.io/badge/uv-managed-0f766e)

Runnable Qdrant 1.19 tutorials, each in its own folder. Demos 01 and 02 are the classic pair (legal ranking + spatial hospitality), plus the per-tenant IDF benchmark.

Both classic demos default to `localhost:6333`. **One Qdrant is enough** - collection names do not collide. Do not `docker compose up` the second stack while the first still owns the port.

---

## Contents

| Folder | Theme | What it shows | Headline |
|---|---|---|---|
| [`01-jurisboost-legal-ranking/`](01-jurisboost-legal-ranking/) | Legal precedent RAG | Hybrid **RRF** → **ColBERT** → **FormulaQuery** (IPC/CrPC ↔ BNS/BNSS) | 19/22 Top-1 (86.4%), 21/22 Recall@3, no external reranker |
| [`02-geosmart-spatial-hospitality/`](02-geosmart-spatial-hospitality/) | Geospatial hospitality | `GaussDecay(GeoDistance)` + `ExpDecay(price)` + viewport `facet` | 37% near-area cliff → 0% ring loss in 8.7 ms |
| [`per-tenant-idf/`](per-tenant-idf/) | Multi-tenant BM25 | `params.idf.corpus` on BEIR CQADupStack | Isolated-tenant nDCG restored |

---

## How to run a classic demo

```bash
cd 01-jurisboost-legal-ranking   # or 02-geosmart-spatial-hospitality
docker compose up -d             # skip if Qdrant is already healthy on :6333
uv sync
uv run python main.py            # first run downloads data + indexes
```

GeoSmart also has a Leaflet map:

```bash
cd 02-geosmart-spatial-hospitality
uv run python main.py --ui       # http://127.0.0.1:8000
```

Each classic demo’s README is the source of truth (architecture, real query snippets, benchmark tables, CLI).

GPU is optional. `uv sync` installs CPU FastEmbed; `uv sync --extra gpu` if you have CUDA.

---

## License

[MIT](LICENSE)
