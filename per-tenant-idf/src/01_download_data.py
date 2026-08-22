"""Download the FULL BeIR/cqadupstack corpus + queries from HuggingFace.

457k real StackExchange posts across 12 forums (= tenants), plus held-out
duplicate-question queries per forum. Saved locally as parquet.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

FORUMS = [
    "android", "english", "gaming", "gis", "mathematica", "physics",
    "programmers", "stats", "tex", "unix", "webmasters", "wordpress",
]


def main() -> None:
    total_c, total_q = 0, 0
    for forum in FORUMS:
        c_path = DATA_DIR / f"beir_{forum}_corpus.parquet"
        q_path = DATA_DIR / f"beir_{forum}_queries.parquet"
        if c_path.exists() and q_path.exists():
            c, q = pd.read_parquet(c_path), pd.read_parquet(q_path)
        else:
            base = f"https://huggingface.co/datasets/BeIR/cqadupstack/resolve/main/{forum}"
            c = pd.read_parquet(f"{base}/corpus/corpus-00000-of-00001.parquet")
            q = pd.read_parquet(f"{base}/queries/queries-00000-of-00001.parquet")
            c.to_parquet(c_path)
            q.to_parquet(q_path)
        total_c += len(c)
        total_q += len(q)
        print(f"  {forum:<12} corpus={len(c):>7}  queries={len(q):>6}")

    print(f"\ntotal corpus: {total_c} docs   total queries: {total_q}")

    # Official TREC qrels (duplicate-question labels). Numeric ids collide
    # across forums, so we materialise one parquet per forum.
    from common import load_qrels

    qrels = load_qrels()
    n_qrels = sum(sum(len(d) for d in per_q.values()) for per_q in qrels.values())
    n_topics = sum(len(per_q) for per_q in qrels.values())
    print(f"qrels: {n_qrels} judgments over {n_topics} queries")


if __name__ == "__main__":
    main()
