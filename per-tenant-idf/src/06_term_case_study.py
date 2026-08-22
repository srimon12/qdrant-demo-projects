"""Human-readable case study: WHICH terms get distorted by global IDF, and why.

Decodes mmh3 token ids back to words (fastembed bm25: id = abs(mmh3.hash(stem(word)))),
then compares global vs tenant-local IDF for the terms where the two disagree most.
"""

import math
from collections import Counter

import numpy as np
from py_rust_stemmers import SnowballStemmer

from common import load_embeddings

stemmer = SnowballStemmer("english")


def token_id(word: str) -> int:
    return abs(hash_stem(stemmer.stem_word(word)))


def hash_stem(stem: str) -> int:
    import mmh3

    return mmh3.hash(stem)


def main() -> None:
    _, forums, embs = load_embeddings()
    n_global = len(forums)
    unique_forums = sorted(set(forums.tolist()))

    # global df
    df_global: Counter[int] = Counter()
    for ind, _ in embs:
        df_global.update(set(ind.tolist()))

    # per-forum df
    df_forum: dict[str, Counter[int]] = {f: Counter() for f in unique_forums}
    n_forum = {f: int((forums == f).sum()) for f in unique_forums}
    for f_idx in range(len(forums)):
        df_forum[forums[f_idx]].update(set(embs[f_idx][0].tolist()))

    def okapi(n: int, df: int) -> float:
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def report(word: str) -> None:
        tid = token_id(word)
        dg = df_global.get(tid, 0)
        print(f"\n  '{word}'  (global df={dg}/{n_global} = {dg / n_global * 100:.1f}%)")
        print(f"    {'forum':<12}{'df_tenant':>10}{'%tenant':>9}{'idf_global':>11}{'idf_tenant':>11}{'ratio':>8}")
        for f in unique_forums:
            dt = df_forum[f].get(tid, 0)
            if dt == 0 and dg == 0:
                continue
            ig = okapi(n_global, dg)
            it = okapi(n_forum[f], dt)
            flag = "  <-- distorted" if abs(ig - it) > 0.35 and dt > 0 else ""
            print(f"    {f:<12}{dt:>10}{dt / n_forum[f] * 100:>8.1f}%{ig:>11.3f}{it:>11.3f}{it / ig:>7.2f}x{flag}")

    print("=" * 78)
    print("CASE STUDY: same word, different rarity depending on the IDF corpus")
    print("=" * 78)

    for w in ["wordpress", "plugin", "latex", "matrix", "android", "regex", "kernel"]:
        report(w)

    # ---- automatic: most distorted frequent terms per selected forum ----
    print("\n" + "=" * 78)
    print("MOST DISTORTED FREQUENT TERMS (df_tenant >= 2% of tenant, top 8 by |delta idf|)")
    print("=" * 78)

    # build reverse map only for stems we can recover from corpus text is expensive;
    # instead decode ids via dictionary of common English + forum-flavored words
    vocab_words = set()
    try:
        import pandas as pd

        from common import load_corpus

        df_txt = load_corpus()
        sample_text = " ".join(df_txt["title"].sample(min(20000, len(df_txt)), random_state=1).tolist())
        import re

        raw = re.findall(r"[a-zA-Z][a-zA-Z\-']{2,}", sample_text.lower())
        vocab_words.update(raw)
        print(f"(dictionary built from {len(vocab_words)} distinct corpus words)")
    except Exception as e:
        print("dictionary fallback:", e)

    id2word = {}
    for w in vocab_words:
        id2word.setdefault(token_id(w), w)

    for forum in ["tex", "wordpress", "gaming", "physics"]:
        n_f, dff = n_forum[forum], df_forum[forum]
        rows = []
        for tid, dt in dff.items():
            if dt < max(50, 0.02 * n_f):
                continue
            dg = df_global.get(tid, 0)
            delta = abs(okapi(n_global, dg) - okapi(n_f, dt))
            rows.append((delta, tid, dg, dt))
        rows.sort(reverse=True)
        print(f"\n  [{forum}] N={n_f}")
        print(f"    {'term':<18}{'df_global':>10}{'df_tenant':>10}{'idf_g':>8}{'idf_t':>8}{'delta':>8}")
        for delta, tid, dg, dt in rows[:8]:
            w = id2word.get(tid, f"?{tid}")
            print(f"    {w:<18}{dg:>10}{dt:>10}{okapi(n_global, dg):>8.3f}{okapi(n_f, dt):>8.3f}{delta:>8.3f}")


if __name__ == "__main__":
    main()
