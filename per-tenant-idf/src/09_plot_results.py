"""Render the benchmark with XY and write PNG/SVG/HTML under figures/."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import xy

from common import ART_DIR, FORUMS, ROOT

FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

# Qdrant-adjacent: slate for the default (global) mode, signal red for 1.19.
GLOBAL_C = "#334155"
TENANT_C = "#E11D48"
DELTA_C = "#0F766E"
BG = "#FAFAF9"
GRID = "#E7E5E4"
INK = "#1C1917"

THEME = xy.theme(
    background=BG,
    plot_background=BG,
    grid_color=GRID,
    axis_color="#A8A29E",
    text_color=INK,
)


def _export(chart, stem: str) -> None:
    png = FIG / f"{stem}.png"
    svg = FIG / f"{stem}.svg"
    html = FIG / f"{stem}.html"
    chart.to_png(str(png), width=1100, height=520, scale=2)
    chart.to_svg(str(svg), width=1100, height=520)
    chart.to_html(str(html))
    print(f"  {png.name}")


def _grouped_columns(categories, series: dict[str, np.ndarray], colors: list[str], *children, **chart_kw):
    names = list(series)
    mat = np.column_stack([np.asarray(series[n], dtype=float) for n in names])
    return xy.column_chart(
        xy.column(
            list(categories), mat, series=names, colors=colors, mode="grouped", corner_radius=3
        ),
        *children,
        xy.legend(),
        THEME,
        **chart_kw,
    )


def plot_correctness_ndcg(acc: pd.DataFrame) -> None:
    order = [f for f in FORUMS if f in set(acc["forum"])]
    g = acc.groupby("forum")["ndcg_global"].mean().reindex(order).to_numpy()
    t = acc.groupby("forum")["ndcg_tenant"].mean().reindex(order).to_numpy()
    chart = _grouped_columns(
        order,
        {"Global IDF": g, "Per-tenant IDF": t},
        [GLOBAL_C, TENANT_C],
        xy.y_axis(label="nDCG@10 vs tenant-local BM25", domain=(0.95, 1.0)),
        xy.x_axis(label="forum (tenant)"),
        title="Correctness: does Qdrant reproduce tenant-local BM25?",
        width=1100,
        height=520,
    )
    _export(chart, "01_correctness_ndcg")


def plot_ranking_shift(acc: pd.DataFrame) -> None:
    order = [f for f in FORUMS if f in set(acc["forum"])]
    t1 = (acc.groupby("forum")["top1_changed"].mean().reindex(order) * 100).to_numpy()
    ov = ((1 - acc.groupby("forum")["top10_overlap"].mean().reindex(order)) * 100).to_numpy()
    chart = _grouped_columns(
        order,
        {"% top-1 changed": t1, "% of top-10 that differs": ov},
        [TENANT_C, GLOBAL_C],
        xy.y_axis(label="percent of queries"),
        xy.x_axis(label="forum (tenant)"),
        title="Operational impact: global IDF changes the ranking",
        width=1100,
        height=520,
    )
    _export(chart, "02_ranking_shift")


def plot_ndcg_delta_hist(acc: pd.DataFrame) -> None:
    delta = (acc["ndcg_tenant"] - acc["ndcg_global"]).to_numpy()
    chart = xy.histogram_chart(
        xy.histogram(delta, bins=40, color=DELTA_C, opacity=0.9),
        xy.x_axis(label="nDCG@10 tenant − nDCG@10 global"),
        xy.y_axis(label="queries"),
        THEME,
        title="Per-query correctness gap (positive = global IDF hurts ranking)",
        width=1100,
        height=480,
    )
    _export(chart, "03_ndcg_delta_hist")


def plot_beir(beir: pd.DataFrame) -> None:
    order = [f for f in FORUMS if f in set(beir["forum"])]
    g = beir.groupby("forum")["ndcg10_global"].mean().reindex(order).to_numpy()
    t = beir.groupby("forum")["ndcg10_tenant"].mean().reindex(order).to_numpy()
    chart = _grouped_columns(
        order,
        {"Global IDF": g, "Per-tenant IDF": t},
        [GLOBAL_C, TENANT_C],
        xy.y_axis(label="nDCG@10 (duplicate-question qrels)"),
        xy.x_axis(label="forum (tenant)"),
        title="BEIR CQADupStack: labeled retrieval quality",
        width=1100,
        height=520,
    )
    _export(chart, "04_beir_ndcg")

    # delta per forum
    d = (t - g) * 100
    chart = xy.column_chart(
        xy.column(order, d, name="Δ nDCG@10 (pp)", color=DELTA_C, corner_radius=3),
        xy.y_axis(label="percentage points (tenant − global)"),
        xy.x_axis(label="forum (tenant)"),
        THEME,
        title="BEIR nDCG@10 lift from scoping IDF to the tenant",
        width=1100,
        height=480,
    )
    _export(chart, "05_beir_ndcg_lift")


def plot_latency(lat: pd.DataFrame, per_q: pd.DataFrame | None) -> None:
    labels = ["Global IDF" if m == "global" else "Per-tenant IDF" for m in lat["mode"].tolist()]
    p50 = lat["p50_ms"].to_numpy()
    p95 = lat["p95_ms"].to_numpy()
    chart = _grouped_columns(
        labels,
        {"p50": p50, "p95": p95},
        [GLOBAL_C, TENANT_C],
        xy.y_axis(label="client-side latency (ms)"),
        title="Latency: same query, same filter, only IDF corpus changes",
        width=900,
        height=480,
    )
    _export(chart, "06_latency")

    if per_q is None or per_q.empty:
        return
    # one point per query = mean over reps
    agg = per_q.groupby("i").agg(g=("global_ms", "mean"), t=("tenant_ms", "mean")).reset_index()
    chart = xy.scatter_chart(
        xy.scatter(agg["g"].to_numpy(), agg["t"].to_numpy(), color=DELTA_C, size=4, opacity=0.55),
        xy.x_axis(label="global IDF latency (ms)"),
        xy.y_axis(label="per-tenant IDF latency (ms)"),
        THEME,
        title="Per-query latency: tenant-scoped IDF vs global",
        width=720,
        height=720,
    )
    _export(chart, "07_latency_scatter")


def plot_overview(acc: pd.DataFrame, beir: pd.DataFrame | None, lat: pd.DataFrame | None) -> None:
    """Headline strip: the numbers a reader should remember, one labeled row each.

    The metrics are in different units (%, pp, ms), so we put each value + unit
    on the row label and let the bar be a rough visual cue; a shared numeric
    axis would make the 0.19 pp and 0.89 ms bars impossible to read.
    """
    rows: list[tuple[str, float, str, str]] = []
    rows.append((
        "Top-1 changed",
        float(acc["top1_changed"].mean() * 100),
        "%",
        TENANT_C,
    ))
    rows.append((
        "nDCG loss vs global",
        float((acc["ndcg_tenant"] - acc["ndcg_global"]).mean() * 100),
        "pp",
        DELTA_C,
    ))
    if beir is not None and not beir.empty:
        rows.append((
            "BEIR nDCG@10 lift",
            float((beir["ndcg10_tenant"] - beir["ndcg10_global"]).mean() * 100),
            "pp",
            "#0369A1",
        ))
    if lat is not None and not lat.empty:
        g = float(lat.loc[lat["mode"] == "global", "p50_ms"].iloc[0])
        t = float(lat.loc[lat["mode"] == "tenant", "p50_ms"].iloc[0])
        rows.append(("p50 overhead", t - g, "ms", GLOBAL_C))

    labels = [f"{name}   {val:+.2f} {unit}" for name, val, unit, _ in rows]
    values = [val for _, val, _, _ in rows]
    colors = [col for _, _, _, col in rows]
    chart = xy.bar_chart(
        *[
            xy.bar([lab], np.array([val]), color=col, orientation="horizontal", corner_radius=4)
            for lab, val, col in zip(labels, values, colors)
        ],
        xy.x_axis(label="value (units differ \u2014 read the labels)"),
        THEME,
        title="Headline numbers",
        width=1100,
        height=340,
    )
    _export(chart, "00_headline")


def main() -> None:
    acc_path = ART_DIR / "accuracy_results.csv"
    if not acc_path.exists():
        raise SystemExit("artifacts/accuracy_results.csv missing — run src/03_accuracy_benchmark.py")
    acc = pd.read_csv(acc_path)
    print(f"accuracy csv: {len(acc)} queries")

    beir = None
    beir_path = ART_DIR / "beir_eval.csv"
    if beir_path.exists():
        beir = pd.read_csv(beir_path)
        print(f"beir csv: {len(beir)} queries")

    lat = None
    per_q = None
    lat_path = ART_DIR / "latency_results.csv"
    if lat_path.exists():
        lat = pd.read_csv(lat_path)
        print("latency csv loaded")
    per_path = ART_DIR / "latency_per_query.csv"
    if per_path.exists():
        per_q = pd.read_csv(per_path)

    print("rendering…")
    plot_overview(acc, beir, lat)
    plot_correctness_ndcg(acc)
    plot_ranking_shift(acc)
    plot_ndcg_delta_hist(acc)
    if beir is not None:
        plot_beir(beir)
    if lat is not None:
        plot_latency(lat, per_q)
    print(f"\nfigures -> {FIG}")


if __name__ == "__main__":
    main()
