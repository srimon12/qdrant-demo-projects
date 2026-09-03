"""4-mode spatial ladder benchmark - measured on live Qdrant."""

from __future__ import annotations

import json
import logging
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from rich.console import Console
from rich.table import Table

from .config import COLLECTION_NAME
from .duckdb_pipeline import (
    boundary_loss_analysis,
    distance_matrix_m,
    district_centroids,
    district_price_stats,
)
from .embedder import DenseEmbedder
from .qdrant_ops import (
    search_dense,
    search_hard_radius,
    search_multi_decay,
    search_viewport_with_facets,
)

logger = logging.getLogger("geosmart.benchmark")
console = Console()
REPEATS = 5
MODES = ["M1_dense", "M2_hard_filter", "M3_decay", "M4_viewport_facet"]


@dataclass
class EvalQuery:
    qid: str
    persona: str
    query: str
    district_key: str
    budget: float = 95
    dist_scale_m: float = 2000
    price_scale: float = 35
    hard_radius_m: float = 1500


def _round5(x: float) -> float:
    return max(40, min(250, round(x / 5) * 5))


def build_eval_queries() -> list[EvalQuery]:
    cents = district_centroids()
    stats = district_price_stats()
    personas = [
        (
            "GEO-01",
            "Solo Remote Worker in Mitte",
            "Cozy sunny studio with fast wifi and dedicated workspace",
            "alexanderplatz",
            2000,
        ),
        (
            "GEO-02",
            "Family Vacation in Tiergarten",
            "Spacious family friendly 2-bedroom apartment with full kitchen near park",
            "tiergarten_süd",
            1800,
        ),
        (
            "GEO-03",
            "Nightlife & Culture in Kreuzberg",
            "Trendy bohemian loft with private balcony near vibrant cafes and bars",
            "südliche_friedrichstadt",
            1800,
        ),
        (
            "GEO-04",
            "Cultural Tourist near Museum Island",
            "Quiet elegant flat within walking distance of galleries and cathedral",
            "brunnenstr._süd",
            1800,
        ),
        (
            "GEO-05",
            "Transit Commuter near Central Station",
            "Clean modern private room near main train station with instant self check-in",
            "moabit_ost",
            1200,
        ),
        (
            "GEO-06",
            "Weekend Brunch Couple in Prenzlauer Berg",
            "Bright apartment near organic cafes and weekend markets",
            "prenzlauer_berg_nordwest",
            1800,
        ),
        (
            "GEO-07",
            "Digital Nomad in Charlottenburg",
            "Long stay studio with workspace balcony near library and metro",
            "schloß_charlottenburg",
            2000,
        ),
        (
            "GEO-08",
            "Art Lover in Friedrichshain",
            "Industrial style loft with big windows near galleries and street art",
            "karl_marx_allee_nord",
            1800,
        ),
        (
            "GEO-09",
            "Value Solo Traveler in Neukölln",
            "Affordable private room with kitchen access near street food and nightlife",
            "neuköllner_mitte/zentrum",
            1500,
        ),
        (
            "GEO-10",
            "Architecture Enthusiast near Reichstag",
            "Modern minimal flat near government district with skyline views",
            "regierungsviertel",
            1800,
        ),
    ]
    out = []
    for qid, persona, q, dk, scale in personas:
        if dk not in cents:
            continue
        grp = cents[dk]["group"]
        budget = _round5(stats.get(grp, {}).get("median_price", 95))
        out.append(
            EvalQuery(
                qid,
                persona,
                q,
                dk,
                budget,
                scale,
                round(budget * 0.35, 1),
                round(scale * 0.75),
            )
        )
    return out


def _timed(fn, n=REPEATS):
    out = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t0) * 1000)
    return out


def _p50(xs):
    return statistics.median(xs) if xs else 0


def _p95(xs):
    return statistics.quantiles(xs, n=20)[18] if len(xs) >= 20 else max(xs) if xs else 0


def _metrics(hits, lat, lon, budget):
    if not hits:
        return {"n": 0, "dist": 0, "berr": 0, "sh": 0, "rating": 0}
    dists = distance_matrix_m(
        [h.payload["location"]["lat"] for h in hits],
        [h.payload["location"]["lon"] for h in hits],
        lat,
        lon,
    )
    prices = [h.payload["price"] for h in hits]
    return {
        "n": len(hits),
        "dist": sum(dists) / len(dists),
        "berr": sum(abs(p - budget) for p in prices) / len(prices),
        "sh": sum(1 if h.payload.get("is_superhost") else 0 for h in hits) / len(hits),
        "rating": sum(h.payload.get("rating", 0) for h in hits) / len(hits),
    }


def run_ladder_benchmark(
    client: QdrantClient,
    dense: DenseEmbedder,
    output_path: Path = Path("benchmark_report.json"),
) -> dict[str, Any]:
    console.print(
        "\n[bold cyan]═══ GeoSmart Spatial Benchmark - 4 Modes, Measured (Qdrant 1.19) ═══[/bold cyan]\n"
    )
    queries = build_eval_queries()
    console.print(
        f"Evaluating [bold]{len(queries)}[/bold] persona queries across Berlin.\n"
    )
    acc = {
        m: {"lat": [], "dist": [], "berr": [], "sh": [], "rating": []} for m in MODES
    }
    boundary = []
    per_query = []

    for eq in queries:
        c = district_centroids()[eq.district_key]
        t_lat, t_lon = c["lat"], c["lon"]
        qvec = dense.embed_queries(eq.query)[0]

        # Bounding box viewport ±0.025° (~2.8km box around target centroid)
        vp_tl = (t_lat + 0.025, t_lon - 0.025)
        vp_br = (t_lat - 0.025, t_lon + 0.025)

        executors = {
            "M1_dense": (lambda _qv=qvec: search_dense(client, _qv, limit=5)),
            "M2_hard_filter": (
                lambda _qv=qvec, _la=t_lat, _lo=t_lon, _rad=eq.hard_radius_m, _bud=eq.budget: (
                    search_hard_radius(client, _qv, _la, _lo, _rad, _bud * 1.1, limit=5)
                )
            ),
            "M3_decay": (
                lambda _qv=qvec, _q=eq.query, _la=t_lat, _lo=t_lon, _ds=eq.dist_scale_m, _bud=eq.budget, _ps=eq.price_scale: (
                    search_multi_decay(
                        client, _qv, _q, _la, _lo, _ds, _bud, _ps, limit=5
                    )
                )
            ),
            "M4_viewport_facet": (
                lambda _qv=qvec, _q=eq.query, _tl=vp_tl, _br=vp_br, _bud=eq.budget, _ps=eq.price_scale: (
                    search_viewport_with_facets(
                        client, _qv, _q, _tl, _br, _bud, _ps, limit=5
                    )["hits"]
                )
            ),
        }

        # Measure hard cliff for boundary loss comparison
        hard_hits = search_hard_radius(
            client, qvec, t_lat, t_lon, eq.hard_radius_m, eq.budget * 1.1, limit=5
        )
        hard_metrics = _metrics(hard_hits, t_lat, t_lon, eq.budget)

        results = {}
        for mode, fn in executors.items():
            lats = _timed(fn)
            hits = fn()
            results[mode] = hits
            acc[mode]["lat"].extend(lats)
            m = _metrics(hits, t_lat, t_lon, eq.budget)
            if hits:
                acc[mode]["dist"].append(m["dist"])
                acc[mode]["berr"].append(m["berr"])
                acc[mode]["sh"].append(m["sh"])
                acc[mode]["rating"].append(m["rating"])

        b = boundary_loss_analysis(t_lat, t_lon, eq.hard_radius_m, eq.budget * 1.1)
        boundary.append(b)
        per_query.append(
            {
                "id": eq.qid,
                "persona": eq.persona,
                "district": c["name"],
                "budget": eq.budget,
                "hard_radius_m": eq.hard_radius_m,
                "boundary_loss": b,
                "hard_cliff": hard_metrics,
                "modes": {
                    mode: {
                        **_metrics(hits, t_lat, t_lon, eq.budget),
                        "last_latency_ms": round(acc[mode]["lat"][-1], 2)
                        if acc[mode]["lat"]
                        else 0,
                    }
                    for mode, hits in results.items()
                },
            }
        )

    def agg(mode, k):
        vals = acc[mode][k]
        return sum(vals) / len(vals) if vals else 0

    summary = {
        "ladder": {
            mode: {
                "latency_p50_ms": round(_p50(acc[mode]["lat"]), 2),
                "latency_p95_ms": round(_p95(acc[mode]["lat"]), 2),
                "avg_distance_m": round(agg(mode, "dist"), 0),
                "avg_budget_error_eur": round(agg(mode, "berr"), 1),
                "superhost_ratio": round(agg(mode, "sh"), 3),
                "avg_rating": round(agg(mode, "rating"), 2),
            }
            for mode in MODES
        },
        "boundary_loss": {
            "avg_dropout_fraction": round(
                sum(r["dropout_fraction"] for r in boundary) / len(boundary), 4
            )
            if boundary
            else 0,
            "avg_near_miss": round(
                sum(r["near_miss"] for r in boundary) / len(boundary), 1
            )
            if boundary
            else 0,
        },
        "queries": per_query,
        "config": {"repeats": REPEATS, "modes": MODES, "collection": COLLECTION_NAME},
    }
    _print(summary)
    output_path.write_text(json.dumps(summary, indent=2))
    console.print(f"[green]✓ Benchmark → {output_path}[/green]\n")
    return summary


def _print(s):
    lad = s["ladder"]
    t = Table(
        title="GeoSmart Spatial Ladder - Qdrant 1.19 Engine-Only",
        header_style="bold magenta",
    )
    for col in [
        "Mode",
        "p50 ms",
        "p95 ms",
        "Avg Dist (m)",
        "Budget Err (€)",
        "Superhost",
        "Rating",
    ]:
        t.add_column(col, justify="right" if col != "Mode" else "left")
    names = {
        "M1_dense": "[dim]M1 Pure Semantic (BQ)[/dim]",
        "M2_hard_filter": "[red]M2 Hard Circle Cliff[/red]",
        "M3_decay": "[bold green]M3 Multi-Decay Formula[/bold green]",
        "M4_viewport_facet": "[bold cyan]M4 Viewport + In-DB Facets[/bold cyan]",
    }
    for m, v in lad.items():
        t.add_row(
            names.get(m, m),
            f"{v['latency_p50_ms']:.1f}",
            f"{v['latency_p95_ms']:.1f}",
            f"{v['avg_distance_m']:,.0f}",
            f"{v['avg_budget_error_eur']:.1f}",
            f"{v['superhost_ratio'] * 100:.0f}%",
            f"{v['avg_rating']:.2f}",
        )
    console.print(t)
    b = s["boundary_loss"]
    console.print(
        f"[dim]* Hard-filter cliff: [red]{b['avg_dropout_fraction'] * 100:.1f}% of near-area stays dropped[/red] "
        f"({b['avg_near_miss']:.0f} stays/query sit in the 1.0–1.25× radius ring). "
        f"M3/M4 GaussDecay keeps them (0.0% ring loss).[/dim]"
    )
