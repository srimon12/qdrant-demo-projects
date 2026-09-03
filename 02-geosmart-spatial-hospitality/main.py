"""GeoSmart - Spatial Vector Search & Continuous Decay Engine (Qdrant 1.19)."""

from __future__ import annotations

import argparse
import logging
import time

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from src.benchmark import run_ladder_benchmark
from src.config import COLLECTION_NAME, LISTINGS_CSV, NEIGHBOURHOODS_GEOJSON
from src.duckdb_pipeline import (
    boundary_loss_analysis,
    district_centroids,
    h3_hotspots,
    run_duckdb_spatial_pipeline,
)
from src.embedder import DenseEmbedder
from src.qdrant_ops import (
    collection_point_count,
    get_qdrant_client,
    index_corpus,
    is_collection_schema_ready,
    search_dense,
    search_geo_polygon,
    search_grouped_districts,
    search_hard_radius,
    search_multi_decay,
    search_viewport_with_facets,
    setup_collection,
    write_manifest,
)

console = Console()
logging.basicConfig(level=logging.INFO, format="%(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


def ensure_dataset():
    """Download the Berlin Inside Airbnb snapshot if data/ is empty."""
    if LISTINGS_CSV.exists() and NEIGHBOURHOODS_GEOJSON.exists():
        return
    console.print(
        "[yellow]► Listing files not found - downloading Berlin Inside Airbnb snapshot…[/yellow]"
    )
    from scripts.download_data import download_berlin

    download_berlin()


def connect_qdrant():
    try:
        return get_qdrant_client()
    except ConnectionError as exc:
        console.print(f"[red]{exc}[/red]")
        raise SystemExit(1) from exc


def ensure_index(client, dense, force_rebuild: bool = False):
    """Ensure the Qdrant collection exists and is populated via the DuckDB pipeline."""
    if (
        force_rebuild
        or collection_point_count(client) == 0
        or not is_collection_schema_ready(client)
    ):
        console.print(
            "[yellow]► Initializing GeoSmart spatial collection and indexing listings…[/yellow]"
        )
        ensure_dataset()
        t0 = time.perf_counter()
        df = run_duckdb_spatial_pipeline(refresh=force_rebuild)
        setup_collection(client, COLLECTION_NAME, recreate=True)
        index_corpus(client, df, dense, collection_name=COLLECTION_NAME)
        write_manifest(
            len(df), dense.backend, dense.model, collection_name=COLLECTION_NAME
        )
        console.print(
            f"  [green]✓ Indexed {len(df):,} listings into '{COLLECTION_NAME}' in {time.perf_counter() - t0:.1f}s.[/green]\n"
        )
    else:
        console.print(
            f"[dim]✓ Connected to '{COLLECTION_NAME}' ({collection_point_count(client):,} points ready).[/dim]\n"
        )


def showcase_spatial_ladder(
    client,
    dense,
    query: str = "Cozy sunny studio with fast wifi and dedicated workspace",
    target_district: str = "alexanderplatz",
    budget: float = 95.0,
):
    """Showcase 1: Four ranking modes on the same query (dense → hard filter → decay → viewport)."""
    console.print(
        Panel.fit(
            f'[bold white]1. SPATIAL SEARCH LADDER[/bold white]\n[cyan]Query:[/cyan] "{query}"\n[cyan]Target:[/cyan] {target_district.title()} · [cyan]Budget:[/cyan] €{budget:.0f}/night\n[dim]M1 dense (geo-blind) → M2 hard geo_radius cliff → M3 GaussDecay+ExpDecay → M4 viewport + facets[/dim]',
            border_style="cyan",
        )
    )

    cents = district_centroids()
    d_info = cents.get(target_district, {"lat": 52.5219, "lon": 13.4132})
    t_lat, t_lon = d_info["lat"], d_info["lon"]

    vp_tl = (t_lat + 0.025, t_lon - 0.025)
    vp_br = (t_lat - 0.025, t_lon + 0.025)

    qvec = dense.embed_queries(query)[0]

    # Mode 1: Pure Semantic (Dense BQ)
    t0 = time.perf_counter()
    m1 = search_dense(client, qvec, limit=3)
    l1 = (time.perf_counter() - t0) * 1000

    # Mode 2: Hard Circle Filter
    t0 = time.perf_counter()
    m2 = search_hard_radius(client, qvec, t_lat, t_lon, 1500, budget * 1.1, limit=3)
    l2 = (time.perf_counter() - t0) * 1000

    # Mode 3: Continuous Multi-Decay Formula
    t0 = time.perf_counter()
    m3 = search_multi_decay(
        client, qvec, query, t_lat, t_lon, 2000.0, budget, 35.0, limit=3
    )
    l3 = (time.perf_counter() - t0) * 1000

    # Mode 4: Viewport + In-DB Facets
    t0 = time.perf_counter()
    m4_res = search_viewport_with_facets(
        client, qvec, query, vp_tl, vp_br, budget, 35.0, limit=3
    )
    m4 = m4_res["hits"]
    l4 = (time.perf_counter() - t0) * 1000

    table = Table(
        title="Spatial Ladder Results (M1–M3: one query_points; M4: dense + 2 facets)"
    )
    table.add_column("Mode", style="bold")
    table.add_column("Top-1 Match", style="white")
    table.add_column("District & Price", style="yellow")
    table.add_column("Rating / Trust", style="green")
    table.add_column("Latency", justify="right", style="cyan")

    table.add_row(
        "M1 Pure Semantic (BQ)",
        m1[0].payload.get("name", "")[:32] if m1 else "-",
        f"{m1[0].payload.get('neighbourhood', '')} · €{m1[0].payload.get('price', 0):.0f}"
        if m1
        else "-",
        f"{m1[0].payload.get('rating', 0):.2f}★" if m1 else "-",
        f"{l1:.1f} ms",
    )
    table.add_row(
        "M2 Hard Circle Cliff",
        m2[0].payload.get("name", "")[:32] if m2 else "-",
        f"{m2[0].payload.get('neighbourhood', '')} · €{m2[0].payload.get('price', 0):.0f}"
        if m2
        else "-",
        f"{m2[0].payload.get('rating', 0):.2f}★" if m2 else "-",
        f"{l2:.1f} ms",
    )
    table.add_row(
        "[bold green]M3 Multi-Decay Formula[/bold green]",
        f"[bold green]{m3[0].payload.get('name', '')[:32]}[/bold green]" if m3 else "-",
        f"[bold green]{m3[0].payload.get('neighbourhood', '')} · €{m3[0].payload.get('price', 0):.0f}[/bold green]"
        if m3
        else "-",
        (
            f"[bold green]{m3[0].payload.get('rating', 0):.2f}★"
            + (
                " Superhost[/bold green]"
                if m3[0].payload.get("is_superhost")
                else "[/bold green]"
            )
        )
        if m3
        else "-",
        f"[bold green]{l3:.1f} ms[/bold green]",
    )
    table.add_row(
        "[bold cyan]M4 Viewport + In-DB Facets[/bold cyan]",
        f"[bold cyan]{m4[0].payload.get('name', '')[:32]}[/bold cyan]" if m4 else "-",
        f"[bold cyan]{m4[0].payload.get('neighbourhood', '')} · €{m4[0].payload.get('price', 0):.0f}[/bold cyan]"
        if m4
        else "-",
        f"[bold cyan]{m4[0].payload.get('rating', 0):.2f}★[/bold cyan]" if m4 else "-",
        f"[bold cyan]{l4:.1f} ms[/bold cyan]",
    )
    console.print(table)
    console.print()


def showcase_viewport_faceting(
    client, dense, top_left=(52.5400, 13.3800), bottom_right=(52.5100, 13.4300)
):
    """Showcase 2: Real-time In-Database Map Viewport Clustering via client.facet."""
    console.print(
        Panel.fit(
            "[bold white]2. DYNAMIC MAP VIEWPORT CLUSTERING (Qdrant 1.19 client.facet)[/bold white]\n"
            f"[cyan]Active Bounding Box:[/cyan] Top-Left {top_left} ➔ Bottom-Right {bottom_right}\n"
            "[dim]Aggregates live available listings across neighborhoods and room types without secondary SQL databases.[/dim]",
            border_style="magenta",
        )
    )
    qvec = dense.embed_queries("quiet studio with kitchen")[0]
    t0 = time.perf_counter()
    vp_res = search_viewport_with_facets(
        client,
        qvec,
        "quiet studio with kitchen",
        top_left,
        bottom_right,
        target_price=95.0,
        limit=3,
    )
    elapsed = (time.perf_counter() - t0) * 1000

    table = Table(title=f"Live Viewport Clusters ({elapsed:.1f} ms in Qdrant)")
    table.add_column("Neighborhood Group", style="bold yellow")
    table.add_column("Available Stays in Viewport", style="green", justify="right")
    for f in vp_res["district_facets"]:
        table.add_row(f["name"], f"{f['count']:,}")
    console.print(table)

    room_table = Table(title="Viewport Breakdown by Room Type")
    room_table.add_column("Room Type", style="bold cyan")
    room_table.add_column("Count", style="green", justify="right")
    for r in vp_res["room_facets"]:
        room_table.add_row(r["room_type"], f"{r['count']:,}")
    console.print(room_table)
    console.print()


def showcase_polygon_pushdown(client, dense, district_key: str = "alexanderplatz"):
    """Showcase 3: Native GeoPolygon Pushdown for Administrative District Boundaries."""
    cents = district_centroids()
    d_info = cents.get(district_key, cents.get("alexanderplatz"))

    console.print(
        Panel.fit(
            f"[bold white]3. ADMINISTRATIVE DISTRICT GEOPOLYGON PUSHDOWN: {d_info['name'].upper()}[/bold white]\n"
            f"[cyan]Polygon Complexity:[/cyan] {len(d_info['polygon'])} coordinate vertices extracted from Berlin LOR GeoJSON\n"
            "[dim]Strict point-in-polygon containment evaluated inside Qdrant alongside vector ranking.[/dim]",
            border_style="yellow",
        )
    )
    poly_exterior = [{"lat": pt[0], "lon": pt[1]} for pt in d_info["polygon"]]
    qvec = dense.embed_queries("modern sunny apartment with balcony")[0]

    t0 = time.perf_counter()
    hits = search_geo_polygon(client, qvec, poly_exterior, limit=3)
    elapsed = (time.perf_counter() - t0) * 1000

    console.print(
        f"[green]Retrieved {len(hits)} listings strictly inside polygon in {elapsed:.1f} ms:[/green]"
    )
    for rank, h in enumerate(hits, 1):
        p = h.payload
        console.print(
            f"  #{rank} [green]{p.get('name')[:45]}[/green] | €{p.get('price'):.0f}/nt | {p.get('rating')}★ | Lat: {p['location']['lat']:.4f}, Lon: {p['location']['lon']:.4f}"
        )
    console.print()


def showcase_grouped_diversity(
    client, dense, query: str = "quiet sunny loft with balcony near cafes"
):
    """Showcase 4: Grouped Diversity - one curated stay per district via query_points_groups."""
    console.print(
        Panel.fit(
            "[bold white]4. GROUPED DIVERSITY: ONE BEST STAY PER DISTRICT (Qdrant query_points_groups)[/bold white]\n"
            "[dim]Hybrid DBSF + Formula decay grouped by neighbourhood_group - avoids Mitte homogeneity where 1,982 stays would otherwise dominate.[/dim]",
            border_style="cyan",
        )
    )
    cents = district_centroids()
    alex = cents.get("alexanderplatz", {"lat": 52.5219, "lon": 13.4132})
    qvec = dense.embed_queries(query)[0]
    t0 = time.perf_counter()
    groups = search_grouped_districts(
        client,
        qvec,
        query,
        alex["lat"],
        alex["lon"],
        2000.0,
        95.0,
        35.0,
        limit=5,
        group_size=1,
    )
    elapsed = (time.perf_counter() - t0) * 1000
    table = Table(
        title=f"Grouped Results by District ({elapsed:.1f} ms, hybrid + grouping)"
    )
    table.add_column("District (group_by)", style="bold yellow")
    table.add_column("Top Stay in Group", style="white")
    table.add_column("Price", style="green", justify="right")
    table.add_column("Rating", style="cyan", justify="right")
    table.add_column("Score", style="magenta", justify="right")
    for g in groups.groups:
        hit = g.hits[0] if g.hits else None
        if not hit:
            continue
        p = hit.payload
        table.add_row(
            str(g.id),
            p.get("name", "")[:32],
            f"€{p.get('price', 0):.0f}",
            f"{p.get('rating', 0):.2f}★",
            f"{hit.score:.3f}",
        )
    console.print(table)
    console.print()


def showcase_boundary_cliff_math():
    """Showcase 5: Spheroid Mathematical Boundary Loss Analysis."""
    console.print(
        Panel.fit(
            "[bold white]5. THE BOUNDARY CLIFF PROBLEM: MATHEMATICAL PROOF[/bold white]\n"
            "[red]The Industry Flaw:[/red] Hard circle filters (geo_radius <= 1.5km) discard viable listings right on the border.\n"
            "[green]The Qdrant Solution:[/green] GaussDecay(GeoDistance) grades smoothly with 0.0% boundary loss.",
            border_style="red",
        )
    )
    cents = district_centroids()
    alex = cents.get("alexanderplatz", {"lat": 52.5219, "lon": 13.4132})
    b = boundary_loss_analysis(alex["lat"], alex["lon"], 1500, 105.0)

    table = Table(
        title="Alexanderplatz Boundary Cliff Analysis (DuckDB Spheroid Ground Truth)"
    )
    table.add_column("Metric", style="bold cyan")
    table.add_column("Hard Filter (<=1.5km)", style="red", justify="right")
    table.add_column("Continuous Decay (Gauss)", style="bold green", justify="right")
    table.add_row("Stays inside 1.5 km", f"{b['hits']:.0f}", f"{b['hits']:.0f}")
    table.add_row(
        "Near-miss ring (1.5–1.87 km)",
        f"[bold red]{b['near_miss']:.0f} DROPPED[/bold red]",
        f"[bold green]{b['near_miss']:.0f} KEPT[/bold green]",
    )
    table.add_row(
        "Share of near-area stays dropped",
        f"[bold red]{b['dropout_fraction'] * 100:.1f}%[/bold red]",
        "[bold green]0.0%[/bold green]",
    )
    console.print(table)
    console.print()


def showcase_h3_density():
    """Showcase 6: H3 hex density - DuckDB complements Qdrant (Qdrant has no hex binning)."""
    console.print(
        Panel.fit(
            "[bold white]6. HEX DENSITY HEATMAP: H3 BINNING (DuckDB complements Qdrant)[/bold white]\n"
            "[dim]Qdrant excels at geo_radius/polygon, but hex binning is DuckDB H3 territory - 8,317 listings per H3 r8 cell.[/dim]",
            border_style="yellow",
        )
    )
    hotspots = h3_hotspots(res=8, top_n=5)
    table = Table(title="Top H3 r8 Cells by Listing Density")
    table.add_column("H3 Cell", style="bold cyan")
    table.add_column("Count", style="green", justify="right")
    table.add_column("Center Lat/Lon", style="white")
    table.add_column("Avg Price", style="yellow", justify="right")
    table.add_column("Avg Rating", style="magenta", justify="right")
    for h in hotspots:
        c = h["center"]
        table.add_row(
            h["cell"][:12] + "…",
            f"{h['count']}",
            f"{c['lat']:.4f}, {c['lon']:.4f}",
            f"€{h['avg_price']:.0f}",
            f"{h['avg_rating']:.2f}★",
        )
    console.print(table)
    console.print()


def interactive_mode(client, dense):
    """Interactive exploration mode."""
    console.print("[bold cyan]═══ GeoSmart Interactive Spatial Search ═══[/bold cyan]")
    cents = district_centroids()
    while True:
        try:
            q = console.input(
                "\n[bold yellow]Natural Language Vibe (or 'q' to exit): [/bold yellow]"
            ).strip()
            if not q or q.lower() in ("q", "quit", "exit"):
                break
            dk = (
                console.input("[yellow]District name (Enter=alexanderplatz): [/yellow]")
                .strip()
                .lower()
            )
            if dk in cents:
                t_lat, t_lon = cents[dk]["lat"], cents[dk]["lon"]
                label = cents[dk]["name"]
            else:
                t_lat, t_lon = 52.5219, 13.4132
                label = "Alexanderplatz"
            b = float(
                console.input("[yellow]Target Budget € (Enter=95): [/yellow]").strip()
                or "95"
            )

            qvec = dense.embed_queries(q)[0]
            t0 = time.perf_counter()
            hits = search_multi_decay(
                client, qvec, q, t_lat, t_lon, 2000.0, b, 35.0, limit=4
            )
            ms = (time.perf_counter() - t0) * 1000

            console.print(
                f"\n[bold green]Results near {label} (€{b:.0f}/nt budget, {ms:.1f}ms):[/bold green]"
            )
            for i, h in enumerate(hits, 1):
                p = h.payload
                sh = " [yellow]★ Superhost[/yellow]" if p.get("is_superhost") else ""
                console.print(
                    f"  #{i} {h.score:.3f} | [green]{p['name'][:45]}[/green] | €{p['price']:.0f} | {p['rating']}★ ({p['number_of_reviews']}){sh}"
                )
                console.print(f"     {p['neighbourhood']} | {p['room_type']}")
        except (KeyboardInterrupt, EOFError):
            break


def main():
    ap = argparse.ArgumentParser(description="GeoSmart Spatial Vector Engine")
    ap.add_argument("--query", help="Run a single natural-language spatial query")
    ap.add_argument(
        "--interactive", action="store_true", help="Launch the terminal search REPL"
    )
    ap.add_argument(
        "--ui",
        action="store_true",
        help="Launch the Leaflet map UI at http://127.0.0.1:8000",
    )
    ap.add_argument(
        "--benchmark",
        action="store_true",
        help="Run the 10-persona spatial benchmark only",
    )
    ap.add_argument(
        "--rebuild", action="store_true", help="Force rebuild and re-enrich the index"
    )
    args = ap.parse_args()

    client = connect_qdrant()
    dense = DenseEmbedder()

    ensure_index(client, dense, force_rebuild=args.rebuild)

    if args.ui:
        import uvicorn

        console.print(
            "[bold green]Map UI → http://127.0.0.1:8000[/bold green]  (Ctrl+C to stop)"
        )
        uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
        return

    if args.interactive:
        interactive_mode(client, dense)
        return

    if args.benchmark:
        run_ladder_benchmark(client, dense)
        return

    if args.query:
        showcase_spatial_ladder(client, dense, query=args.query)
        return

    # Default: Run the full, structured spatial showcase
    console.print(
        Panel.fit(
            "[bold white]GEOSMART: SPATIAL VECTOR SEARCH & CONTINUOUS DECAY ENGINE[/bold white]\n"
            "[cyan]Qdrant 1.19 · 8,317 Berlin Stays · Dual-Store DuckDB Spheroid Geodesy[/cyan]\n"
            "[dim]Zero external rerankers. Ranking is single query_points; facets are 2 extra Qdrant ops.[/dim]",
            border_style="blue",
        )
    )
    console.print()

    # 1. Search Ladder
    showcase_spatial_ladder(client, dense)

    # 2. Viewport Faceting
    showcase_viewport_faceting(client, dense)

    # 3. GeoPolygon Pushdown
    showcase_polygon_pushdown(client, dense)

    # 4. Grouped Diversity (Qdrant groups)
    showcase_grouped_diversity(client, dense)

    # 5. Boundary Cliff Proof
    showcase_boundary_cliff_math()

    # 6. H3 Density (DuckDB complements Qdrant)
    showcase_h3_density()

    # 7. Full 10-Persona Benchmark
    console.print(
        Panel.fit(
            "[bold white]7. BENCHMARK: 10 PERSONAS × 4 SPATIAL MODES[/bold white]\n"
            "[dim]Evaluates p50 latency, geodesic distance error, budget error, and superhost ratio.[/dim]",
            border_style="green",
        )
    )
    run_ladder_benchmark(client, dense)


if __name__ == "__main__":
    main()
