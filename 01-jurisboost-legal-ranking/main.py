"""JurisBoost - Indian Legal Precedent Retrieval Engine (Qdrant 1.19)."""

from __future__ import annotations

import argparse
import time

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from src.config import COLLECTION_NAME
from src.duckdb_pipeline import load_and_aggregate_precedents, resolve_dataset_filepath
from src.embedder import ColbertLateInteractionEmbedder, DensePrecedentEmbedder
from src.qdrant_ops import (
    collection_point_count,
    facet_statutory_landscape,
    get_qdrant_client,
    index_precedents,
    is_collection_schema_ready,
    search_dense,
    search_formula,
    search_grouped_statutes,
    search_hybrid,
    search_universal_pipeline,
    setup_collection,
)
from src.sentence_isolator import isolate_ratio_colbert

console = Console()


def ensure_dataset():
    """Download NyayaRAG if the JSON is not on disk yet."""
    path = resolve_dataset_filepath(None)
    if path.exists():
        return path
    console.print(
        "[yellow]► NyayaRAG dataset not found - downloading from Hugging Face (~35 MB)…[/yellow]"
    )
    from scripts.download_data import download_nyayarag_dataset

    return download_nyayarag_dataset()


def connect_qdrant():
    try:
        return get_qdrant_client()
    except ConnectionError as exc:
        console.print(f"[red]{exc}[/red]")
        raise SystemExit(1) from exc


def ensure_index(client, dense, colbert, force_rebuild: bool = False):
    """Ensure the Qdrant collection exists and is populated with NyayaRAG precedents."""
    if (
        force_rebuild
        or collection_point_count(client) == 0
        or not is_collection_schema_ready(client)
    ):
        console.print(
            "[yellow]► Initializing JurisBoost collection and indexing NyayaRAG precedents…[/yellow]"
        )
        t0 = time.perf_counter()
        records = load_and_aggregate_precedents(
            ensure_dataset(), source_limit=None, domain_filter="criminal"
        )
        setup_collection(client, COLLECTION_NAME, recreate=True)
        index_precedents(
            client, records, dense, colbert, collection_name=COLLECTION_NAME
        )
        console.print(
            f"  [green]✓ Indexed {len(records):,} legal precedents into '{COLLECTION_NAME}' in {time.perf_counter() - t0:.1f}s.[/green]\n"
        )
    else:
        console.print(
            f"[dim]✓ Connected to '{COLLECTION_NAME}' ({collection_point_count(client):,} precedents ready).[/dim]\n"
        )


def showcase_search_ladder(
    client,
    dense,
    colbert,
    query: str = "Can anticipatory bail under Section 482 BNSS continue without a fixed time limit, as under Section 438 CrPC?",
):
    """Showcase 1: The 4-Stage Legal Retrieval Progression Ladder (single query_points per mode)."""
    console.print(
        Panel.fit(
            f"[bold white]1. SEARCH LADDER: HOW RETRIEVAL ACCURACY SCALES[/bold white]\n"
            f'[cyan]Target Query:[/cyan] "{query}"\n'
            "[dim]M1 Dense BQ → M2 Hybrid RRF → M3 Hybrid + Formula → M4 RRF → ColBERT → Formula. One query_points call per mode.[/dim]",
            border_style="cyan",
        )
    )

    qvec = dense.embed_legal_queries(query)[0].tolist()
    colvec = colbert.embed_legal_queries(query)[0].tolist()

    t0 = time.perf_counter()
    m1 = search_dense(client, qvec, limit=3)
    l1 = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    m2 = search_hybrid(client, qvec, query, limit=3)
    l2 = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    m3 = search_formula(client, qvec, query, limit=3)
    l3 = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    m4 = search_universal_pipeline(client, qvec, colvec, query, limit=3)
    l4 = (time.perf_counter() - t0) * 1000

    table = Table(title="Qdrant Search Ladder (single query_points per mode)")
    table.add_column("Mode", style="bold")
    table.add_column("Top-1 Precedent", style="white")
    table.add_column("Score", style="yellow")
    table.add_column("Latency", justify="right", style="cyan")

    table.add_row(
        "M1 Dense (BQ)",
        m1[0].payload.get("title", "")[:45] if m1 else "-",
        f"{m1[0].score:.4f}" if m1 else "-",
        f"{l1:.1f} ms",
    )
    table.add_row(
        "M2 Hybrid RRF",
        m2[0].payload.get("title", "")[:45] if m2 else "-",
        f"{m2[0].score:.4f}" if m2 else "-",
        f"{l2:.1f} ms",
    )
    table.add_row(
        "M3 Hybrid → Formula",
        m3[0].payload.get("title", "")[:45] if m3 else "-",
        f"{m3[0].score:.4f}" if m3 else "-",
        f"{l3:.1f} ms",
    )
    table.add_row(
        "[bold green]M4 RRF → ColBERT → Formula[/bold green]",
        f"[bold green]{m4[0].payload.get('title', '')[:45]}[/bold green]"
        if m4
        else "-",
        f"[bold green]{m4[0].score:.4f}[/bold green]" if m4 else "-",
        f"[bold green]{l4:.1f} ms[/bold green]",
    )
    console.print(table)

    if m4:
        p = m4[0].payload
        text = p.get("ratio") or p.get("text", "")
        if text:
            selected_sents, stats = isolate_ratio_colbert(text, colvec, colbert)
            selected_text = " ".join([s[0] for s in selected_sents])
            cut_pct = (
                1.0 - (stats["isolated_words"] / max(1, stats["raw_words"]))
            ) * 100
            console.print(
                "\n[bold green]Reference-Fidelity ColBERT MAX_SIM Isolation:[/bold green]"
            )
            console.print(f"  [bold]Precedent:[/bold] {p.get('title')}")
            console.print(f'  [bold]Holding:[/bold] "{selected_text[:150]}…"')
            console.print(
                f"  [cyan]Compression:[/cyan] {stats['raw_words']} → {stats['isolated_words']} words "
                f"({cut_pct:.1f}% cut, ColBERT MaxSim locally - no LLM API)\n"
            )


def showcase_analytics(client, dense):
    """Showcase 3: In-database analytics - faceting + grouping (no SQL)."""
    console.print(
        Panel.fit(
            "[bold white]2. IN-DATABASE ANALYTICS (Facet + Grouping)[/bold white]\n"
            "[dim]facet + grouping - both server-side via PINNED keyword indexes, no secondary DB.[/dim]",
            border_style="green",
        )
    )
    # Facet - increase limit to 30, then filter noisy section:* where act extraction missed window
    t0 = time.perf_counter()
    facets = facet_statutory_landscape(client, key="legal_references", limit=30)
    f_ms = (time.perf_counter() - t0) * 1000

    desc_map = {
        "ipc:302": "Murder → 103 BNS",
        "ipc:34": "Common intention → 3(5) BNS",
        "crpc:482": "Quashing → 528 BNSS",
        "crpc:438": "Anticipatory bail → 482 BNSS",
        "ipc:304": "Culpable homicide → 105 BNS",
        "ipc:304b": "Dowry death → 80 BNS",
        "crpc:41a": "Notice before arrest → 35(3) BNSS",
        "crpc:154": "FIR → 173 BNSS",
        "iea:27": "Discovery → 23(2) BSA",
        "iea:113a": "Abetment presumption → 117 BSA",
        "ipc:420": "Cheating → 318 BNS",
        "ipc:498a": "Matrimonial cruelty → 85 BNS",
    }
    ft = Table(title=f"Facets - legal_references ({f_ms:.1f} ms, top canonical)")
    ft.add_column("Section", style="bold cyan")
    ft.add_column("Count", justify="right", style="green")
    ft.add_column("Desc", style="dim")
    clean = [
        f
        for f in facets
        if ":" in f.value
        and f.value.split(":")[0] in ("ipc", "crpc", "bns", "bnss", "iea", "bsa")
    ]
    # If still polluted, show honest note
    for f in clean[:10]:
        ft.add_row(f.value, str(f.count), desc_map.get(f.value, ""))
    console.print(ft)
    if len([f for f in facets if f.value.startswith("section:")]) > 5:
        console.print(
            "[dim]Note: raw facet includes noisy section:* keys (e.g. section:3) where the 80-char act window missed - filtered to canonical act:section above.[/dim]"
        )

    # Grouping - multi-statute query that hits diverse provisions (murder/theft/dowry/bail/quashing)
    g_query = "murder and theft and dowry death and anticipatory bail and quashing"
    qvec = dense.embed_legal_queries(g_query)[0].tolist()
    t0 = time.perf_counter()
    grouped = search_grouped_statutes(
        client, qvec, group_by_field="legal_references", group_size=2, limit=6
    )
    g_ms = (time.perf_counter() - t0) * 1000

    gt = Table(
        title=f'Groups - query_points_groups by legal_references ({g_ms:.1f} ms)\nQuery: "{g_query}"'
    )
    gt.add_column("Statute Group", style="bold yellow")
    gt.add_column("Top Hits in Group", style="green")
    shown = 0
    for g in grouped.groups:
        if ":" not in str(g.id) or str(g.id).startswith("section:"):
            continue
        hits_str = "\n".join(
            [f"• {h.payload.get('title')[:46]} (score={h.score:.4f})" for h in g.hits]
        )
        gt.add_row(str(g.id), hits_str)
        shown += 1
        if shown >= 4:
            break
    console.print(gt)
    console.print()


def main():
    ap = argparse.ArgumentParser(
        description="JurisBoost Legal Precedent Retrieval Engine - Qdrant 1.19 RRF Demo"
    )
    ap.add_argument("--query", help="Run single natural language legal search query")
    ap.add_argument(
        "--benchmark",
        action="store_true",
        help="Run 22-query legal evaluation benchmark",
    )
    ap.add_argument(
        "--rebuild", action="store_true", help="Force rebuild and re-embed index"
    )
    args = ap.parse_args()

    client = connect_qdrant()
    dense = DensePrecedentEmbedder()
    colbert = ColbertLateInteractionEmbedder()

    ensure_index(client, dense, colbert, force_rebuild=args.rebuild)

    if args.benchmark:
        from src.benchmark import run_benchmark

        run_benchmark(top_k=3)
        return

    if args.query:
        showcase_search_ladder(client, dense, colbert, query=args.query)
        return

    console.print(
        Panel.fit(
            "[bold white]JURISBOOST: INDIAN LEGAL PRECEDENT RETRIEVAL ENGINE[/bold white]\n"
            "[cyan]Qdrant 1.19 · NyayaRAG criminal precedents · Hybrid RRF + ColBERT + FormulaQuery[/cyan]\n"
            "[dim]Zero external reranker. Each ladder mode is a single query_points round-trip.[/dim]",
            border_style="blue",
        )
    )
    console.print()

    showcase_search_ladder(client, dense, colbert)
    showcase_analytics(client, dense)

    console.print(
        Panel.fit(
            "[bold white]3. BENCHMARK: 22 QUERIES ACROSS 3 TIERS[/bold white]\n"
            "[dim]Explicit (8) + Natural (8) + Edge (6). M3 and M4 share the same FormulaQuery; M4 adds ColBERT.[/dim]",
            border_style="green",
        )
    )
    from src.benchmark import run_benchmark

    run_benchmark(top_k=3)


if __name__ == "__main__":
    main()
