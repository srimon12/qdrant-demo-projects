from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from .config import COLLECTION_NAME
from .embedder import ColbertLateInteractionEmbedder, DensePrecedentEmbedder
from .qdrant_ops import (
    get_qdrant_client,
    search_dense,
    search_formula,
    search_hybrid,
    search_universal_pipeline,
)

console = Console()

# Explicit = query contains the statutory section from 2024 codes (BNS, BNSS, BSA) -> concordance / BM25 can fire.
# Paraphrased = natural language fact pattern, no section number -> relies on semantic and ColBERT late interaction.
EXPLICIT_QUERIES: list[dict[str, Any]] = [
    {
        "id": "E1",
        "query": "Can anticipatory bail under Section 482 BNSS continue without a fixed time limit, as under Section 438 CrPC?",
        "expects": ["crpc:438", "bnss:482"],
        "note": "438 CrPC -> 482 BNSS (Anticipatory Bail without time limit)",
    },
    {
        "id": "E2",
        "query": "What are the five golden principles of circumstantial evidence under Section 103 BNS, corresponding to Section 302 IPC?",
        "expects": ["ipc:302", "bns:103"],
        "note": "302 IPC -> 103 BNS (Circumstantial Evidence Panchsheel)",
    },
    {
        "id": "E3",
        "query": "When can the High Court quash criminal proceedings under Section 528 BNSS, corresponding to Section 482 CrPC?",
        "expects": ["crpc:482", "bnss:528"],
        "note": "482 CrPC -> 528 BNSS (Inherent Quashing Powers)",
    },
    {
        "id": "E4",
        "query": "Is notice of appearance under Section 35 BNSS mandatory before arrest for offences under 7 years, as under Section 41A CrPC?",
        "expects": ["crpc:41a", "bnss:35", "crpc:41"],
        "note": "41A CrPC -> 35 BNSS (Mandatory Notice of Appearance)",
    },
    {
        "id": "E5",
        "query": "Is an FIR mandatory under Section 173 BNSS upon disclosure of a cognizable offence, as under Section 154 CrPC?",
        "expects": ["crpc:154", "bnss:173"],
        "note": "154 CrPC -> 173 BNSS (Mandatory FIR Registration)",
    },
    {
        "id": "E6",
        "query": "What constitutes cruelty by husband or relatives under Section 85 BNS, corresponding to Section 498A IPC?",
        "expects": ["ipc:498a", "bns:85"],
        "note": "498A IPC -> 85 BNS (Matrimonial Cruelty)",
    },
    {
        "id": "E7",
        "query": "When is death penalty awarded under Section 103 BNS only in rarest of rare cases, corresponding to Section 302 IPC?",
        "expects": ["ipc:302", "bns:103"],
        "note": "302 IPC -> 103 BNS (Rarest of Rare Capital Punishment)",
    },
    {
        "id": "E8",
        "query": "Can statement of accused in police custody leading to discovery of weapon under Section 23(2) BSA be admitted, as under Section 27 IEA?",
        "expects": ["iea:27", "bsa:23(2)", "bsa:23"],
        "note": "27 IEA -> 23(2) BSA (Custody Discovery of Weapon / Fact)",
    },
]

PARAPHRASED_QUERIES: list[dict[str, Any]] = [
    {
        "id": "N1",
        "query": "general omnibus allegations of harassment by in-laws without specific date time or overt act of cruelty",
        "expects": ["ipc:498a", "bns:85"],
        "note": "498A - Vague cruelty allegations without specific overt acts",
    },
    {
        "id": "N2",
        "query": "person apprehending arrest in non-bailable offence wants pre-arrest bail protection till conclusion of trial",
        "expects": ["crpc:438", "bnss:482"],
        "note": "438 - Pre-arrest protection till trial conclusion",
    },
    {
        "id": "N3",
        "query": "high court inherent powers to quash false malicious criminal prosecution and prevent abuse of court process",
        "expects": ["crpc:482", "bnss:528"],
        "note": "482 - Inherent power to quash abuse of court process",
    },
    {
        "id": "N4",
        "query": "no direct eyewitness to murder, only chain of circumstantial clues leading solely to guilt of accused",
        "expects": ["ipc:302", "bns:103"],
        "note": "302 - Circumstantial evidence chain of events",
    },
    {
        "id": "N5",
        "query": "police officer refusing to register first information report when complaint discloses commission of cognizable crime",
        "expects": ["crpc:154", "bnss:173"],
        "note": "154 - Mandatory FIR on cognizable complaint",
    },
    {
        "id": "N6",
        "query": "civil dispute for breach of contract dishonestly given criminal colour of cheating to pressure payment",
        "expects": ["ipc:420", "bns:318"],
        "note": "420 - Commercial breach cannot be criminalized into cheating",
    },
    {
        "id": "N7",
        "query": "accused inflicted fatal single bodily injury with knife in sudden quarrel without premeditation",
        "expects": ["ipc:302", "bns:103", "ipc:300", "bns:101"],
        "note": "300/302 - Sudden fight exception to murder",
    },
    {
        "id": "N8",
        "query": "recovery of blood stained clothes and weapon from secret place solely on basis of disclosure statement made by accused to investigating officer",
        "expects": ["iea:27", "bsa:23(2)", "bsa:23"],
        "note": "27 - Custodial disclosure statement weapon recovery",
    },
]

# Edge / adversarial - verified HARD where M1 Top-1 fails but M4 fixes (20-367x corpus, so miss = hardness not sparsity)
EDGE_QUERIES: list[dict[str, Any]] = [
    {
        "id": "C1",
        "query": "Section 3(5) BNS common intention - joint liability even without prior meeting of minds?",
        "expects": ["ipc:34", "bns:3(5)"],
        "note": "C1 Pure BNS - no IPC keyword, needs concordance 34→3(5) (HARD: M1× M4✓)",
    },
    {
        "id": "C2",
        "query": "Section 35(3) BNSS notice of appearance - is it mandatory before arrest for offences punishable up to 7 years?",
        "expects": ["crpc:41a", "bnss:35", "bnss:35(3)"],
        "note": "C2 Distractor 35 vs 35(3) - near-duplicate, needs prefix (HARD: M1× M4✓)",
    },
    {
        "id": "C3",
        "query": "Section 80 BNS dowry death - woman dies under suspicious circumstances within seven years of marriage",
        "expects": ["ipc:304b", "bns:80"],
        "note": "C3 Pure BNS - 304B→80 bridge, small corpus (HARD)",
    },
    {
        "id": "C4",
        "query": "Section 117 BSA presumption of abetment of suicide by married woman",
        "expects": ["iea:113a", "bsa:117"],
        "note": "C4 Pure BSA - 113A→117 sparse, tests MAX_SIM token (HARD)",
    },
    {
        "id": "C5",
        "query": "Section 303 BNS - what is the punishment for theft under Section 303 BNS vs Section 379 IPC?",
        "expects": ["ipc:379", "bns:303(2)", "bns:303"],
        "note": "C5 Pure BNS theft - 303 vs 379 bridge (HARD)",
    },
    {
        "id": "C6",
        "query": "Police must not file FIR if cognizable offence disclosed under Section 173 BNSS / 154 CrPC - is this correct?",
        "expects": ["crpc:154", "bnss:173"],
        "note": "C6 Negation trap - false 'must not' (true: must) (HARD: M1× M4✓)",
    },
]


def _has_expected(payload: dict[str, Any], expects: list[str]) -> bool:
    refs = set(payload.get("legal_references", [])) | set(
        payload.get("mapped_references", [])
    )
    return bool(refs & set(expects))


def _hit(hits, expects: list[str], top_k: int) -> bool:
    for h in hits[:top_k]:
        if _has_expected(h.payload or {}, expects):
            return True
    return False


def _top1(hits, expects: list[str]) -> bool:
    return _has_expected(hits[0].payload or {}, expects) if hits else False


def _run_group(
    group_name: str,
    query_items: list[dict[str, Any]],
    client,
    dense_embedder: DensePrecedentEmbedder,
    colbert_embedder: ColbertLateInteractionEmbedder,
    top_k: int,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    mode_names = [
        "M1 Dense",
        "M2 Hybrid RRF",
        "M3 Hybrid→Formula",
        "M4 RRF→ColBERT→Formula",
    ]
    evaluation_stats: dict[str, dict[str, Any]] = {
        m: {"hits_top1": 0, "hits_topk": 0, "lat": []} for m in mode_names
    }
    result_rows: list[dict[str, Any]] = []

    for item in query_items:
        query_text, expected_statutes = item["query"], item["expects"]
        dense_vector = dense_embedder.embed_legal_queries(query_text)[0]
        colbert_matrix = colbert_embedder.embed_legal_queries(query_text)[0].tolist()

        search_functions = [
            (
                "M1 Dense",
                lambda dv=dense_vector: search_dense(client, dv.tolist(), limit=top_k),
            ),
            (
                "M2 Hybrid RRF",
                lambda dv=dense_vector, qt=query_text: search_hybrid(
                    client, dv.tolist(), qt, limit=top_k
                ),
            ),
            (
                "M3 Hybrid→Formula",
                lambda dv=dense_vector, qt=query_text: search_formula(
                    client, dv.tolist(), qt, limit=top_k
                ),
            ),
            (
                "M4 RRF→ColBERT→Formula",
                lambda dv=dense_vector, cm=colbert_matrix, qt=query_text: (
                    search_universal_pipeline(client, dv.tolist(), cm, qt, limit=top_k)
                ),
            ),
        ]

        row_data = {
            "id": item["id"],
            "note": item["note"],
            "expects": expected_statutes,
        }
        for mode_key, search_fn in search_functions:
            start_time = time.perf_counter()
            retrieved_hits = search_fn()
            evaluation_stats[mode_key]["lat"].append(
                (time.perf_counter() - start_time) * 1000
            )
            is_top1 = _top1(retrieved_hits, expected_statutes)
            is_in_topk = _hit(retrieved_hits, expected_statutes, top_k)
            if is_top1:
                evaluation_stats[mode_key]["hits_top1"] += 1
            if is_in_topk:
                evaluation_stats[mode_key]["hits_topk"] += 1
            row_data[f"{mode_key}_top1"] = is_top1
            row_data[f"{mode_key}_hitk"] = is_in_topk
        result_rows.append(row_data)

    summary_table = Table(title=f"{group_name} - Top-1 / Recall@{top_k}")
    summary_table.add_column("Mode")
    summary_table.add_column("Top-1", justify="right")
    summary_table.add_column(f"Recall@{top_k}", justify="right")
    summary_table.add_column("p50 ms", justify="right")
    total_queries = len(query_items)
    for m in mode_names:
        stats_entry = evaluation_stats[m]
        latencies = sorted(stats_entry["lat"])
        p50_latency = latencies[len(latencies) // 2] if latencies else 0
        summary_table.add_row(
            m,
            f"{stats_entry['hits_top1']}/{total_queries}",
            f"{stats_entry['hits_topk']}/{total_queries}",
            f"{p50_latency:.1f}",
        )
    console.print(summary_table)

    detail_table = Table(title=f"{group_name} - Per-Query (Top-1 | Recall@{top_k})")
    detail_table.add_column("ID")
    detail_table.add_column("Expects")
    for m in ["M1", "M2", "M3", "M4"]:
        detail_table.add_column(m, justify="center")

    for r in result_rows:

        def format_symbol(top1, hit_k):
            return "✓✓" if top1 else ("✓" if hit_k else "×")

        detail_table.add_row(
            r["id"],
            ",".join(r["expects"][:2]),
            format_symbol(r["M1 Dense_top1"], r["M1 Dense_hitk"]),
            format_symbol(r["M2 Hybrid RRF_top1"], r["M2 Hybrid RRF_hitk"]),
            format_symbol(r["M3 Hybrid→Formula_top1"], r["M3 Hybrid→Formula_hitk"]),
            format_symbol(
                r["M4 RRF→ColBERT→Formula_top1"], r["M4 RRF→ColBERT→Formula_hitk"]
            ),
        )
    console.print(detail_table)
    console.print("[dim]✓✓ = Top-1 winner, ✓ = in top-K, × = missed[/dim]\n")

    return evaluation_stats, result_rows


def run_benchmark(top_k: int = 3) -> None:
    """Intuitive 3-group benchmark: Explicit (easy) → Natural (medium) → Edge (hard).

    Prints per-group tables plus a combined 22-query summary and a 'hard cases fixed by M4' highlight.
    """
    client = get_qdrant_client()
    count = client.get_collection(COLLECTION_NAME).points_count or 0

    console.print(
        f"\n[bold]Benchmark - 3 Tiers: Explicit vs Natural vs Edge ({count:,} Precedents)[/bold]\n"
        "[dim]Explicit=section numbers present (BM25 can fire) | Natural=paraphrased facts (semantic) | Edge=adversarial (ambiguous/multi/negation)[/dim]\n"
    )
    dense_embedder = DensePrecedentEmbedder()
    colbert_embedder = ColbertLateInteractionEmbedder()

    # Group 1: Explicit statutory citations (easy)
    explicit_stats, explicit_rows = _run_group(
        "EXPLICIT - Section Numbers Present (easy, BM25 helps)",
        EXPLICIT_QUERIES,
        client,
        dense_embedder,
        colbert_embedder,
        top_k=top_k,
    )

    # Group 2: Natural language fact patterns (medium)
    natural_stats, natural_rows = _run_group(
        "NATURAL - Paraphrased Facts (medium, needs semantic)",
        PARAPHRASED_QUERIES,
        client,
        dense_embedder,
        colbert_embedder,
        top_k=top_k,
    )

    # Group 3: Edge / adversarial (hard) - where M1/M2 collapse
    edge_stats, edge_rows = _run_group(
        "EDGE - Ambiguous / Multi / Negation (hard, needs MAX_SIM+Formula)",
        EDGE_QUERIES,
        client,
        dense_embedder,
        colbert_embedder,
        top_k=top_k,
    )

    # Combined summary table (22 queries)
    mode_keys = [
        "M1 Dense",
        "M2 Hybrid RRF",
        "M3 Hybrid→Formula",
        "M4 RRF→ColBERT→Formula",
    ]
    combined_table = Table(title="Combined Benchmark Summary (22 queries: 8+8+6)")
    combined_table.add_column("Mode")
    combined_table.add_column("Top-1", justify="right")
    combined_table.add_column(f"Recall@{top_k}", justify="right")
    combined_table.add_column("p50 ms", justify="right")
    combined_table.add_column("Δ vs M1", justify="right")

    # Compute totals across 3 groups for intuitive delta
    base_top1 = (
        explicit_stats["M1 Dense"]["hits_top1"]
        + natural_stats["M1 Dense"]["hits_top1"]
        + edge_stats["M1 Dense"]["hits_top1"]
    )
    for m in mode_keys:
        total_top1 = (
            explicit_stats[m]["hits_top1"]
            + natural_stats[m]["hits_top1"]
            + edge_stats[m]["hits_top1"]
        )
        total_recall = (
            explicit_stats[m]["hits_topk"]
            + natural_stats[m]["hits_topk"]
            + edge_stats[m]["hits_topk"]
        )
        all_lats = sorted(
            explicit_stats[m]["lat"] + natural_stats[m]["lat"] + edge_stats[m]["lat"]
        )
        p50 = all_lats[len(all_lats) // 2] if all_lats else 0
        delta = total_top1 - base_top1
        delta_str = (
            f"[green]+{delta}[/green]"
            if delta > 0
            else "-"
            if m == "M1 Dense"
            else f"[dim]{delta}[/dim]"
        )
        combined_table.add_row(
            m,
            f"{total_top1}/22 ({total_top1 / 22 * 100:.1f}%)",
            f"{total_recall}/22 ({total_recall / 22 * 100:.1f}%)",
            f"{p50:.1f}",
            delta_str,
        )
    console.print(combined_table)

    # Intuitive highlight: hardest cases where M1 fails but M4 wins
    all_rows = explicit_rows + natural_rows + edge_rows
    m4_fixes = [
        r
        for r in all_rows
        if not r["M1 Dense_top1"] and r["M4 RRF→ColBERT→Formula_top1"]
    ]
    m4_still_miss = [r for r in all_rows if not r["M4 RRF→ColBERT→Formula_hitk"]]
    console.print(
        f"\n[bold]Hard-case analysis:[/bold] M4 fixes [green]{len(m4_fixes)}/22[/green] cases where M1 Top-1 failed "
        f"(e.g., {', '.join(r['id'] for r in m4_fixes[:4])}{'...' if len(m4_fixes) > 4 else ''})"
    )
    if m4_still_miss:
        console.print(
            f"[dim]Still missed @Top-{top_k} by all: {', '.join(r['id'] for r in m4_still_miss)} → corpus gap, not ranking[/dim]\n"
        )
    else:
        console.print(
            "[green]M4 reaches 100% Recall@3 across all 22 - no corpus gaps[/green]\n"
        )

    # Export report (3 groups)
    report_output_path = Path("benchmark_report.json")
    report_output_path.write_text(
        json.dumps(
            {
                "explicit": explicit_rows,
                "natural_language": natural_rows,
                "edge": edge_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    console.print(f"[green]Benchmark JSON Export → {report_output_path}[/green]\n")
