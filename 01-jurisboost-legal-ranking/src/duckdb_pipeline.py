from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from .config import DEFAULT_NYAYARAG_DATA_FILE
from .data import (
    ARTICLE_CITATION_PATTERN,
    deserialize_json_or_literal,
    extract_judgment_decision_date,
    extract_ratio_decidendi,
    extract_statutory_references,
    format_statutory_keys,
    normalize_case_title,
    normalize_whitespace,
    resolve_concordance_bridge_keys,
)


def resolve_dataset_filepath(custom_path: Path | None = None) -> Path:
    """Return configured dataset filepath or default bundled NyayaRAG path."""
    return custom_path or DEFAULT_NYAYARAG_DATA_FILE


def deserialize_list_field(raw_field: Any) -> list[Any]:
    """Safely deserialize stringified list structures from JSON records."""
    if isinstance(raw_field, list):
        return raw_field
    if not raw_field:
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed_list = parser(raw_field)
            if isinstance(parsed_list, list):
                return parsed_list
        except (ValueError, TypeError, json.JSONDecodeError):
            continue
    return []


def load_and_aggregate_precedents(
    dataset_path: Path,
    source_limit: int | None = None,
    domain_filter: str = "criminal",
) -> list[dict[str, Any]]:
    """
    DuckDB In-Memory OLAP Precedent Ingestion:
    1. Flattens cited precedent occurrences across raw source judgments.
    2. Executes in-memory columnar SQL to deduplicate precedents by normalized title.
    3. Computes citation graph in-degree authority (COUNT of distinct citing judgments).
    4. Enriches payloads with 2024 statutory concordance bridges and decision dates.
    """
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"NyayaRAG dataset missing: {dataset_path}\n"
            "From this folder run:  uv run python scripts/download_data.py"
        )

    source_judgments = json.loads(dataset_path.read_text(encoding="utf-8"))
    if source_limit is not None:
        source_judgments = source_judgments[:source_limit]

    # 1. Flatten all cited precedent occurrences
    precedent_occurrences = []
    for judgment in source_judgments:
        judgment_id = normalize_whitespace(judgment.get("document_id"))
        cited_titles = deserialize_list_field(judgment.get("cited_cases"))
        cited_data_dict = deserialize_json_or_literal(judgment.get("cited_cases_data"))

        for index, raw_title in enumerate(cited_titles, 1):
            title = normalize_whitespace(raw_title)
            text = normalize_whitespace(cited_data_dict.get(f"cited_case_{index}"))
            if title and len(text) >= 80:
                precedent_occurrences.append(
                    {
                        "precedent_key": normalize_case_title(title),
                        "title": title,
                        "text": text,
                        "text_length": len(text),
                        "source_case_id": judgment_id,
                    }
                )

    if not precedent_occurrences:
        raise ValueError("No valid cited precedents found in NyayaRAG dataset.")

    # 2. In-Memory DuckDB OLAP: Deduplicate & calculate citation in-degree counts
    duckdb_connection = duckdb.connect(database=":memory:")
    duckdb_connection.register(
        "precedent_occurrences", pd.DataFrame(precedent_occurrences)
    )
    aggregated_precedents_df = duckdb_connection.execute("""
        SELECT
            precedent_key,
            arg_max(title, text_length) AS title,
            arg_max(text, text_length) AS text,
            count(DISTINCT source_case_id)::INTEGER AS source_case_count,
            list(DISTINCT source_case_id) AS source_case_ids
        FROM precedent_occurrences
        GROUP BY ALL
        ORDER BY precedent_key
    """).df()
    duckdb_connection.close()

    # 3. Enrich precedents with statutory sections, concordance bridges, and dates
    enriched_precedents = []
    for record in aggregated_precedents_df.to_dict("records"):
        references = extract_statutory_references(record["text"])
        statute_labels = sorted(
            {f"Section {sec.upper()}" for sec, _ in references}
            | {
                f"Article {m.group(1).upper()}"
                for m in ARTICLE_CITATION_PATTERN.finditer(record["text"])
            }
        )
        enriched_precedents.append(
            {
                **record,
                "ratio": extract_ratio_decidendi(record["text"]),
                "judgment_date": extract_judgment_decision_date(
                    record["title"], record["text"]
                ),
                "statute_labels": statute_labels,
                "legal_references": format_statutory_keys(references),
                "mapped_references": resolve_concordance_bridge_keys(references),
                "source_case_ids": sorted(
                    normalize_whitespace(s) for s in record["source_case_ids"] if s
                )[:5],
                "word_count": len(record["text"].split()),
                "search_text": f"{record['title']}. {record['text']}",
            }
        )

    # 4. Domain filtering
    if domain_filter == "criminal":
        enriched_precedents = [
            p
            for p in enriched_precedents
            if any(
                k.startswith(("ipc:", "crpc:", "bns:", "bnss:", "iea:", "bsa:"))
                for k in p["legal_references"] + p["mapped_references"]
            )
        ]
    return enriched_precedents
