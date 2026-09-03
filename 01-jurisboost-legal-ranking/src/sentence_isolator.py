from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np

from .embedder import ColbertLateInteractionEmbedder

# Protected statutory & legal terms to prevent erroneous sentence boundary splits.
# NOTE: BNSS/BNS/BSA share prefixes - the trailing (?!S) guard stops the
# shorter pattern from matching the prefix of the longer one (e.g. BNS
# inside BNSS). BNSS must stay before BNS in this list.
PROTECTED_LEGAL_ABBREVIATIONS = [
    (re.compile(r"\bCr\.?\s*P\.?\s*C\.?", re.IGNORECASE), "CrPC"),
    (re.compile(r"\bI\.?\s*P\.?\s*C\.?", re.IGNORECASE), "IPC"),
    (re.compile(r"\bI\.?\s*E\.?\s*A\.?", re.IGNORECASE), "IEA"),
    (re.compile(r"\bB\.?\s*N\.?\s*S\.?\s*S\.?", re.IGNORECASE), "BNSS"),
    (re.compile(r"\bB\.?\s*N\.?\s*S\.?(?!S)", re.IGNORECASE), "BNS"),
    (re.compile(r"\bB\.?\s*S\.?\s*A\.?(?!S)", re.IGNORECASE), "BSA"),
    (
        re.compile(
            r"\b(Sec|Art|No|Dr|Mr|Mrs|Ms|Hon\'?ble|Ors|Anr|Ltd|Pvt|Co|v|vs|para|cl|u/s|r/w)\.",
            re.IGNORECASE,
        ),
        lambda m: m.group(1) + "@@DOT@@",
    ),
]


def split_legal_text_into_sentences(raw_passage_text: str) -> list[str]:
    """Split legal passage text into clean, complete sentences without fracturing abbreviations."""
    normalized_text = raw_passage_text
    for regex_pattern, replacement in PROTECTED_LEGAL_ABBREVIATIONS:
        normalized_text = regex_pattern.sub(replacement, normalized_text)  # type: ignore

    candidate_sentences = re.split(r"(?<=[.!?])\s+", normalized_text)
    sanitized_sentences = [
        s.replace("@@DOT@@", ".").strip()
        for s in candidate_sentences
        if len(s.strip()) > 15
    ]
    return sanitized_sentences or [raw_passage_text.strip()]


def compute_colbert_max_sim(
    query_token_matrix: Sequence[Sequence[float]] | np.ndarray,
    sentence_token_matrix: Sequence[Sequence[float]] | np.ndarray,
) -> float:
    """
    Compute ColBERT Late-Interaction MaxSim score between Query tokens and Sentence tokens:
    MaxSim(Q, D) = sum_{i=1}^{|Q|} max_{j=1}^{|D|} (Q_i dot D_j^T)
    """
    q_mat = np.asarray(query_token_matrix, dtype=np.float32)
    d_mat = np.asarray(sentence_token_matrix, dtype=np.float32)
    if q_mat.size == 0 or d_mat.size == 0:
        return 0.0
    token_sim_matrix = q_mat @ d_mat.T
    return float(token_sim_matrix.max(axis=1).sum())


def isolate_ratio_colbert(
    precedent_text: str,
    query_colbert_matrix: Sequence[Sequence[float]] | np.ndarray,
    colbert_embedder: ColbertLateInteractionEmbedder,
    relative_semantic_threshold: float = 0.90,
    max_selected_sentences: int = 2,
) -> tuple[list[tuple[str, float]], dict[str, int]]:
    """
    Reference-Fidelity ColBERT Token MAX_SIM Sentence Isolation:
    1. Splits legal passage into complete sentences with legal abbreviation preservation.
    2. Embeds each sentence token matrix via ColBERT multi-vector encoder.
    3. Computes exact token MaxSim(Q, S) = sum(max(Q @ S.T)) against query token matrix.
    4. Selects top ratio sentences within threshold to achieve 55%-80% token reduction.
    """
    candidate_sentences = split_legal_text_into_sentences(precedent_text)
    if len(candidate_sentences) <= 1:
        return [(candidate_sentences[0], 1.0)], {
            "raw_words": len(precedent_text.split()),
            "isolated_words": len(candidate_sentences[0].split()),
            "total_sentences": 1,
            "selected_sentences": 1,
        }

    sentence_token_matrices = colbert_embedder.embed_precedent_passages(
        candidate_sentences
    )
    similarity_scores = np.array(
        [
            compute_colbert_max_sim(query_colbert_matrix, s_mat)
            for s_mat in sentence_token_matrices
        ],
        dtype=np.float32,
    )

    highest_score = (
        float(np.max(similarity_scores)) if len(similarity_scores) > 0 else 0.0
    )
    minimum_score_cutoff = highest_score * relative_semantic_threshold

    qualified_indices = [
        idx
        for idx, score in enumerate(similarity_scores)
        if score >= minimum_score_cutoff
    ]
    if len(qualified_indices) > max_selected_sentences:
        ranked_indices = sorted(
            qualified_indices, key=lambda idx: similarity_scores[idx], reverse=True
        )[:max_selected_sentences]
        qualified_indices = sorted(ranked_indices)
    elif not qualified_indices:
        qualified_indices = [int(np.argmax(similarity_scores))]

    isolated_sentences = [
        (candidate_sentences[idx], float(similarity_scores[idx]))
        for idx in qualified_indices
    ]
    raw_word_count = len(precedent_text.split())
    isolated_word_count = sum(len(sent.split()) for sent, _ in isolated_sentences)

    return isolated_sentences, {
        "raw_words": raw_word_count,
        "isolated_words": isolated_word_count,
        "total_sentences": len(candidate_sentences),
        "selected_sentences": len(isolated_sentences),
    }
