from __future__ import annotations

import ast
import json
import re
from datetime import UTC, datetime
from typing import Any

# =============================================================================
# 1. Statutory Concordance Crosswalk: Historic (1860-1973) <-> 2024 Bharatiya Codes
# =============================================================================
STATUTORY_CONCORDANCE_MAP: dict[tuple[str, str], tuple[str, str]] = {
    # Indian Penal Code (IPC, 1860) <-> Bharatiya Nyaya Sanhita (BNS, 2023)
    ("ipc", "34"): ("bns", "3(5)"),  # Common Intention
    ("ipc", "120b"): ("bns", "61"),  # Criminal Conspiracy
    ("ipc", "124a"): ("bns", "152"),  # Endangering Sovereignty
    ("ipc", "149"): ("bns", "190"),  # Unlawful Assembly
    ("ipc", "191"): ("bns", "227"),  # Perjury
    ("ipc", "193"): ("bns", "229"),  # Punishment for Perjury
    ("ipc", "201"): ("bns", "238"),  # Disappearance of Evidence
    ("ipc", "300"): ("bns", "101"),  # Murder definition
    ("ipc", "302"): ("bns", "103"),  # Murder punishment
    ("ipc", "304"): ("bns", "105"),  # Culpable Homicide
    ("ipc", "304a"): ("bns", "106"),  # Death by Negligence
    ("ipc", "304b"): ("bns", "80"),  # Dowry Death
    ("ipc", "306"): ("bns", "108"),  # Abetment of Suicide
    ("ipc", "307"): ("bns", "109"),  # Attempt to Murder
    ("ipc", "323"): ("bns", "115(2)"),  # Causing Hurt
    ("ipc", "326"): ("bns", "118(1)"),  # Grievous Hurt
    ("ipc", "354"): ("bns", "74"),  # Outraging Modesty
    ("ipc", "354a"): ("bns", "75"),  # Sexual Harassment
    ("ipc", "375"): ("bns", "63"),  # Rape definition
    ("ipc", "376"): ("bns", "64"),  # Rape punishment
    ("ipc", "376d"): ("bns", "70"),  # Gang Rape
    ("ipc", "378"): ("bns", "303(1)"),  # Theft definition
    ("ipc", "379"): ("bns", "303(2)"),  # Theft punishment
    ("ipc", "392"): ("bns", "309"),  # Robbery
    ("ipc", "395"): ("bns", "310"),  # Dacoity
    ("ipc", "406"): ("bns", "316"),  # Criminal Breach of Trust
    ("ipc", "415"): ("bns", "318(1)"),  # Cheating definition
    ("ipc", "420"): ("bns", "318"),  # Cheating & Dishonesty
    ("ipc", "467"): ("bns", "338"),  # Forgery of Valuable Security
    ("ipc", "468"): ("bns", "336(3)"),  # Forgery for Cheating
    ("ipc", "471"): ("bns", "340"),  # Using Forged Document
    ("ipc", "498a"): ("bns", "85"),  # Matrimonial Cruelty
    ("ipc", "499"): ("bns", "356(1)"),  # Defamation definition
    ("ipc", "500"): ("bns", "356(2)"),  # Defamation punishment
    ("ipc", "506"): ("bns", "351"),  # Criminal Intimidation
    # Code of Criminal Procedure (CrPC, 1973) <-> Bharatiya Nagarik Suraksha Sanhita (BNSS, 2023)
    ("crpc", "41"): ("bnss", "35"),  # Arrest without Warrant
    ("crpc", "41a"): ("bnss", "35(3)"),  # Notice of Appearance
    ("crpc", "50"): ("bnss", "47"),  # Inform Grounds of Arrest
    ("crpc", "57"): ("bnss", "58"),  # 24-hr Detention Limit
    ("crpc", "125"): ("bnss", "144"),  # Maintenance
    ("crpc", "144"): ("bnss", "163"),  # Urgent Nuisance
    ("crpc", "154"): ("bnss", "173"),  # FIR Registration
    ("crpc", "161"): ("bnss", "180"),  # Witness Examination
    ("crpc", "164"): ("bnss", "183"),  # Recording Confessions
    ("crpc", "167"): ("bnss", "187"),  # Remand Procedure
    ("crpc", "173"): ("bnss", "193"),  # Police Final Report
    ("crpc", "190"): ("bnss", "210"),  # Magistrate Cognizance
    ("crpc", "200"): ("bnss", "223"),  # Complainant Examination
    ("crpc", "227"): ("bnss", "250"),  # Sessions Discharge
    ("crpc", "239"): ("bnss", "262"),  # Warrant Case Discharge
    ("crpc", "313"): ("bnss", "351"),  # Accused Examination
    ("crpc", "320"): ("bnss", "359"),  # Compounding Offences
    ("crpc", "357"): ("bnss", "395"),  # Victim Compensation
    ("crpc", "389"): ("bnss", "430"),  # Sentence Suspension
    ("crpc", "436"): ("bnss", "478"),  # Bailable Bail
    ("crpc", "436a"): ("bnss", "479"),  # Undertrial Detention
    ("crpc", "437"): ("bnss", "480"),  # Non-Bailable Bail
    ("crpc", "438"): ("bnss", "482"),  # Anticipatory Bail
    ("crpc", "439"): ("bnss", "483"),  # Special Bail Powers
    ("crpc", "482"): ("bnss", "528"),  # Inherent Quashing Powers
    # Indian Evidence Act (IEA, 1872) <-> Bharatiya Sakshya Adhiniyam (BSA, 2023)
    ("iea", "24"): ("bsa", "22"),  # Involuntary Confession
    ("iea", "25"): ("bsa", "23(1)"),  # Police Confession
    ("iea", "27"): ("bsa", "23(2)"),  # Discovery of Fact / Weapon
    ("iea", "32"): ("bsa", "26"),  # Dying Declaration
    ("iea", "45"): ("bsa", "39"),  # Expert Opinions
    ("iea", "65a"): ("bsa", "61"),  # Electronic Record Admissibility
    ("iea", "65b"): ("bsa", "63"),  # Electronic Certificate
    ("iea", "101"): ("bsa", "104"),  # Burden of Proof
    ("iea", "102"): ("bsa", "105"),  # On Whom Burden Lies
    ("iea", "106"): ("bsa", "109"),  # Special Knowledge Burden
    ("iea", "113a"): ("bsa", "117"),  # Suicide Abetment Presumption
    ("iea", "113b"): ("bsa", "118"),  # Dowry Death Presumption
    ("iea", "114"): ("bsa", "119"),  # Presumption of Facts
    ("iea", "118"): ("bsa", "124"),  # Competency of Witnesses
    ("iea", "133"): ("bsa", "138"),  # Accomplice Testimony
}

SECTION_CITATION_PATTERN = re.compile(
    r"\b(?:section|sec\.?|s\.)\s*([0-9]+[A-Za-z]*(?:\([0-9A-Za-z]+\))?)", re.IGNORECASE
)
ARTICLE_CITATION_PATTERN = re.compile(
    r"\b(?:article|art\.?)\s*([0-9]+[A-Za-z]*)", re.IGNORECASE
)

STATUTORY_ACT_KEYWORD_PATTERNS = [
    ("bnss", r"\b(bharatiya nagarik|nagarik suraksha|bnss)\b"),
    ("crpc", r"\b(code of criminal procedure|cr\.?p\.?c|crpc)\b"),
    ("bns", r"\b(bharatiya nyaya|nyaya sanhita|bns)\b"),
    ("ipc", r"\b(indian penal|penal code|ipc)\b"),
    ("bsa", r"\b(bharatiya sakshya|sakshya adhiniyam|bsa)\b"),
    ("iea", r"\b(evidence act|indian evidence|iea)\b"),
]


# =============================================================================
# 2. Text Normalization & Legal Citation Extractors
# =============================================================================
def normalize_whitespace(raw_value: Any) -> str:
    """Normalize multi-space, tab, and newline sequences to single space."""
    return re.sub(r"\s+", " ", str(raw_value or "")).strip()


def deserialize_json_or_literal(serialized_content: Any) -> dict[str, Any]:
    """Safely deserialize stringified JSON or python dictionary literals."""
    if isinstance(serialized_content, dict):
        return serialized_content
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed_dict = parser(serialized_content)
            if isinstance(parsed_dict, dict):
                return parsed_dict
        except (ValueError, SyntaxError, TypeError):
            continue
    return {}


def detect_statutory_act_context(text_snippet: str) -> str | None:
    """Detect canonical statute acronym (e.g., 'crpc', 'bnss') from surrounding text context."""
    for act_acronym, act_regex in STATUTORY_ACT_KEYWORD_PATTERNS:
        if re.search(act_regex, text_snippet, re.IGNORECASE):
            return act_acronym
    return None


def extract_statutory_references(legal_text: str) -> set[tuple[str, str | None]]:
    """Extract structured (section_identifier, statutory_act) tuples from legal text."""
    extracted_references = set()
    for match_obj in SECTION_CITATION_PATTERN.finditer(legal_text):
        section_number = match_obj.group(1).replace(" ", "").replace("-", "").lower()
        context_window = legal_text[
            max(0, match_obj.start() - 80) : min(len(legal_text), match_obj.end() + 80)
        ]
        detected_act = detect_statutory_act_context(context_window)
        extracted_references.add((section_number, detected_act))
    return extracted_references


def format_statutory_keys(references: set[tuple[str, str | None]]) -> list[str]:
    """Format structured statutory references into Qdrant keyword tags (e.g. 'crpc:482')."""
    return sorted(
        {f"{act}:{sec}" if act else f"section:{sec}" for sec, act in references}
    )


def resolve_concordance_bridge_keys(
    references: set[tuple[str, str | None]],
) -> list[str]:
    """Resolve bidirectional 2024 concordance bridges for payload indexing."""
    bridged_statute_keys = set()
    for section_num, act_name in references:
        base_num = re.sub(r"\(.*?\)", "", section_num)
        for (old_act, old_sec), (new_act, new_sec) in STATUTORY_CONCORDANCE_MAP.items():
            old_base = re.sub(r"\(.*?\)", "", old_sec)
            new_base = re.sub(r"\(.*?\)", "", new_sec)
            if (
                act_name == old_act and (section_num == old_sec or base_num == old_base)
            ) or (
                act_name is None and (section_num == old_sec or base_num == old_base)
            ):
                bridged_statute_keys.add(f"{new_act}:{new_sec}")
            elif (
                act_name == new_act and (section_num == new_sec or base_num == new_base)
            ) or (
                act_name is None and (section_num == new_sec or base_num == new_base)
            ):
                bridged_statute_keys.add(f"{old_act}:{old_sec}")
    return sorted(bridged_statute_keys)


def resolve_query_statutory_bridges(search_query: str) -> tuple[list[str], list[str]]:
    """Extract direct query provisions and bidirectional concordance bridge targets for search."""
    extracted_refs = extract_statutory_references(search_query)
    direct_statute_keys = format_statutory_keys(extracted_refs)
    concordance_bridge_keys = set()

    for section_num, act_name in extracted_refs:
        base_num = re.sub(r"\(.*?\)", "", section_num)
        for (old_act, old_sec), (new_act, new_sec) in STATUTORY_CONCORDANCE_MAP.items():
            old_base = re.sub(r"\(.*?\)", "", old_sec)
            new_base = re.sub(r"\(.*?\)", "", new_sec)
            if (
                (
                    act_name == old_act
                    and (section_num == old_sec or base_num == old_base)
                )
                or (
                    act_name == new_act
                    and (section_num == new_sec or base_num == new_base)
                )
                or act_name is None
                and (
                    section_num in (old_sec, new_sec)
                    or base_num in (old_base, new_base)
                )
            ):
                concordance_bridge_keys.update(
                    [f"{old_act}:{old_sec}", f"{new_act}:{new_sec}"]
                )

    return direct_statute_keys, sorted(concordance_bridge_keys)


def extract_ratio_decidendi(precedent_text: str) -> str:
    """Extract operative ratio decidendi from structured NyayaRAG precedent text."""
    match_obj = re.search(
        r"ratio of the decision:\s*(.*?)(?=\s+ruling by present court|\s+conclusion:|$)",
        precedent_text,
        re.IGNORECASE,
    )
    return normalize_whitespace(match_obj.group(1) if match_obj else precedent_text)


def extract_judgment_decision_date(case_title: str, precedent_text: str) -> str | None:
    """Extract standard ISO-8601 decision date for Qdrant exponential temporal decay scoring."""
    match_obj = re.search(
        r"\bon\s+(\d{1,2}\s+[A-Za-z]+,?\s+\d{4})\b", f"{case_title} {precedent_text}"
    )
    if match_obj:
        try:
            date_string = normalize_whitespace(match_obj.group(1).replace(",", ""))
            return (
                datetime.strptime(date_string, "%d %B %Y")
                .replace(tzinfo=UTC)
                .strftime("%Y-%m-%dT00:00:00Z")
            )
        except ValueError:
            pass
    return None


def normalize_case_title(raw_title: str) -> str:
    """Normalize precedent case titles for deterministic deduplication."""
    return re.sub(r"[^a-z0-9]+", " ", raw_title.lower()).strip()
