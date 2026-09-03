#!/usr/bin/env python3
from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

NYAYARAG_HF_ZIP_URL = "https://huggingface.co/datasets/L-NLProc/NyayaRAG/resolve/main/8.Facts_Statutes_Precedents.zip"
TARGET_DATASET_MEMBER = (
    "7.Facts_Statutes_Precedents/5k_single_summarised_CitedPlusFacts.json"
)
DEFAULT_OUTPUT_JSON_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "nyayarag"
    / "nyayarag_single_cited_precedents.json"
)


def download_nyayarag_dataset(
    destination_filepath: Path = DEFAULT_OUTPUT_JSON_PATH,
) -> Path:
    """Download and extract the NyayaRAG dataset from Hugging Face if not already cached."""
    destination_path = Path(destination_filepath)
    if destination_path.exists():
        return destination_path

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading NyayaRAG precedent dataset from Hugging Face (~35MB archive)...")

    with (
        urllib.request.urlopen(NYAYARAG_HF_ZIP_URL) as response,
        zipfile.ZipFile(io.BytesIO(response.read())) as zip_archive,
    ):
        extracted_bytes = zip_archive.read(TARGET_DATASET_MEMBER)
        destination_path.write_bytes(extracted_bytes)

    print(f"Saved extracted dataset to {destination_path}")
    return destination_path


if __name__ == "__main__":
    download_nyayarag_dataset()
