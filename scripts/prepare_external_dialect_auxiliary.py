#!/usr/bin/env python3
"""Download and normalize licensed auxiliary dialect resources.

This utility does not train a model, build stories, or create ANHC/CKG data.
It keeps source labels and licensing metadata attached to every exported row.

Prerequisites when running outside this restricted workspace:
    pip install datasets
    python prepare_external_dialect_auxiliary.py --output-dir data/external_dialects

QCRI rows are exported for auxiliary DIA/POS inspection. Lahga rows are exported
as lexicon candidates with their needs_review flag intact. Do not merge these
outputs into narrative-story training data or treat them as thesis results.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


QCRI_SOURCE = "QCRI/arabic_pos_dialect"
LAHGA_SOURCE = "lahga/arabic-dialects"
QCRI_LICENSE = "Apache-2.0 (as published on the dataset card)"
LAHGA_LICENSE = "CC BY-SA 4.0 (as published on the dataset card)"


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_qcri() -> list[dict[str, Any]]:
    from datasets import load_dataset

    result: list[dict[str, Any]] = []
    for config, variety in (
        ("egy", "Egyptian"),
        ("lev", "Levantine"),
        ("glf", "Gulf"),
        ("mgr", "Maghrebi"),
    ):
        dataset = load_dataset(QCRI_SOURCE, config, split="train")
        for row_number, row in enumerate(dataset):
            result.append({
                "schema": "ansg.external_dialect_auxiliary/1",
                "resource_id": "qcri_arabic_pos_dialect",
                "source_url": "https://huggingface.co/datasets/QCRI/arabic_pos_dialect",
                "source_license": QCRI_LICENSE,
                "source_row_id": f"{config}:{row_number}",
                "target_family": variety,
                "task_scope": ["dialect_support", "segmentation", "pos"],
                "story_training_eligible": False,
                "ckg_eligible": False,
                "privacy_review": "required_before_any_public_release",
                "words": row["words"],
                "segments": row["segments"],
                "pos_tags": row["pos_tags"],
                "source_fold": row.get("fold"),
                "source_subfold": row.get("subfold"),
            })
    return result


def load_lahga() -> list[dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset(LAHGA_SOURCE, "entries", split="train")
    target_groups = {"egyptian", "levantine", "gulf", "maghrebi"}
    result: list[dict[str, Any]] = []
    for row_number, row in enumerate(dataset):
        group = str(row.get("dialect_group") or "").lower()
        if group not in target_groups:
            continue
        result.append({
            "schema": "ansg.external_lexicon_candidate/1",
            "resource_id": "lahga_arabic_dialects",
            "source_url": "https://huggingface.co/datasets/lahga/arabic-dialects",
            "source_license": LAHGA_LICENSE,
            "attribution": "Lahga, snapshot 2026-10-02",
            "source_row_id": str(row.get("entry_id", row_number)),
            "headword_msa": row.get("headword"),
            "dialect_form": row.get("form"),
            "dialect": row.get("dialect"),
            "dialect_group": group,
            "meaning": row.get("meaning"),
            "notes": row.get("notes"),
            "examples": row.get("examples"),
            "source_reference": row.get("source"),
            "needs_review": bool(row.get("needs_review", True)),
            "native_speaker_confirmed": False,
            "lexicon_candidate_only": True,
            "story_training_eligible": False,
            "ckg_eligible": False,
            "share_alike_notice": "Review CC BY-SA 4.0 conditions before redistributing adaptations.",
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/external_dialects"))
    args = parser.parse_args()
    try:
        qcri_rows = load_qcri()
        lahga_rows = load_lahga()
    except ImportError as exc:
        raise SystemExit("Install the optional 'datasets' package to fetch these public datasets.") from exc

    write_jsonl(args.output_dir / "qcri_dialect_auxiliary.jsonl", qcri_rows)
    write_jsonl(args.output_dir / "lahga_lexicon_candidates.jsonl", lahga_rows)
    summary = {
        "schema": "ansg.external_dialect_auxiliary_manifest/1",
        "qcri_rows": len(qcri_rows),
        "lahga_candidate_rows": len(lahga_rows),
        "story_training_started": False,
        "story_training_eligible": False,
        "ckg_created": False,
        "note": "Auxiliary dialect resources only; review privacy, provenance, speaker validity, and license conditions before wider use.",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
