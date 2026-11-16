#!/usr/bin/env python3
"""Train a small, reproducible dialect-ID baseline on QCRI POS tweets.

This is an auxiliary DIA experiment only. It is not ANSG story-model training,
does not use ANHC, and must not be reported as thesis story-generation results.
The QCRI dataset card identifies the examples as tweets and its bias/limitations
sections are incomplete; keep fetched text in the private runtime workspace.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.svm import LinearSVC


DATASET = "QCRI/arabic_pos_dialect"
CONFIGS = (("egy", "Egyptian"), ("lev", "Levantine"),
           ("glf", "Gulf"), ("mgr", "Maghrebi"))


def read_rows():
    texts: list[str] = []
    labels: list[str] = []
    folds: list[int] = []
    for config, dialect in CONFIGS:
        rows = load_dataset(DATASET, config, split="train")
        for row in rows:
            texts.append(" ".join(row["words"]))
            labels.append(dialect)
            folds.append(int(row["fold"]))
    return texts, labels, folds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/qcri_dia"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    texts, labels, folds = read_rows()

    # Respect QCRI's published five-fold assignment for evaluation.
    gold: list[str] = []
    pred: list[str] = []
    for held_out_fold in sorted(set(folds)):
        train_idx = [i for i, fold in enumerate(folds) if fold != held_out_fold]
        test_idx = [i for i, fold in enumerate(folds) if fold == held_out_fold]
        vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 5),
                                     min_df=2, sublinear_tf=True)
        train_x = vectorizer.fit_transform([texts[i] for i in train_idx])
        test_x = vectorizer.transform([texts[i] for i in test_idx])
        classifier = LinearSVC(C=1.0, class_weight="balanced", random_state=42)
        classifier.fit(train_x, [labels[i] for i in train_idx])
        predictions = classifier.predict(test_x)
        gold.extend(labels[i] for i in test_idx)
        pred.extend(str(label) for label in predictions)

    dialects = [name for _, name in CONFIGS]
    summary = {
        "experiment": "auxiliary_qcri_dialect_id_char_tfidf_linear_svc",
        "dataset": DATASET,
        "dataset_license_as_published": "Apache-2.0",
        "dataset_rows": len(texts),
        "evaluation": "out-of-fold using the dataset's published fold IDs",
        "accuracy": accuracy_score(gold, pred),
        "macro_f1": f1_score(gold, pred, labels=dialects, average="macro"),
        "labels": dialects,
        "confusion_matrix_labels_order": dialects,
        "confusion_matrix": confusion_matrix(gold, pred, labels=dialects).tolist(),
        "scope_warning": (
            "Auxiliary tweet-domain dialect identification only; not story generation, "
            "not ANHC training, and not a thesis result. QCRI card notes incomplete "
            "bias/limitations information; review before public release."
        ),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # Fit a deployable baseline on all rows after the cross-fold evaluation.
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 5),
                                 min_df=2, sublinear_tf=True)
    features = vectorizer.fit_transform(texts)
    classifier = LinearSVC(C=1.0, class_weight="balanced", random_state=42)
    classifier.fit(features, labels)
    try:
        import joblib
    except ImportError as exc:
        raise SystemExit("Install scikit-learn and joblib to save the baseline model") from exc
    joblib.dump({"vectorizer": vectorizer, "classifier": classifier,
                 "labels": dialects}, args.output_dir / "qcri_dia_baseline.joblib")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
