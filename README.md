# ANSG Auxiliary Dialect Resources

This repository contains **resource pointers, preparation code, and a small DIA baseline**. It does not contain story texts or copies of the upstream datasets. The runtime loads source data directly from Hugging Face so source attribution and licenses remain attached.

## Scope

- **QCRI Arabic POS Dialect**: 350 annotated tweets for each of Egyptian, Levantine, Gulf, and Maghrebi Arabic. The source card lists Apache-2.0 and POS/segmentation fields. Its bias and limitations sections are marked as incomplete. The data contains social-media text, so the scripts keep downloaded rows in the runtime workspace and do not publish them here.
- **Lahga Arabic Dialects**: a CC BY-SA 4.0 dictionary snapshot (2026-10-02), with dialect forms and examples. The publisher reports 3,606 forms as not yet speaker-confirmed. The preparation script retains `needs_review` and source fields. Any adapted Lahga-derived resource/model must be reviewed for the publisher's share-alike notice.
- **MADAR** is deliberately excluded from this repository and from downloads: its terms allow internal research/evaluation only and prohibit redistribution/modification.

These are auxiliary dialect/POS/lexicon resources. They are **not ANHC story data**, are not merged into the narrative corpus, and do not train the ANSG story generator. The DIA baseline is a small tweet-domain classification experiment, not a thesis result or a measure of story-generation quality.

## Run the auxiliary DIA experiment

```bash
python -m pip install -r requirements.txt
python scripts/train_qcri_dialect_id.py --output-dir outputs/qcri_dia
```

The script uses the QCRI dataset's published five-fold IDs for out-of-fold accuracy and macro-F1, then fits a final char-TF-IDF + LinearSVC baseline on all 1,400 rows. Outputs are `metrics.json` and a local `qcri_dia_baseline.joblib`; do not commit them with tweet text.

## Prepare local auxiliary files

```bash
python scripts/prepare_external_dialect_auxiliary.py --output-dir data/external_dialects
```

The script writes QCRI rows and Lahga lexicon candidates locally with source/license metadata. It does not add them to Git history. Review row-level provenance, QCRI privacy considerations, Lahga `needs_review`, and share-alike obligations before any wider distribution.

## Sources

- QCRI dataset and card: https://huggingface.co/datasets/QCRI/arabic_pos_dialect
- QCRI citation: Darwish et al. (2018), “Multi-Dialect Arabic POS Tagging: A CRF Approach,” LREC 2018.
- Lahga dataset and card: https://huggingface.co/datasets/lahga/arabic-dialects
- Lahga source site: https://lahga.fyi/
- MADAR access and terms: https://camel.abudhabi.nyu.edu/madar-parallel-corpus/

No single repository-level license is applied because upstream components have different terms. Use the source-specific notices above.
