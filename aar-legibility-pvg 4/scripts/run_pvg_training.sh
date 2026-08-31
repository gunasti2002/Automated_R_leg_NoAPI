#!/usr/bin/env bash
# Full PVG training pipeline, end to end.
#
# Prerequisites:
#   1. Export real AAR findings into data/source_findings/ (one JSON file
#      per finding, matching automated-w2s-research's findings_sync.py format).
#   2. Wire the TODOs in data/build_findings_dataset.py (generate_sneaky_counterpart)
#      and training/pvg_loop.py (score_with_verifier, train_verifier_on_round,
#      generate_helpful_finding, generate_sneaky_finding) to your actual model
#      clients — this script will fail loudly at those points until you do.

set -euo pipefail

SOURCE_DIR="${SOURCE_DIR:-data/source_findings}"
DATASET_OUT="${DATASET_OUT:-data/findings_pvg_dataset.jsonl}"
ROUNDS="${ROUNDS:-20}"

echo "=== [1/2] Building PVG training dataset from ${SOURCE_DIR} ==="
python data/build_findings_dataset.py --source-dir "${SOURCE_DIR}" --out "${DATASET_OUT}"

echo ""
echo "=== [2/2] Running PVG training loop (${ROUNDS} rounds) ==="
python training/pvg_loop.py --dataset "${DATASET_OUT}" --rounds "${ROUNDS}"

echo ""
echo "Done. Round-by-round metrics: training/pvg_round_history.jsonl"
echo "Next: run scripts/run_eval.sh to build a human-eval batch from the"
echo "final round's findings and get the real headline metric."
