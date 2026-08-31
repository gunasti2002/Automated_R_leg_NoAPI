#!/usr/bin/env bash
# Build a human-eval batch and (optionally) run a CLI review session.
#
# The automated metrics (helpful_prover_accuracy, verifier_robustness) are
# already logged during training — see training/pvg_round_history.jsonl.
# This script is for the metric that actually requires a human: run it
# once BEFORE training (on round-0 / untrained-verifier findings) and once
# AFTER training, so you can compare human accuracy pre/post, matching the
# paper's design.

set -euo pipefail

FINDINGS_FILE="${1:?Usage: run_eval.sh <findings.jsonl> <reviewer_id>}"
REVIEWER_ID="${2:?Usage: run_eval.sh <findings.jsonl> <reviewer_id>}"

BATCH_OUT="eval/human_eval_sessions/$(basename "${FINDINGS_FILE}" .jsonl)_batch.jsonl"
RESULTS_OUT="eval/human_eval_sessions/$(basename "${FINDINGS_FILE}" .jsonl)_${REVIEWER_ID}_results.jsonl"

echo "=== Building blinded review batch ==="
python eval/human_eval_harness.py build --dataset "${FINDINGS_FILE}" --out "${BATCH_OUT}"

echo ""
echo "=== Starting review session for reviewer '${REVIEWER_ID}' ==="
python eval/human_eval_harness.py review --batch "${BATCH_OUT}" --reviewer-id "${REVIEWER_ID}" --out "${RESULTS_OUT}"

echo ""
echo "Results written to ${RESULTS_OUT}"
echo "Aggregate across reviewers with eval/metrics.py's compute_human_checking_accuracy()"
echo "once you have enough sessions to trust the numbers (the paper used many reviewers"
echo "per condition — a single session is not a study)."
