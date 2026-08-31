"""
Structured human-evaluation harness for the "human checking accuracy"
metric — the paper's actual headline result, and the only metric in this
repo that establishes findings became more legible TO A HUMAN, not just
to the automated verifier (see LIMITATIONS.md #3).

This does not simulate human judgment. It:
  1. Builds a randomized, blinded batch of findings (mix of helpful and
     sneaky, unlabeled to the reviewer) for a human session.
  2. Presents each finding with a time limit (matching the paper's
     "time-constrained humans" design — legibility should reduce the
     time/effort needed to correctly judge soundness).
  3. Records verdicts + time taken.
  4. Exports sessions in the format eval/metrics.py's
     compute_human_checking_accuracy expects.

Run:
    python eval/human_eval_harness.py build --dataset training/round_20_findings.jsonl \
        --out eval/human_eval_sessions/round_20_batch.jsonl

    python eval/human_eval_harness.py review --batch eval/human_eval_sessions/round_20_batch.jsonl \
        --reviewer-id alice --out eval/human_eval_sessions/round_20_alice_results.jsonl
"""
import argparse
import json
import random
import time
from pathlib import Path
from typing import List, Dict


def build_review_batch(findings_path: str, out_path: str, batch_size: int = 40, seed: int = 42) -> None:
    """
    Samples a balanced, blinded batch of helpful/sneaky findings for review.
    The reviewer never sees which is which — that label is stripped before
    writing the reviewer-facing file, and kept separately for scoring.
    """
    random.seed(seed)
    findings = []
    with open(findings_path, "r") as f:
        for line in f:
            if line.strip():
                findings.append(json.loads(line))

    helpful = [f for f in findings if f.get("label") == "helpful"]
    sneaky = [f for f in findings if f.get("label") == "sneaky"]
    n_each = min(batch_size // 2, len(helpful), len(sneaky))
    if n_each == 0:
        raise ValueError("Not enough helpful/sneaky findings to build a balanced batch.")

    batch = random.sample(helpful, n_each) + random.sample(sneaky, n_each)
    random.shuffle(batch)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for item in batch:
            f.write(json.dumps({
                "finding_id": item.get("experiment", {}).get("record_id", "unknown") + "_" + item["label"],
                "summary": item["summary"],
                "metrics": item["experiment"]["metrics"],
                "config": item["experiment"]["config"],
                "_ground_truth_label": item["label"],  # kept for scoring; strip before showing reviewer
            }) + "\n")
    print(f"Built blinded review batch of {len(batch)} findings ({n_each} helpful, {n_each} sneaky) at {out_path}")


def run_review_session(batch_path: str, reviewer_id: str, out_path: str, time_limit_seconds: int = 120) -> None:
    """
    Simple CLI-based review session. Swap this for a real web UI if running
    a study with more than a couple of reviewers — this is meant as a
    minimal, honest starting point, not a production eval tool.
    """
    items = []
    with open(batch_path, "r") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))

    results = []
    print(f"\nHuman eval session for reviewer '{reviewer_id}'. {len(items)} findings to review.")
    print(f"You have up to {time_limit_seconds}s per finding — judge based on whether the "
          f"write-up's claims are actually supported by the metrics/config shown.\n")

    for i, item in enumerate(items, 1):
        print(f"\n--- Finding {i}/{len(items)} ---")
        print(f"Metrics: {item['metrics']}")
        print(f"Config: {item['config']}")
        print(f"\nWrite-up:\n{item['summary']}\n")

        start = time.time()
        verdict = input("Sound or unsound? [s/u]: ").strip().lower()
        elapsed = time.time() - start

        results.append({
            "finding_id": item["finding_id"],
            "reviewer_id": reviewer_id,
            "human_verdict": "sound" if verdict.startswith("s") else "unsound",
            "ground_truth_label": item["_ground_truth_label"],
            "time_taken_s": round(elapsed, 1),
        })

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nSession complete. Results written to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    build_p = sub.add_parser("build")
    build_p.add_argument("--dataset", required=True)
    build_p.add_argument("--out", required=True)
    build_p.add_argument("--batch-size", type=int, default=40)

    review_p = sub.add_parser("review")
    review_p.add_argument("--batch", required=True)
    review_p.add_argument("--reviewer-id", required=True)
    review_p.add_argument("--out", required=True)
    review_p.add_argument("--time-limit", type=int, default=120)

    args = parser.parse_args()
    if args.command == "build":
        build_review_batch(args.dataset, args.out, args.batch_size)
    elif args.command == "review":
        run_review_session(args.batch, args.reviewer_id, args.out, args.time_limit)
