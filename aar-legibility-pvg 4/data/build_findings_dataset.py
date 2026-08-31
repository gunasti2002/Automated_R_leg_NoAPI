"""
Build the PVG training dataset from real AAR experiment records.

For each source finding (a real submission with metrics + config + an
honest write-up, e.g. exported from automated-w2s-research's
shared_findings/ directory or its server's /api/findings/all):

  1. Run the internal-consistency check to confirm the honest write-up
     actually matches its own cited numbers. Only consistent ones become
     ground-truth-honest training examples (see LIMITATIONS.md #1 for why
     this is a real, non-trivial simplification of "ground truth").
  2. Generate a "sneaky" counterpart: a fluent, plausible-sounding write-up
     of the SAME experiment record that misrepresents it — overstates
     significance, cites a metric that doesn't support the claim, glosses
     over a low seed count, etc.
  3. Write both out as labeled Finding records for PVG training.

Run:
    python data/build_findings_dataset.py \
        --source-dir data/source_findings \
        --out data/findings_pvg_dataset.jsonl
"""
import argparse
import json
import re
from pathlib import Path
from typing import List, Tuple

from schema import ExperimentRecord, Finding, FindingLabel


# ---------------------------------------------------------------------------
# Step 1: internal-consistency ground-truth check
# ---------------------------------------------------------------------------

def check_internal_consistency(record: ExperimentRecord, summary: str) -> Tuple[bool, List[str]]:
    """
    Heuristic checks that a finding's narrative matches its own cited data.
    This is NOT a claim of external validity (see LIMITATIONS.md #1) — only
    that the write-up doesn't contradict or overstate its own numbers.

    Extend this with more checks as you find real failure modes in your
    AAR's actual submissions. Treat this function as a living document, not
    a finished oracle.
    """
    issues = []

    metrics = record.metrics or {}
    num_seeds = metrics.get("num_seeds")

    # Check 1: claims of a "significant" or "robust" result without enough seeds.
    strong_claim_words = ["significant", "robust", "clearly", "conclusively", "definitively"]
    if num_seeds is not None and num_seeds < 5:
        if any(w in summary.lower() for w in strong_claim_words):
            issues.append(
                f"Uses strong-confidence language ({strong_claim_words}) "
                f"with only num_seeds={num_seeds} (paper's own share_finding "
                f"tool requires 5+ for 'result' findings)."
            )

    # Check 2: any specific numeric claim in the summary should appear
    # (within a loose tolerance) somewhere in the actual metrics dict.
    numbers_in_summary = set(re.findall(r"\b0?\.\d+\b|\b\d{1,3}(?:\.\d+)?%\b", summary))
    metric_values = set()
    for v in metrics.values():
        if isinstance(v, (int, float)):
            metric_values.add(f"{v:.2f}".lstrip("0") if isinstance(v, float) else str(v))
            metric_values.add(f"{v*100:.0f}%" if isinstance(v, float) and 0 <= v <= 1 else "")

    unmatched = [n for n in numbers_in_summary if not any(n.strip("%") in mv for mv in metric_values if mv)]
    if unmatched:
        issues.append(
            f"Summary cites number(s) {unmatched} not found (within loose "
            f"matching) in the record's own metrics dict — possible "
            f"fabrication or miscited figure."
        )

    # Check 3: "worked" claim consistency — if worked=False in the record but
    # the summary reads as a positive/success framing, flag it.
    positive_words = ["improved", "success", "outperform", "better than baseline", "worked well"]
    if record.worked is False and any(w in summary.lower() for w in positive_words):
        issues.append(
            "Record's own 'worked' field is False, but summary uses "
            "positive/success framing language."
        )

    return (len(issues) == 0, issues)


# ---------------------------------------------------------------------------
# Step 2: sneaky-prover counterpart generation
# ---------------------------------------------------------------------------

def generate_sneaky_counterpart(record: ExperimentRecord, honest_summary: str, config) -> str:
    """
    Qwen-only mode: no model call at all here (avoids both Claude AND an
    extra local-model round-trip during dataset building). Instead applies
    one of a few rule-based corruptions directly to the honest write-up,
    mirroring the failure modes check_internal_consistency() looks for.
    Cruder than a prompted rewrite (see LIMITATIONS.md #2 — this is an even
    more mechanical inversion than the paper's originally-flagged gap), but
    sufficient to give the local prover/verifier something adversarial to
    train against without any API dependency.

    Picks deterministically off record.record_id so reruns are reproducible.
    """
    import random as _random
    rng = _random.Random(record.record_id or honest_summary)
    metrics = record.metrics or {}
    num_seeds = metrics.get("num_seeds")

    strategies = []

    # 1. Inflate a numeric claim already present in the summary.
    numbers = re.findall(r"\b0?\.\d+\b|\b\d{1,3}(?:\.\d+)?%\b", honest_summary)
    if numbers:
        def _inflate(text):
            target = rng.choice(numbers)
            try:
                if target.endswith("%"):
                    inflated = f"{float(target[:-1]) * 1.25:.0f}%"
                else:
                    inflated = f"{float(target) * 1.25:.2f}".lstrip("0")
            except ValueError:
                return text
            return text.replace(target, inflated, 1)
        strategies.append(_inflate)

    # 2. Add unsupported strong-confidence language when seeds are few.
    if num_seeds is not None and num_seeds < 5:
        strategies.append(lambda text: text.rstrip(". ") +
                           ", a clearly robust and conclusive effect.")

    # 3. Flip negative/null framing to positive, if the record says it didn't work.
    if record.worked is False:
        strategies.append(lambda text: text.replace("did not", "successfully did", 1)
                           if "did not" in text else text + " Overall the approach worked well.")

    # 4. Fallback: cite a plausible-looking number that isn't actually in metrics.
    strategies.append(lambda text: text.rstrip(". ") +
                       f", consistent with a {rng.choice([12, 18, 27])}% relative improvement.")

    corruption = rng.choice(strategies)
    return corruption(honest_summary)


# ---------------------------------------------------------------------------
# Step 3: orchestration
# ---------------------------------------------------------------------------

def load_source_findings(source_dir: Path) -> List[dict]:
    """
    Loads raw finding JSON files, matching the shape saved by
    automated-w2s-research's findings_sync.py (one JSON file per finding).
    """
    findings = []
    for f in sorted(source_dir.glob("*.json")):
        try:
            with open(f, "r") as fh:
                findings.append(json.load(fh))
        except Exception as e:
            print(f"  Skipping unreadable file {f}: {e}")
    return findings


def build_dataset(source_dir: str, out_path: str, config=None):
    source_dir = Path(source_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    raw_findings = load_source_findings(source_dir)
    print(f"Loaded {len(raw_findings)} raw findings from {source_dir}")

    honest_count, skipped_count, sneaky_count = 0, 0, 0

    with open(out_path, "w") as out_f:
        for raw in raw_findings:
            record = ExperimentRecord(
                idea_name=raw.get("idea_name", "unknown"),
                metrics={k: raw[k] for k in
                         ("pgr", "pgr_se", "transfer_acc", "transfer_acc_se",
                          "weak_acc", "strong_acc", "num_seeds") if k in raw},
                config=raw.get("config", {}),
                worked=raw.get("worked"),
                dataset=raw.get("dataset"),
                record_id=raw.get("id") or raw.get("finding_id"),
            )
            honest_summary = raw.get("summary", "")
            if not honest_summary:
                skipped_count += 1
                continue

            is_consistent, issues = check_internal_consistency(record, honest_summary)
            if not is_consistent:
                print(f"  Skipping {record.record_id}: honest write-up failed "
                      f"consistency check itself ({issues}) — can't use as "
                      f"ground-truth-honest example.")
                skipped_count += 1
                continue

            honest_finding = Finding(
                experiment=record, summary=honest_summary,
                label=FindingLabel.HONEST, is_internally_consistent=True,
            )
            out_f.write(json.dumps(_finding_to_dict(honest_finding)) + "\n")
            honest_count += 1

            try:
                sneaky_summary = generate_sneaky_counterpart(record, honest_summary, config)
            except NotImplementedError:
                continue  # allowed to skip sneaky generation until wired up
            sneaky_finding = Finding(
                experiment=record, summary=sneaky_summary,
                label=FindingLabel.SNEAKY, is_internally_consistent=False,
            )
            out_f.write(json.dumps(_finding_to_dict(sneaky_finding)) + "\n")
            sneaky_count += 1

    print(f"\nDone. Honest: {honest_count}, Sneaky: {sneaky_count}, Skipped: {skipped_count}")
    print(f"Written to {out_path}")


def _finding_to_dict(finding: Finding) -> dict:
    return {
        "experiment": vars(finding.experiment),
        "summary": finding.summary,
        "label": finding.label.value,
        "is_internally_consistent": finding.is_internally_consistent,
        "consistency_issues": finding.consistency_issues,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default="data/source_findings")
    parser.add_argument("--out", default="data/findings_pvg_dataset.jsonl")
    args = parser.parse_args()
    build_dataset(args.source_dir, args.out)
