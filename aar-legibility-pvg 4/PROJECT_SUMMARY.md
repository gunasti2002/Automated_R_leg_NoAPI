# AAR Legibility via Prover-Verifier Games — Summary

## Problem
Automated Alignment Researchers (AARs) can produce findings that look
rigorous without being well-supported — optimizing for "does this
result look good" can degrade legibility, the same failure Kirchner et
al. (2024) found in math CoT solutions. We need a way to catch AAR
findings that misrepresent their own data.

## Method
Apply Kirchner et al.'s Prover-Verifier Game (arXiv:2407.13692) to AAR
research findings instead of math solutions. Three roles, trained in
alternation:
- **Helpful prover** — writes accurate findings the verifier accepts
- **Sneaky prover** — writes findings that misrepresent the data but
  look plausible enough to fool the verifier
- **Verifier** — a small model trained each round to tell them apart

Ground truth is constructed by checking internal consistency: does a
finding's narrative match its own cited metrics/config. The trained
verifier is then used as a gate on the real AAR's `share_finding` tool.

## Files
| File | Purpose |
|---|---|
| `README.md` / `LIMITATIONS.md` | Design + honest scope gaps |
| `config.py` | All tunable settings |
| `data/schema.py` | `ExperimentRecord` / `Finding` data model |
| `data/build_findings_dataset.py` | Working consistency checker + sneaky-example generation (stubbed) |
| `prompts/*.jinja2` | Helpful / sneaky / verifier prompts |
| `training/pvg_loop.py` | Alternating training loop (model calls stubbed) |
| `eval/metrics.py` | The three metrics below, fully implemented |
| `eval/human_eval_harness.py` | Working blinded human-review CLI |
| `integration/share_finding_gate.py` | Patch for the real AAR's submission gate |
| `scripts/*.sh` | End-to-end run scripts |

## Metrics
1. **Helpful prover accuracy** — % of honest findings the verifier accepts
2. **Verifier robustness** — % of sneaky findings the verifier correctly rejects (should rise across rounds)
3. **Human checking accuracy** — the metric that actually proves the claim: do humans get better at judging helpful findings and worse at judging sneaky ones, before vs. after training

Metrics 1–2 only show the automated game converged; only metric 3 shows legibility improved for a human.

## Next steps
1. **Decide scope**: gate-only on the real (frozen) AAR, or fine-tune a stand-in open-weights model — the real AAR can't be fine-tuned directly
2. Export real AAR findings into `data/source_findings/`
3. Wire the 5 stubbed model-call functions to your LLM clients
4. Validate small-scale by hand before trusting any metric
5. Run a real multi-reviewer human-eval study, before and after training
6. If gating: patch `share_finding`, start with a conservative threshold
7. If fine-tuning: scope the stand-in AAR model as its own sub-project
