# AAR Legibility via Prover-Verifier Games

## Goal

Apply Kirchner et al.'s Prover-Verifier Game (PVG) method
(https://arxiv.org/abs/2407.13692) directly to an Automated Alignment
Researcher's (AAR) own research output — findings, hypotheses, result
write-ups — instead of to chain-of-thought math solutions (the original
paper's domain) or to chat-answer critiques (an earlier, now-abandoned
direction in this project).

The concrete question this repo is built to answer:

    Does training a "verifier" to distinguish sound AAR findings from
    unsound-but-plausible-looking ones, and training the AAR (the
    "prover") against that verifier, produce AAR research output that
    is more checkable — without making the AAR's actual findings less
    useful or less frequently correct?

This is a full redesign, not an extension of the earlier
critique-based `critic_legibility` work in this project. That work
(training a critic to write persuasive-vs-legible critiques of chat
answers, scored via a single blended GRPO reward) is set aside
entirely. Nothing here depends on it.

## Why the original paper's method, not the simplified blend

The earlier direction used one scalar reward:
`reward = alpha*task_reward + (1-alpha)*legibility_score`. That is a
simplification of PVG, not PVG itself — it has no adversarial
component, so the verifier is never stress-tested against an opponent
actively trying to fool it. Kirchner et al.'s actual result rests on
three roles trained in alternation:

- **Helpful prover** — produces solutions/findings that are correct
  AND that the verifier accepts.
- **Sneaky prover** — produces solutions/findings that are INCORRECT
  but constructed to fool the verifier into accepting them anyway.
- **Verifier** — a small model trained to predict correctness, updated
  against both provers' current behavior.

This repo implements that three-role structure, retargeted at research
findings instead of math solutions.

## Domain mapping (math → AAR findings)

| Kirchner et al. (math CoT)              | This repo (AAR findings)                                   |
|------------------------------------------|--------------------------------------------------------------|
| Grade-school math problem                | A completed experiment (metrics + config already known)      |
| Chain-of-thought solution                | A research finding write-up (summary, claimed result)         |
| Ground-truth correct/incorrect answer    | Whether the finding's claims are actually supported by its own cited metrics/config |
| Helpful prover                           | AAR producing an accurate, well-supported finding             |
| Sneaky prover                            | AAR (or a trained adversary) producing a finding that misrepresents, overstates, or fabricates support for its claim |
| Small verifier                           | A model trained to accept/reject a finding as sound            |
| Human checking accuracy                  | Human reviewers' accuracy judging findings, before vs. after training |

## What "ground truth" means here

Kirchner et al. have an unambiguous ground truth: the math answer is
right or wrong. AAR findings don't have that automatically — a
finding is a claim ABOUT data that already exists (metrics, configs,
logs from an experiment run). We construct ground truth by checking
internal consistency: does the finding's narrative claim match what
its own cited numbers actually show. See `data/build_findings_dataset.py`
for exactly how this is operationalized, and `LIMITATIONS.md` for why
this is a real, non-trivial simplification worth being upfront about.

## Repo layout

```
aar-legibility-pvg/
  README.md                        - this file
  LIMITATIONS.md                   - honest scope and known gaps
  config.py                        - all tunable settings
  requirements.txt
  data/
    schema.py                      - the Finding / ExperimentRecord data model
    build_findings_dataset.py      - constructs (experiment, honest write-up,
                                      fabricated write-up) training pairs
  prompts/
    helpful_prover_prompt.jinja2
    sneaky_prover_prompt.jinja2
    verifier_prompt.jinja2
  training/
    pvg_loop.py                    - the main iterative training loop
    train_verifier_step.py         - one verifier update
    train_prover_step.py           - one prover update (helpful or sneaky)
  eval/
    metrics.py                     - the three core metrics (see below)
    human_eval_harness.py          - structured human-eval CLI/export
  integration/
    share_finding_gate.py          - hook the trained verifier into the
                                      AAR's existing share_finding tool
  scripts/
    run_pvg_training.sh
    run_eval.sh
```

## Metrics (directly mirroring the paper)

1. **Helpful prover accuracy** — of findings the helpful prover
   produces, what fraction does the verifier accept, and (separately,
   the metric that actually matters) what fraction are ACTUALLY
   correct per the ground-truth check.
2. **Verifier robustness** — of findings the sneaky prover produces
   (which are, by construction, unsupported/misrepresented), what
   fraction does the verifier correctly reject. Tracked over training
   rounds — should increase as the verifier adapts to the sneaky
   prover's current tricks.
3. **Human checking accuracy** — time-boxed human reviewers judge a
   mixed batch of helpful-prover and sneaky-prover findings as
   sound/unsound, WITHOUT seeing the verifier's own verdict. Tracked
   at the start and end of training. The paper's headline result is
   that this rises for helpful-prover output and falls for
   sneaky-prover output over training — this repo's eval harness is
   built to produce that same comparison, but it requires you to
   actually recruit and run reviewers; nothing here fakes it
   automatically. See `eval/human_eval_harness.py`.

## What "proving legibility improved research quality" requires

To make that claim rigorously you need, at minimum:
- Metric 1 showing helpful-prover accuracy holds or improves (legibility
  training didn't come at the cost of usefulness)
- Metric 2 showing verifier robustness increases over rounds (the
  verifier is actually getting harder to fool, not just static)
- Metric 3 (the real payoff metric) showing human accuracy rises on
  helpful output and falls on sneaky output post-training — this is
  the part that actually says "legibility improved," since 1 and 2
  are about the automated game, not about whether humans (the actual
  target of legibility) benefit

Skipping metric 3 and only reporting 1 and 2 would not, by itself,
establish that AAR research became more legible to a human reader —
only that the automated verifier game converged. Say this explicitly
in any write-up.
