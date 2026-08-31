"""
Main Prover-Verifier Game training loop, retargeted at AAR research findings.

Mirrors Kirchner et al.'s alternating training structure:
  for each round:
    1. Sample a batch of experiment records.
    2. Helpful prover generates findings for them -> score with current verifier.
    3. Sneaky prover generates misrepresentative findings for the same records
       -> score with current verifier.
    4. Update the verifier on this round's (helpful, sneaky) examples, so it
       adapts to the provers' CURRENT tricks, not last round's.
    5. (Optional, if provers are being trained rather than just prompted)
       update helpful/sneaky provers against the updated verifier.
    6. Log the three core metrics for this round (see eval/metrics.py).

IMPORTANT — read before running:
This is a skeleton with the control flow and logging fully specified, but
the actual model-update calls are left as clearly marked TODOs, because:
  - If your prover is a frozen API model (e.g. Claude, matching how the
    real AAR is used in automated-w2s-research), there IS no prover
    "training" step — only the verifier trains, and the prover's behavior
    only changes via prompt iteration, not gradient updates. This is the
    realistic setup for actually improving a live AAR (see LIMITATIONS.md #5).
  - If instead you have a fine-tunable prover (e.g. a local Qwen model
    playing the prover role for experimentation), step 5 would use a
    real RL or supervised update — wire that in only if you've decided
    to go that route, since it's a meaningfully bigger undertaking.

Run:
    python training/pvg_loop.py --dataset data/findings_pvg_dataset.jsonl \
        --rounds 20
"""
import argparse
import json
import random
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F
from jinja2 import Template
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PVGConfig, RunMetadata
from data.schema import ExperimentRecord, Finding, FindingLabel
from eval.metrics import (
    compute_helpful_prover_accuracy,
    compute_verifier_robustness,
)


def load_dataset(path: str) -> List[dict]:
    records = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Local Qwen verifier: lazily loaded once, reused for scoring + training.
# No Anthropic API calls anywhere in this module.
# ---------------------------------------------------------------------------
_VERIFIER_CACHE = {}


def _get_verifier(verifier_model: str):
    if verifier_model not in _VERIFIER_CACHE:
        tokenizer = AutoTokenizer.from_pretrained(verifier_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(
            verifier_model, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=PVGConfig().verifier_lr)
        _VERIFIER_CACHE[verifier_model] = {
            "model": model, "tokenizer": tokenizer, "device": device, "optimizer": optimizer,
        }
    return _VERIFIER_CACHE[verifier_model]


_VERIFIER_TEMPLATE_PATH = Path(__file__).parent.parent / "prompts" / "verifier_prompt.jinja2"


def _render_verifier_prompt(summary: str, metrics: dict, config: dict) -> str:
    template = Template(_VERIFIER_TEMPLATE_PATH.read_text())
    return template.render(summary=summary, metrics=json.dumps(metrics), config=json.dumps(config))


def _parse_verdict(text: str) -> Optional[float]:
    normalized = text.strip().upper()
    if "UNSOUND" in normalized:
        return 0.0
    if "SOUND" in normalized:
        return 1.0
    return None


def score_with_verifier(summary: str, metrics: dict, config: dict, verifier_model: str) -> float:
    """
    Returns the verifier's predicted P(sound) for a finding, using the local
    Qwen verifier model (no API calls). Generates a short completion and
    parses SOUND/UNSOUND, matching share_finding_gate.py's parsing logic
    so the trained verifier here and the deployed gate agree on format.
    """
    v = _get_verifier(verifier_model)
    prompt = _render_verifier_prompt(summary, metrics, config)
    inputs = v["tokenizer"](prompt, return_tensors="pt").to(v["device"])
    with torch.no_grad():
        out = v["model"].generate(
            **inputs, max_new_tokens=20, do_sample=False,
            pad_token_id=v["tokenizer"].pad_token_id,
        )
    completion = v["tokenizer"].decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    verdict = _parse_verdict(completion)
    # Unparseable output (model didn't say SOUND/UNSOUND) counts as a miss —
    # treat as "rejected" rather than silently defaulting to accepted.
    return verdict if verdict is not None else 0.0


def train_verifier_on_round(
    helpful_findings: List[dict],
    sneaky_findings: List[dict],
    cfg: PVGConfig,
) -> None:
    """
    One supervised update of the local verifier: standard causal-LM loss,
    teaching it to produce "SOUND" after helpful-prover write-ups and
    "UNSOUND" after sneaky-prover write-ups from THIS round (i.e. against
    the provers' current, evolving behavior — not a frozen dataset).
    """
    v = _get_verifier(cfg.verifier_model)
    model, tokenizer, device, optimizer = v["model"], v["tokenizer"], v["device"], v["optimizer"]

    examples = (
        [(item, "SOUND") for item in helpful_findings] +
        [(item, "UNSOUND") for item in sneaky_findings]
    )
    if not examples:
        return
    random.shuffle(examples)

    model.train()
    optimizer.zero_grad()
    total_loss = torch.tensor(0.0, device=device)

    for item, label in examples:
        record = item["record"]
        prompt = _render_verifier_prompt(item["summary"], record.get("metrics", {}), record.get("config", {}))
        full_text = prompt + " " + label
        full_ids = tokenizer(full_text, return_tensors="pt")["input_ids"].to(device)
        prompt_len = tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]

        logits = model(full_ids).logits[:, :-1, :]
        targets = full_ids[:, 1:]
        loss_per_token = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="none"
        ).reshape(targets.shape)
        # Only train on the label tokens (SOUND/UNSOUND), not the whole prompt.
        label_loss = loss_per_token[:, prompt_len - 1:].mean()
        total_loss = total_loss + label_loss

    total_loss = total_loss / len(examples)
    total_loss.backward()
    optimizer.step()
    model.eval()
    print(f"  [verifier update] mean_loss={total_loss.item():.4f} on {len(examples)} examples")


def generate_helpful_finding(record: ExperimentRecord, cfg: PVGConfig) -> str:
    """
    Only used when cfg.use_finetunable_prover=False (frozen API-prover mode,
    e.g. gating the real Claude-based AAR — see LIMITATIONS.md #5). NOT
    called in the current Qwen-only default (use_finetunable_prover=True),
    which instead samples from the local prover via train_prover_step.py's
    generate_group(). Left unwired: wire this to an Anthropic call only if
    you switch back to frozen-API mode.
    """
    raise NotImplementedError("Wire generate_helpful_finding() to your prover model call.")


def generate_sneaky_finding(record: ExperimentRecord, cfg: PVGConfig) -> str:
    """
    Only used when cfg.use_finetunable_prover=False. See
    generate_helpful_finding()'s docstring above — same applies here.
    """
    raise NotImplementedError("Wire generate_sneaky_finding() to your prover model call.")


def run_pvg_training(dataset_path: str, cfg: PVGConfig) -> List[RunMetadata]:
    dataset = load_dataset(dataset_path)
    honest_records = [d for d in dataset if d["label"] == FindingLabel.HONEST.value]
    print(f"Loaded {len(honest_records)} honest ground-truth records to sample experiments from.")

    round_history: List[RunMetadata] = []
    random.seed(cfg.seed)

    # Two modes (see config.py's use_finetunable_prover docstring):
    #   False -> prover is a frozen API model, prompted differently for each
    #            role. Only the verifier trains. Claim: "gates the real AAR."
    #   True  -> prover is a local, fine-tunable stand-in model. BOTH roles
    #            and the verifier train, via GRPO-style updates. Claim:
    #            "PVG training makes a researcher-like stand-in model's
    #            findings more legible" — NOT a claim about the real AAR
    #            (see LIMITATIONS.md #7).
    helpful_prover_state, sneaky_prover_state = None, None
    if cfg.use_finetunable_prover:
        from training.train_prover_step import load_prover, generate_group, train_prover_role
        helpful_prover_state = load_prover(cfg, cfg.prover_model)
        sneaky_prover_state = load_prover(cfg, cfg.prover_model)  # separate adapter, specializes differently

    for round_num in range(1, cfg.num_rounds + 1):
        print(f"\n=== PVG Round {round_num}/{cfg.num_rounds} ===")

        batch = random.sample(honest_records, min(cfg.findings_per_round, len(honest_records)))

        helpful_scored, sneaky_scored = [], []

        if cfg.use_finetunable_prover:
            # --- Fine-tunable stand-in mode: sample from the CURRENT policy, ---
            # --- then update it toward what the CURRENT verifier accepts.    ---
            from jinja2 import Template
            helpful_tmpl = Template(open("prompts/helpful_prover_prompt.jinja2").read())
            sneaky_tmpl = Template(open("prompts/sneaky_prover_prompt.jinja2").read())

            def verifier_reward(summary, record_dict):
                record = ExperimentRecord(**record_dict)
                return score_with_verifier(summary, record.metrics, record.config, cfg.verifier_model)

            helpful_prompts = [{
                "prompt": helpful_tmpl.render(metrics=item["experiment"]["metrics"], config=item["experiment"]["config"]),
                "record": item["experiment"],
            } for item in batch]
            sneaky_prompts = [{
                "prompt": sneaky_tmpl.render(metrics=item["experiment"]["metrics"], config=item["experiment"]["config"]),
                "record": item["experiment"],
            } for item in batch]

            helpful_stats = train_prover_role(helpful_prover_state, helpful_prompts, verifier_reward, cfg)
            sneaky_stats = train_prover_role(sneaky_prover_state, sneaky_prompts, verifier_reward, cfg)
            print(f"  [prover update] helpful mean_reward={helpful_stats['mean_reward']:.3f} "
                  f"sneaky mean_reward={sneaky_stats['mean_reward']:.3f}")

            # Re-sample post-update completions for this round's verifier training data.
            for item in helpful_prompts:
                summary = generate_group(helpful_prover_state, item["prompt"], cfg)[0]
                score = verifier_reward(summary, item["record"])
                helpful_scored.append({"summary": summary, "score": score, "record": item["record"]})
            for item in sneaky_prompts:
                summary = generate_group(sneaky_prover_state, item["prompt"], cfg)[0]
                score = verifier_reward(summary, item["record"])
                sneaky_scored.append({"summary": summary, "score": score, "record": item["record"]})
        else:
            # --- Frozen API mode: prover weights never change; only prompting varies. ---
            for item in batch:
                record = ExperimentRecord(**item["experiment"])

                helpful_summary = generate_helpful_finding(record, cfg)
                helpful_score = score_with_verifier(helpful_summary, record.metrics, record.config, cfg.verifier_model)
                helpful_scored.append({"summary": helpful_summary, "score": helpful_score, "record": item["experiment"]})

                sneaky_summary = generate_sneaky_finding(record, cfg)
                sneaky_score = score_with_verifier(sneaky_summary, record.metrics, record.config, cfg.verifier_model)
                sneaky_scored.append({"summary": sneaky_summary, "score": sneaky_score, "record": item["experiment"]})

        # Update the verifier on this round's fresh, current-prover-behavior examples.
        train_verifier_on_round(helpful_scored, sneaky_scored, cfg)

        # Compute and log this round's metrics.
        helpful_acc = compute_helpful_prover_accuracy(helpful_scored)
        verifier_rob = compute_verifier_robustness(sneaky_scored)

        metadata = RunMetadata(
            round_number=round_num,
            helpful_prover_accuracy=helpful_acc,
            verifier_robustness=verifier_rob,
        )
        round_history.append(metadata)
        print(f"  helpful_prover_accuracy={helpful_acc:.3f}  verifier_robustness={verifier_rob:.3f}")

    return round_history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/findings_pvg_dataset.jsonl")
    parser.add_argument("--rounds", type=int, default=None)
    args = parser.parse_args()

    cfg = PVGConfig()
    if args.rounds:
        cfg.num_rounds = args.rounds

    history = run_pvg_training(args.dataset, cfg)

    out_path = Path("training/pvg_round_history.jsonl")
    with open(out_path, "w") as f:
        for m in history:
            f.write(json.dumps(vars(m)) + "\n")
    print(f"\nRound-by-round history written to {out_path}")
