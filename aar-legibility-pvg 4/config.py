"""
Central configuration for the AAR legibility PVG pipeline.
Single source of truth — all modules import from here.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PVGConfig:
    # --- Models ---
    # Two modes, chosen by use_finetunable_prover below:
    #   False (default): prover_model is a frozen API model (e.g. Claude) —
    #       matches the real, live AAR. No prover fine-tuning happens; only
    #       the verifier trains (see LIMITATIONS.md #5). Use this if the
    #       claim you want is "we can gate the real AAR's submissions."
    #   True: prover_model is a LOCAL, fine-tunable stand-in model (e.g.
    #       Qwen2.5-7B-Instruct) trained via GRPO against the verifier's
    #       score, mirroring Kirchner et al.'s actual method. Use this if
    #       the claim you want is "PVG training makes a researcher-like
    #       model's findings more legible" — NOTE this is then a claim
    #       about the stand-in model, not about the real Claude-based AAR
    #       (see LIMITATIONS.md #7).
    # Set to True (Qwen-only mode): prover is a LOCAL, LoRA-fine-tunable
    # stand-in model, trained via GRPO in train_prover_step.py. No Claude
    # API calls happen anywhere in pvg_loop.py in this mode.
    use_finetunable_prover: bool = True
    # Small model so a free-tier Colab T4 (16GB) can hold prover + ref copy
    # + verifier at once. Bump to 3B/7B once this runs end-to-end cleanly.
    prover_model: str = "Qwen/Qwen2.5-1.5B-Instruct"

    # The verifier should be small relative to the prover, per the paper's
    # design (a weak verifier that must be genuinely convinced, not a
    # same-capability model that can be talked into agreeing with itself).
    verifier_model: str = "Qwen/Qwen2.5-1.5B-Instruct"

    # --- Fine-tunable prover settings (only used if use_finetunable_prover=True) ---
    prover_lora_r: int = 16
    prover_lora_alpha: int = 16
    prover_lr: float = 1e-5
    prover_num_generations: int = 4       # samples per experiment, per role, per round
                                           # (GRPO-style group for advantage normalization)
    prover_max_new_tokens: int = 512
    prover_temperature: float = 0.9
    prover_kl_penalty: float = 0.02       # penalize drifting too far from the base model

    # --- PVG training loop ---
    num_rounds: int = 20                 # outer alternation rounds
    findings_per_round: int = 64         # (experiment, prover) pairs sampled per round
    verifier_train_epochs_per_round: int = 1
    verifier_lr: float = 1e-5
    verifier_batch_size: int = 16

    # --- Data construction ---
    # Source of real experiment records (metrics + config + an honest
    # write-up) to build training pairs from. Point this at wherever your
    # AAR's actual findings/results are stored (e.g. the automated-w2s-research
    # server's /api/findings/all, or a local export of shared_findings/).
    source_findings_dir: str = "data/source_findings"
    output_dataset_path: str = "data/findings_pvg_dataset.jsonl"

    # Minimum internal-consistency checks a "helpful" finding must pass to be
    # used as ground-truth-honest training data (see data/schema.py).
    require_cited_metrics_match: bool = True
    require_num_seeds_disclosed: bool = True

    # --- Sneaky prover generation ---
    # Only used by data/build_findings_dataset.py's generate_sneaky_counterpart,
    # which is now a local heuristic corruption (no API calls) — see that
    # file. This field is unused while that's the case; left here in case
    # you swap in a real model-based generator later.
    sneaky_generator_model: str = "claude-sonnet-4-6"
    sneaky_generation_temperature: float = 0.9

    # --- Evaluation ---
    human_eval_batch_size: int = 40      # findings shown per human reviewer session
    human_eval_time_limit_seconds: int = 120  # "time-constrained" per the paper's design
    human_eval_output_dir: str = "eval/human_eval_sessions"

    # --- Integration with the live AAR loop ---
    # Verifier acceptance threshold below which a share_finding submission is
    # rejected outright (see integration/share_finding_gate.py). This is a
    # judgment call, not derived from theory — start conservative (low bar,
    # catch only clear misrepresentation) and tighten based on round-over-round
    # verifier robustness once you trust it.
    gate_reject_threshold: float = 0.3

    seed: int = 42


@dataclass
class RunMetadata:
    """Attached to every PVG training round's output for later analysis."""
    round_number: int
    helpful_prover_accuracy: Optional[float] = None
    verifier_robustness: Optional[float] = None
    human_eval_helpful_accuracy: Optional[float] = None
    human_eval_sneaky_accuracy: Optional[float] = None
    notes: str = ""
