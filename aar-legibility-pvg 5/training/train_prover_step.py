"""
Fine-tunes a STAND-IN prover model (not the real AAR — see LIMITATIONS.md #7)
via a GRPO-style policy update, using the verifier's score as reward.

This is what makes "legibility training" literally true for the prover, not
just the verifier: the prover's own weights change, round over round, to
produce output the verifier scores as sound (helpful role) or as
convincingly-sound-but-actually-unsound (sneaky role).

Method (mirrors GRPO, as used elsewhere in this project's underlying
weak-to-strong codebase):
  1. For each experiment record, sample `num_generations` findings from the
     CURRENT prover policy (temperature > 0, so they differ).
  2. Score each with the current verifier -> reward.
  3. Compute the group-relative advantage: (reward - group_mean) / group_std.
     This is what GRPO uses instead of a separate value/critic model.
  4. Policy-gradient update: maximize advantage-weighted log-probability of
     the generated tokens, with a KL penalty against the frozen base model
     to prevent the prover from drifting into degenerate, high-reward-but-
     incoherent text.

This trains ONE role at a time (helpful or sneaky) — call it once for each
per round from pvg_loop.py, since they optimize toward different targets
(helpful reward = verifier accepting an ACTUALLY accurate finding; sneaky
reward = verifier accepting an INACCURATE one) even though both use the
same underlying "did the verifier accept this" signal.
"""
from dataclasses import dataclass
from typing import Callable, List
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import PVGConfig


@dataclass
class ProverState:
    """Holds the loaded model/tokenizer/optimizer across rounds, so the
    prover's weights persist and accumulate updates round over round."""
    model: AutoModelForCausalLM
    ref_model: AutoModelForCausalLM  # frozen copy, for the KL penalty
    tokenizer: AutoTokenizer
    optimizer: torch.optim.Optimizer


def load_prover(cfg: PVGConfig, base_model_name: str) -> ProverState:
    """
    Loads the stand-in prover with a LoRA adapter (only the adapter trains —
    keeps the update cheap and keeps a clean frozen reference model for the
    KL term, without needing two full copies of a large model in memory).
    """
    from peft import LoraConfig, get_peft_model

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.bfloat16)
    ref_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.bfloat16)
    for p in ref_model.parameters():
        p.requires_grad = False

    lora_cfg = LoraConfig(r=cfg.prover_lora_r, lora_alpha=cfg.prover_lora_alpha,
                           target_modules=["q_proj", "v_proj"], lora_dropout=0.0)
    model = get_peft_model(base, lora_cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.prover_lr)

    return ProverState(model=model, ref_model=ref_model, tokenizer=tokenizer, optimizer=optimizer)


@torch.no_grad()
def generate_group(state: ProverState, prompt: str, cfg: PVGConfig) -> List[str]:
    """Samples cfg.prover_num_generations completions for one prompt."""
    inputs = state.tokenizer(prompt, return_tensors="pt")
    outputs = state.model.generate(
        **inputs,
        max_new_tokens=cfg.prover_max_new_tokens,
        do_sample=True,
        temperature=cfg.prover_temperature,
        num_return_sequences=cfg.prover_num_generations,
        pad_token_id=state.tokenizer.pad_token_id,
    )
    completions = state.tokenizer.batch_decode(
        outputs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    return completions


def compute_sequence_logprob(model, tokenizer, prompt: str, completion: str) -> torch.Tensor:
    """Sum of token log-probs for `completion` conditioned on `prompt`, under `model`."""
    full_text = prompt + completion
    full_ids = tokenizer(full_text, return_tensors="pt")["input_ids"]
    prompt_len = tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]

    logits = model(full_ids).logits[:, :-1, :]
    targets = full_ids[:, 1:]

    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    completion_log_probs = token_log_probs[:, prompt_len - 1:]
    return completion_log_probs.sum()


def train_prover_role(
    state: ProverState,
    prompts_and_records: List[dict],
    score_fn: Callable[[str, dict], float],
    cfg: PVGConfig,
) -> dict:
    """
    One GRPO-style update for a single prover role (helpful or sneaky).

    Args:
        state: the loaded ProverState for this role (helpful and sneaky
            should be separate ProverState instances / separate LoRA
            adapters, so they specialize into different behaviors rather
            than fighting over one set of weights).
        prompts_and_records: list of {"prompt": str, "record": ExperimentRecord-as-dict}
        score_fn: takes (completion, record) -> float reward, i.e. a call
            into the CURRENT verifier (score_with_verifier from pvg_loop.py).
        cfg: PVGConfig.

    Returns:
        {"mean_reward": float, "loss": float} for logging.
    """
    state.optimizer.zero_grad()
    total_loss = torch.tensor(0.0)
    all_rewards = []

    for item in prompts_and_records:
        prompt, record = item["prompt"], item["record"]
        completions = generate_group(state, prompt, cfg)

        rewards = [score_fn(c, record) for c in completions]
        all_rewards.extend(rewards)

        rewards_t = torch.tensor(rewards)
        mean_r, std_r = rewards_t.mean(), rewards_t.std().clamp_min(1e-6)
        advantages = (rewards_t - mean_r) / std_r  # GRPO's group-relative advantage

        for completion, advantage in zip(completions, advantages):
            logprob = compute_sequence_logprob(state.model, state.tokenizer, prompt, completion)
            with torch.no_grad():
                ref_logprob = compute_sequence_logprob(state.ref_model, state.tokenizer, prompt, completion)
            kl = logprob - ref_logprob  # per-sequence KL proxy

            # Policy gradient loss: push up logprob of high-advantage completions,
            # down for low-advantage ones, penalized by drift from the reference.
            loss = -(advantage.detach() * logprob) + cfg.prover_kl_penalty * kl
            total_loss = total_loss + loss

    total_loss = total_loss / max(len(prompts_and_records) * cfg.prover_num_generations, 1)
    total_loss.backward()
    state.optimizer.step()

    return {
        "mean_reward": sum(all_rewards) / len(all_rewards) if all_rewards else 0.0,
        "loss": total_loss.item(),
    }
