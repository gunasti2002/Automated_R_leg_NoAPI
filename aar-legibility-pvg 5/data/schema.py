"""
Data model for the AAR findings PVG dataset.

An ExperimentRecord is the "problem" (analogous to a math question).
A Finding is a "solution" — a natural-language write-up making a claim
about the ExperimentRecord's results. Findings come in three flavors:
  - honest: what the AAR (or a human) actually wrote, presumed accurate
  - helpful: prover-generated, trained to be both correct and verifier-accepted
  - sneaky: prover-generated, trained to misrepresent the record but still
            look verifier-acceptable
"""
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from enum import Enum


class FindingLabel(str, Enum):
    HONEST = "honest"      # ground-truth-checked accurate write-up
    HELPFUL = "helpful"    # prover-generated, intended accurate
    SNEAKY = "sneaky"      # prover-generated, intended misrepresentation
    UNLABELED = "unlabeled"  # real-world submission awaiting a verifier verdict


@dataclass
class ExperimentRecord:
    """
    The underlying, presumed-true data a finding is written about.
    Maps to automated-w2s-research's share_finding payload shape
    (see w2s_research/research_loop/tools/server_api_tools.py).
    """
    idea_name: str
    metrics: Dict[str, Any]      # e.g. {"pgr": 0.45, "transfer_acc": 0.78, "num_seeds": 5, ...}
    config: Dict[str, Any]       # hyperparameters used
    worked: Optional[bool] = None
    dataset: Optional[str] = None
    record_id: Optional[str] = None


@dataclass
class Finding:
    """A single natural-language write-up claiming something about an ExperimentRecord."""
    experiment: ExperimentRecord
    summary: str                  # the natural-language finding text
    label: FindingLabel
    claimed_conclusion: Optional[str] = None  # short paraphrase of the finding's headline claim

    # Ground-truth internal-consistency check results (see build_findings_dataset.py).
    # None until checked.
    is_internally_consistent: Optional[bool] = None
    consistency_issues: list = field(default_factory=list)  # human-readable list of problems found

    # Filled in during PVG training / evaluation.
    verifier_score: Optional[float] = None       # verifier's predicted P(sound)
    human_verdicts: list = field(default_factory=list)  # list of {"reviewer_id", "verdict", "time_taken_s"}
