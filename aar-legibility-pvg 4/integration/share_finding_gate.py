"""
Hooks the TRAINED verifier from this PVG pipeline into the real AAR's
share_finding tool, in the automated-w2s-research repo.

This does NOT retrain the AAR itself — the AAR (Claude, called via API in
the real automated-w2s-research setup) is a frozen model, not fine-tuned
here (see LIMITATIONS.md #5). This is a submission GATE: it rejects
findings the trained verifier scores as unsound, forcing the AAR to
revise before a finding is accepted, the same way share_finding already
rejects "result" findings with fewer than 5 seeds.

Integration point (in the real repo):
    w2s_research/research_loop/tools/server_api_tools.py
    -> share_finding(), inside the `if finding_type == "result":` block,
       right after the existing num_seeds check (~line 302-318 as of the
       version reviewed for this project).

To apply: copy the check below into that location, importing
score_finding_with_verifier from this module (or inline the logic if you
prefer to keep server_api_tools.py dependency-free of this repo).
"""
import json
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PVGConfig


def render_verifier_prompt(summary: str, metrics: dict, config: dict) -> str:
    from jinja2 import Template
    template_path = Path(__file__).parent.parent / "prompts" / "verifier_prompt.jinja2"
    with open(template_path, "r") as f:
        template = Template(f.read())
    return template.render(summary=summary, metrics=json.dumps(metrics), config=json.dumps(config))


def parse_verifier_verdict(text: str) -> Optional[float]:
    normalized = text.strip().upper()
    if "UNSOUND" in normalized:
        return 0.0
    if "SOUND" in normalized:
        return 1.0
    return None


async def score_finding_with_verifier(summary: str, metrics: dict, config: dict, cfg: PVGConfig) -> Optional[float]:
    """
    Calls the TRAINED verifier model (from training/pvg_loop.py's output,
    not the base/untrained model) to score a real, live finding submission.

    TODO: point this at wherever your trained verifier is served (a local
    vLLM instance, a hosted endpoint, etc.) — this is intentionally left
    unwired since it depends on how you deployed the verifier after
    training completed.
    """
    prompt = render_verifier_prompt(summary, metrics, config)
    raise NotImplementedError(
        "Wire score_finding_with_verifier() to your trained, deployed "
        "verifier model. See render_verifier_prompt()/parse_verifier_verdict() "
        "for the expected prompt and response shape."
    )


# ---------------------------------------------------------------------------
# The actual patch to apply inside share_finding() in server_api_tools.py
# ---------------------------------------------------------------------------
PATCH_SNIPPET = '''
# --- INSERT into share_finding(), right after the existing num_seeds check ---
if finding_type == "result":
    from aar_legibility_pvg.integration.share_finding_gate import score_finding_with_verifier
    from aar_legibility_pvg.config import PVGConfig

    verifier_score = await score_finding_with_verifier(
        summary=summary, metrics=metrics or {}, config=config or {}, cfg=PVGConfig(),
    )
    if verifier_score is not None and verifier_score < PVGConfig().gate_reject_threshold:
        return {
            "content": [{
                "type": "text",
                "text": json.dumps({
                    "success": False,
                    "error": (
                        f"Rejected: finding scored {verifier_score:.2f} on the trained "
                        f"legibility verifier (threshold {PVGConfig().gate_reject_threshold}). "
                        f"Check that every specific claim is directly supported by the "
                        f"metrics/config you cited before resubmitting."
                    ),
                }, indent=2)
            }]
        }
    # Surface the score on the leaderboard/forum too, not just as a gate.
    payload["legibility_verifier_score"] = verifier_score
# --- END INSERT ---
'''

if __name__ == "__main__":
    print("This module is meant to be imported from a patched copy of")
    print("server_api_tools.py in automated-w2s-research. See PATCH_SNIPPET")
    print("above for the exact code to insert, and the module docstring for")
    print("where it goes.")
    print()
    print(PATCH_SNIPPET)
