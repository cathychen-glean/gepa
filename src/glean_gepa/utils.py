from collections.abc import Mapping

from glean_gepa.al_adapter import Candidate


def apply_module_edits(parent: Candidate, edits: Mapping[str, str]) -> Candidate:
    """Copy ``parent`` with every module in ``edits`` replaced."""
    pm = dict(parent.prompt_modules)
    pm.update(edits)
    return Candidate(
        model=parent.model,
        prompt_modules=pm,
        module_specs=parent.module_specs,
        global_token_cap=parent.global_token_cap,
        baseline_prompt_hash=parent.baseline_prompt_hash,
        parent_id=parent.candidate_id,
    )
