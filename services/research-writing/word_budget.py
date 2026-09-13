"""Deterministic allocation of a total paper-length target across the
outline's sections and the two "bookend" sections (Introduction/
Conclusion, generated separately — see DECISIONS.md D-025).

See DECISIONS.md D-025/D-026 for the full rationale. In short: the
previous architecture had a `target_words` field that was (a) never
actually reachable from any user interface, and (b) even when set, only
ever enforced as an upper bound on a naive equal split across sections —
never a real per-section target, minimum, or generation instruction. This
module is the fix: one real total-word target divided into per-role
budgets (with a target, a minimum, and a maximum), used both to guide
generation and to detect real under-generation.
"""

from __future__ import annotations

from dataclasses import dataclass

# Length presets offered to the user (researchgenie/cli.py). Informed by
# PAPER_OUTPUT_DIAGNOSTIC.md's real measurement that the previous
# architecture averaged ~1,353 words across 4 sections — these targets are
# deliberately well above that baseline, but still realistic for a
# local-model-generated draft at this project's evidence depth (D-010, since
# extended by D-039 to include the authors' own retrieved Discussion text).
# Deliberately NOT raised when D-039 landed: whether the deeper evidence
# actually moves drafted length is unmeasured, and changing the targets in
# the same pass would confound that measurement.
LENGTH_PRESETS: dict[str, int] = {
    "short": 2200,
    "standard": 4000,
    "detailed": 6000,
}
DEFAULT_LENGTH_PRESET = "standard"

# Fraction of the TOTAL target given to each generated piece. Introduction
# and Conclusion are "bookends" (writing/modules/review_writer.py), not
# part of writing_order, so they get their own weighted slice rather than
# an equal per-tag split. Weights sum to 1.0.
SECTION_WEIGHTS: dict[str, float] = {
    "Introduction": 0.12,   # bookend
    "Background": 0.13,
    "Literature Review": 0.32,
    "Discussion": 0.22,
    "Limitations": 0.11,
    "Conclusion": 0.10,     # bookend
}

# A section under this fraction of its target is flagged as genuinely
# under-generated (see section_auditor.py) rather than merely "shorter than
# ideal" — deliberately generous so a well-supported but naturally concise
# section (e.g. Limitations) is not endlessly re-litigated.
MIN_FRACTION = 0.55
# A section over this fraction of its target is flagged as over-length —
# loosened from the old strict "any excess" rule now that target_words
# means a target, not a hard ceiling.
MAX_FRACTION = 1.75

_MIN_SECTION_WORDS = 40  # a budget floor so short presets don't round to ~0


@dataclass(frozen=True)
class WordBudget:
    target: int
    minimum: int
    maximum: int


# Tolerance for the WHOLE paper's actual length against its requested total
# target (see ResearchRequest.max_draft_length / DraftMetadata.length_status)
# — deliberately tighter than a single section's MIN_FRACTION/MAX_FRACTION
# above, since per-section variance mostly cancels out at the paper level,
# so the total should track the user's chosen preset more closely than any
# one section tracks its own budget.
_TOTAL_MIN_FRACTION = 0.7
_TOTAL_MAX_FRACTION = 1.4


def classify_length(actual_words: int, total_target_words: int | None) -> str | None:
    """Returns "short" / "on_target" / "over_target", or None when no total
    target was requested (old, length-unaware behavior is preserved)."""
    if not total_target_words:
        return None
    if actual_words < total_target_words * _TOTAL_MIN_FRACTION:
        return "short"
    if actual_words > total_target_words * _TOTAL_MAX_FRACTION:
        return "over_target"
    return "on_target"


def allocate_section_budgets(total_target_words: int | None) -> dict[str, WordBudget]:
    """Returns a WordBudget per section role name (matching the outline's
    own heading titles, plus "Introduction"/"Conclusion" for the bookends).

    Returns an empty dict when total_target_words is falsy — no budget is
    enforced, exactly preserving pre-D-025 behavior (no target_words
    anywhere in the prompt or audit) for any caller that doesn't opt into
    length planning."""
    if not total_target_words:
        return {}
    budgets: dict[str, WordBudget] = {}
    for role, weight in SECTION_WEIGHTS.items():
        target = max(_MIN_SECTION_WORDS, round(total_target_words * weight))
        budgets[role] = WordBudget(
            target=target,
            minimum=round(target * MIN_FRACTION),
            maximum=round(target * MAX_FRACTION),
        )
    return budgets
