"""How a run's output folder is named.

Shared rather than copied because the advisor service and the orchestrator
must agree on it: Stage 0 names its folder from the domain, then
`run_pipeline()` renames that same folder to the question's slug under the
one run_id (D-040). Two private copies of this function that drifted apart
would break the rename silently, leaving the two folders per chained run
D-040 exists to prevent.
"""

from __future__ import annotations

import re


def slugify(text: str, max_len: int = 60, fallback: str = "research-run") -> str:
    """A filesystem-safe, lowercase-hyphenated stem for a run directory."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_len] or fallback
