Extract one evidence-rich literature card from one supplied Markdown paper for an
outline-driven review. Use only the current paper and supplied tag definitions.

### Evidence extraction

Respect the supplied `evidence_depth`. When it is `abstract`, the document is an abstract
evidence packet rather than a full paper: extract only claims explicitly stated in the
abstract, preserve abstract-level uncertainty, never infer unreported methods or results,
and never describe the source as having been reviewed in full text.

When `evidence_depth` is `abstract_plus_discussion`, the document carries the abstract plus
the authors' own Discussion / Limitations / Future-work sections, under their own headings,
retrieved from the real full text. Extract from both, and extract the whole of what the
Discussion section supports — the authors' interpretations of their results, the boundary
conditions and confounders they name, the limitations they concede, and the future work they
call for, attributed to them rather than asserted as established fact. The rest of the paper
(methods, full results tables, related work) is still absent, so everything above applies
unchanged: never infer an unreported method or result, and never describe the source as
having been reviewed in full text.

Extract claims that can support comparative review writing, not a generic paper summary.
Each point should express one citable proposition while retaining the methodological context,
studied population or system, comparison conditions, direction, uncertainty, and boundary
conditions needed to judge transferability. Split claims that differ in evidence type or
certainty; do not fragment away qualifiers that determine what the result means.

Classify empirical findings, methods, research questions, theoretical arguments, author
interpretations, stated limitations, background, and future directions accurately. Preserve
the paper's distinction between observation and explanation. Never convert correlation into
causation, a null result into proof of no effect, or author speculation into a finding.

### Provenance and tagging

1. Return every supplied tag ID exactly once; use `[]` when the paper is irrelevant.
2. Assign each point only to the deepest accurately matching tag. Do not duplicate a point in
   an ancestor. A genuinely cross-branch claim may reuse one point ID across those branches.
3. Never fill a tag by inference, outside knowledge, or a claim that appears only in the
   paper's reference list or its summary of prior work.
4. Use unique point IDs beginning with the supplied paper ID, for example
   `P003-POINT-01`.
5. Paraphrase faithfully in compact scientific prose. Do not quote text or cite page numbers.
6. Prefer supplied structured metadata; complete only genuinely missing fields that the
   current paper establishes. Produce paper-level citation information for this paper.

Before returning, verify that another writer could compare each point with other studies
without silently broadening its population, method, outcome, certainty, or scope.

### If a previous attempt is supplied

When the input includes `previous_attempt_errors`, a prior attempt at this exact card failed
deterministic validation for those specific reasons. Fix each one directly rather than
regenerating from scratch and repeating the same mistake — most commonly this means a point
was assigned to both a tag and one of its ancestors (see rule 2 above: assign each point to
only its single deepest matching tag, never duplicate it upward), or a tag/point ID did not
match the required shape.

### Output shape (do not confuse these two dictionaries)

`points` is keyed by POINT ID (each point's own `point_id`, e.g. `"P003-POINT-01"`).
`tag_values` is keyed by TAG ID (e.g. `"TAG-ROOT"`, `"TAG-1"`) and its values are lists of
point IDs that must already exist as keys in `points`. Never use a tag ID as a `points` key,
and never use a point ID as a `tag_values` key. Example shape for paper `P003` with tags
`TAG-ROOT` and `TAG-1`:

```json
{
  "points": {
    "P003-POINT-01": {"point_id": "P003-POINT-01", "content": "...", "content_type": "empirical_finding", "certainty": "supported"}
  },
  "tag_values": {
    "TAG-ROOT": [],
    "TAG-1": ["P003-POINT-01"]
  }
}
```
