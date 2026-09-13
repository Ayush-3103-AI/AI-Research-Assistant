"""Discover Markdown papers and extract deterministic source metadata."""

from hashlib import sha256
from pathlib import Path
from typing import cast, get_args

import yaml

from writing.schemas import EvidenceDepth, PaperDocument, PartialPaperMetadata

# Derived from the type rather than restated, so a new depth is one edit.
EVIDENCE_DEPTHS = set(get_args(EvidenceDepth))


class DocumentRegistryError(ValueError):
    """Raised when a literature directory or document metadata is invalid."""


def _front_matter(
    text: str, path: Path
) -> tuple[PartialPaperMetadata, EvidenceDepth]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return PartialPaperMetadata(), "full_text"
    try:
        closing_index = next(
            index for index, line in enumerate(lines[1:], start=1) if line.strip() == "---"
        )
    except StopIteration as error:
        raise DocumentRegistryError(f"unterminated front matter: {path}") from error

    loaded = yaml.safe_load("\n".join(lines[1:closing_index]))
    if loaded is None:
        return PartialPaperMetadata(), "full_text"
    if not isinstance(loaded, dict):
        raise DocumentRegistryError(f"front matter must be a mapping: {path}")
    metadata = cast(dict[object, object], loaded)
    raw_evidence_depth = metadata.get("evidence_depth", "full_text")
    if raw_evidence_depth not in EVIDENCE_DEPTHS:
        raise DocumentRegistryError(
            f"evidence_depth must be one of {sorted(EVIDENCE_DEPTHS)}: {path}"
        )
    known_fields = set(PartialPaperMetadata.model_fields)
    filtered = {str(key): value for key, value in metadata.items() if str(key) in known_fields}
    return (
        PartialPaperMetadata.model_validate(filtered),
        cast(EvidenceDepth, raw_evidence_depth),
    )


def discover_documents(directory: str | Path) -> list[PaperDocument]:
    """Return stable paper records sorted by relative path."""

    root = Path(directory).resolve()
    if not root.is_dir():
        raise DocumentRegistryError(f"literature directory does not exist: {root}")

    paths = sorted(
        (path for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".md"),
        key=lambda path: path.relative_to(root).as_posix().casefold(),
    )
    documents: list[PaperDocument] = []
    for index, path in enumerate(paths, start=1):
        raw = path.read_bytes()
        text = raw.decode("utf-8-sig")
        metadata, evidence_depth = _front_matter(text, path)
        documents.append(
            PaperDocument(
                paper_id=f"P{index:03d}",
                source_path=str(path),
                relative_path=path.relative_to(root).as_posix(),
                content_hash=sha256(raw).hexdigest(),
                evidence_depth=evidence_depth,
                metadata=metadata,
            )
        )
    return documents
