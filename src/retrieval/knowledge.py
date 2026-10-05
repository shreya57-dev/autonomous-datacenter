"""Load operational Markdown and split it into stable policy chunks.

This layer only reads documents. It does not embed them or call a model.
"""

from dataclasses import dataclass
from pathlib import Path


OPERATIONAL_DOCUMENTS: tuple[str, ...] = (
    "capacity_management",
    "resource_threshold_policy",
    "server_failure_recovery",
    "server_maintenance",
    "workload_placement_policy",
)


@dataclass(frozen=True)
class Document:
    """One Markdown policy file. document_id is the file stem."""

    document_id: str
    text: str

    def __post_init__(self) -> None:
        if not isinstance(self.document_id, str) or not self.document_id.strip():
            raise ValueError("document_id must be a non-empty string")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError(f"document {self.document_id!r} is empty")


@dataclass(frozen=True)
class Chunk:
    """One operational section. The text is the policy, not an action."""

    document_id: str
    chunk_id: str
    topic: str
    text: str

    def __post_init__(self) -> None:
        _require_text("document_id", self.document_id)
        _require_text("chunk_id", self.chunk_id)
        _require_text("topic", self.topic)
        _require_text("text", self.text)


def load_document(path: Path) -> Document:
    """Read one Markdown file. A missing or empty file is an error."""
    if not isinstance(path, Path):
        raise TypeError("path must be a Path")
    if not path.is_file():
        raise FileNotFoundError(f"document not found: {path}")
    document_id = path.stem
    if not document_id.strip():
        raise ValueError(f"document id is empty for {path}")
    return Document(document_id=document_id, text=path.read_text(encoding="utf-8"))


def chunk_document(document: Document) -> list[Chunk]:
    """Split a document on level-2 headings, in file order.

    The level-1 title is not a chunk. A heading with no body is invalid.
    """
    if not isinstance(document, Document):
        raise TypeError("document must be a Document")
    sections = _sections(document.text)
    if not sections:
        raise ValueError(f"document {document.document_id!r} has no sections")
    chunks: list[Chunk] = []
    for index, (topic, body) in enumerate(sections, start=1):
        chunks.append(
            Chunk(
                document_id=document.document_id,
                chunk_id=f"{document.document_id}:{index}",
                topic=topic,
                text=body,
            )
        )
    return chunks


def load_operational_knowledge(directory: Path) -> list[Chunk]:
    """Load the five policy documents in document-id order."""
    if not isinstance(directory, Path):
        raise TypeError("directory must be a Path")
    if not directory.is_dir():
        raise FileNotFoundError(f"knowledge directory not found: {directory}")
    chunks: list[Chunk] = []
    for document_id in OPERATIONAL_DOCUMENTS:
        document = load_document(directory / f"{document_id}.md")
        if document.document_id != document_id:
            raise ValueError(f"document id {document.document_id!r} is not {document_id!r}")
        chunks.extend(chunk_document(document))
    chunk_ids = [chunk.chunk_id for chunk in chunks]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("chunk ids must be unique")
    return chunks


def _sections(text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    topic: str | None = None
    body: list[str] = []
    for line in text.splitlines():
        if line.startswith("## "):
            if topic is not None:
                sections.append((topic, _body(body)))
            topic = line[3:].strip()
            if not topic:
                raise ValueError("section heading is empty")
            body = []
            continue
        if topic is not None:
            body.append(line)
    if topic is not None:
        sections.append((topic, _body(body)))
    for topic, body_text in sections:
        if not body_text:
            raise ValueError(f"section {topic!r} is empty")
    return sections


def _body(lines: list[str]) -> str:
    return "\n".join(lines).strip()


def _require_text(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
