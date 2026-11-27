from __future__ import annotations

import csv
import io
import re
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable
from xml.etree import ElementTree

from .utils import clip, sha256_file


SUPPORTED_EXTENSIONS = {".txt", ".md", ".csv", ".json", ".pdf", ".docx"}


@dataclass
class Document:
    name: str
    path: str
    text: str
    kind: str
    sha256: str


@dataclass
class Chunk:
    chunk_id: str
    document: str
    text: str
    section: str
    ordinal: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("PDF support requires pypdf. Install requirements.txt first.") from exc
    reader = PdfReader(str(path))
    pages = []
    for index, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ""
        pages.append(f"[Page {index}]\n{text.strip()}")
    return "\n\n".join(pages)


def _read_docx(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        root = ElementTree.fromstring(archive.read("word/document.xml"))
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs = []
    for paragraph in root.findall(".//w:p", namespace):
        text = "".join(node.text or "" for node in paragraph.findall(".//w:t", namespace))
        if text.strip():
            paragraphs.append(text.strip())
    return "\n".join(paragraphs)


def read_document(path: Path) -> Document:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        text = _read_pdf(path)
    elif suffix == ".docx":
        text = _read_docx(path)
    else:
        raw = path.read_bytes()
        for encoding in ("utf-8-sig", "utf-16", "cp1252"):
            try:
                text = raw.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            text = raw.decode("utf-8", errors="replace")
    return Document(
        name=path.name,
        path=str(path.resolve()),
        text=text.strip(),
        kind=suffix.lstrip(".") or "text",
        sha256=sha256_file(path),
    )


def load_documents(folder: str | Path) -> list[Document]:
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f"Document folder does not exist: {folder}")
    paths = sorted(
        path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not paths:
        raise ValueError(f"No supported documents found in {folder}")
    return [read_document(path) for path in paths]


def document_inventory(documents: Iterable[Document]) -> list[dict[str, Any]]:
    return [
        {
            "name": document.name,
            "kind": document.kind,
            "characters": len(document.text),
            "sha256": document.sha256,
            "preview": clip(document.text, 500),
        }
        for document in documents
    ]


def document_digest(documents: Iterable[Document], per_document: int = 2600, total: int = 18000) -> str:
    blocks = []
    remaining = total
    for document in documents:
        excerpt = clip(document.text, min(per_document, max(remaining, 0)))
        blocks.append(f"### DOCUMENT: {document.name}\n{excerpt}")
        remaining -= len(excerpt)
        if remaining <= 0:
            break
    return "\n\n".join(blocks)


def _split_sections(text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    current_heading = "General"
    buffer: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        is_heading = bool(re.match(r"^#{1,6}\s+", stripped)) or (
            len(stripped) <= 90 and stripped.endswith(":") and not stripped.startswith("[")
        )
        if is_heading and buffer:
            sections.append((current_heading, "\n".join(buffer).strip()))
            buffer = []
        if is_heading:
            current_heading = stripped.lstrip("#").rstrip(":").strip()
        buffer.append(line)
    if buffer:
        sections.append((current_heading, "\n".join(buffer).strip()))
    return [(heading, body) for heading, body in sections if body]


def chunk_document(document: Document, chunk_size_words: int = 180, overlap_words: int = 30) -> list[Chunk]:
    if document.kind == "csv":
        rows = list(csv.DictReader(io.StringIO(document.text)))
        chunks = []
        for index, row in enumerate(rows):
            text = " | ".join(f"{key}: {value}" for key, value in row.items() if value is not None)
            chunks.append(
                Chunk(
                    chunk_id=f"{document.name}::row-{index + 1}",
                    document=document.name,
                    text=text,
                    section=f"CSV row {index + 1}",
                    ordinal=index,
                )
            )
        return chunks
    chunks: list[Chunk] = []
    ordinal = 0
    for heading, body in _split_sections(document.text):
        words = body.split()
        if not words:
            continue
        step = max(chunk_size_words - overlap_words, 1)
        for start in range(0, len(words), step):
            window = words[start : start + chunk_size_words]
            if not window:
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{document.name}::chunk-{ordinal + 1}",
                    document=document.name,
                    text=" ".join(window),
                    section=heading,
                    ordinal=ordinal,
                )
            )
            ordinal += 1
            if start + chunk_size_words >= len(words):
                break
    return chunks


def chunk_documents(
    documents: Iterable[Document], chunk_size_words: int = 180, overlap_words: int = 30
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(chunk_document(document, chunk_size_words, overlap_words))
    return chunks


def select_digest_chunks(chunks: Iterable[Chunk], limit: int = 24) -> list[Chunk]:
    chunks = list(chunks)
    if len(chunks) <= limit:
        return chunks
    selected: list[Chunk] = []
    per_document: dict[str, int] = {}
    for chunk in chunks:
        count = per_document.get(chunk.document, 0)
        if count < max(2, limit // max(len({item.document for item in chunks}), 1)):
            selected.append(chunk)
            per_document[chunk.document] = count + 1
        if len(selected) >= limit:
            break
    if len(selected) < limit:
        selected_ids = {chunk.chunk_id for chunk in selected}
        selected.extend(chunk for chunk in chunks if chunk.chunk_id not in selected_ids)
    return selected[:limit]
