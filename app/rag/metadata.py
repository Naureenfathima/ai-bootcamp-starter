"""
metadata.py — Document metadata extraction (RAG Stage 1b).

Metadata is the data ABOUT your data. Rich metadata enables:
  - Filtered retrieval: "only search documents from 2024"
  - Source attribution: cite chapter and author alongside the answer
  - Quality scoring: prefer primary sources over secondary summaries
  - Debug visibility: understand exactly what was ingested

Systems lesson:
  Every document entering your pipeline should carry a structured metadata schema.
  Think of it like a database row — define the schema up front, enforce types,
  log missing fields. Metadata upgrades retrieval from "find similar text" to
  "find relevant text from source Y in time period Z with tag T".

Architecture:
  DocumentMetadata (schema) ← MetadataExtractor (dispatcher)
                                  ├── _extract_from_text()      (heuristics)
                                  ├── _extract_from_markdown()  (YAML front matter)
                                  └── _extract_from_pdf()       (pypdf)

STUDENT TODO:
  - Add DateFilter: retrieve only documents newer than a given date.
  - Add SourceFilter: restrict retrieval to specific authors or domains.
  - Connect metadata to PgVectorStore: store as JSONB and filter in SQL.
  - Add a metadata validation step that warns on missing required fields.
"""

import re
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class DocumentMetadata:
    """
    Structured metadata about a document.

    Every field has a clear type and a sensible default so downstream components
    can safely access any field without defensive None checks.
    """
    source: str                               # filename, URL, or identifier
    title: str = ""                           # document title
    author: str = ""                          # author name(s)
    date: str = ""                            # ISO date string YYYY-MM-DD
    doc_type: str = "text"                    # "text" | "markdown" | "pdf"
    page_count: int = 0                       # PDF pages or MD section count
    word_count: int = 0                       # total word count
    language: str = "en"                      # ISO 639-1 language code
    section: str = ""                         # section/chapter within the document
    tags: list[str] = field(default_factory=list)
    custom: dict = field(default_factory=dict)  # domain-specific fields

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "title": self.title,
            "author": self.author,
            "date": self.date,
            "doc_type": self.doc_type,
            "page_count": self.page_count,
            "word_count": self.word_count,
            "language": self.language,
            "section": self.section,
            "tags": self.tags,
            **self.custom,
        }

    @classmethod
    def minimal(cls, source: str) -> "DocumentMetadata":
        """Minimal metadata from just a source label — use as a fallback."""
        return cls(source=source)


class MetadataExtractor:
    """
    Dispatches to type-specific extractors based on file extension.

    .txt / .py / etc → _extract_from_text()     (heuristics)
    .md / .mdx        → _extract_from_markdown() (YAML front matter + headings)
    .pdf              → _extract_from_pdf()      (requires: pip install pypdf)

    Usage:
        extractor = MetadataExtractor()
        meta = extractor.extract("reports/q3_2024.pdf")
        meta = extractor.extract_from_string(text, source="api_upload.txt")
        print(meta.to_dict())
    """

    def extract(self, path: str | Path) -> DocumentMetadata:
        """
        Extract metadata from a file on disk.

        Dispatches to the appropriate extractor based on file extension.
        """
        path = Path(path)
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            return self._extract_from_pdf(path)
        elif suffix in (".md", ".mdx", ".markdown"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            return self._extract_from_markdown(text, source=path.name)
        else:
            text = path.read_text(encoding="utf-8", errors="ignore")
            return self._extract_from_text(text, source=path.name)

    def extract_from_string(
        self, text: str, source: str, doc_type: str = "text"
    ) -> DocumentMetadata:
        """
        Extract metadata from a raw text string (e.g. from an API upload).

        Args:
            text:     Raw document text.
            source:   Identifier for this document.
            doc_type: Hint for how to parse ("text" | "markdown").
        """
        if doc_type == "markdown":
            return self._extract_from_markdown(text, source=source)
        return self._extract_from_text(text, source=source)

    # ── Extractors ─────────────────────────────────────────────────────────────

    def _extract_from_text(self, text: str, source: str) -> DocumentMetadata:
        """
        Heuristic metadata from plain text.

        Patterns detected:
          Title  — first line if it looks title-cased or ALL CAPS
          Author — lines matching "Author:" or "By <Name>"
          Date   — ISO or US-style dates in the first 500 chars
        """
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        return DocumentMetadata(
            source=source,
            title=self._guess_title(lines),
            author=self._guess_author(lines[:20]),
            date=self._find_date(text[:500]),
            doc_type="text",
            word_count=len(text.split()),
            language=self._detect_language(text[:300]),
        )

    def _extract_from_markdown(self, text: str, source: str) -> DocumentMetadata:
        """
        Extract from YAML front matter and Markdown headings.

        Supported front matter fields: title, author, date, tags.

        Example:
            ---
            title: My Design Doc
            author: Jane Smith
            date: 2024-01-15
            tags: [rag, architecture]
            ---
        """
        meta = DocumentMetadata(source=source, doc_type="markdown")
        body = text

        fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
        if fm_match:
            front_matter = fm_match.group(1)
            body = text[fm_match.end():]
            for key, value in re.findall(r"^(\w+):\s*(.+)$", front_matter, re.MULTILINE):
                key = key.lower()
                value = value.strip().strip('"\'')
                if key == "title":
                    meta.title = value
                elif key == "author":
                    meta.author = value
                elif key == "date":
                    meta.date = value
                elif key == "tags":
                    meta.tags = [t.strip().strip('"\'') for t in value.strip("[]").split(",")]

        # Fall back to H1 as title
        if not meta.title:
            h1 = re.search(r"^#\s+(.+)$", body, re.MULTILINE)
            if h1:
                meta.title = h1.group(1).strip()

        headings = re.findall(r"^#{1,6}\s+.+$", body, re.MULTILINE)
        meta.page_count = len(headings)
        meta.word_count = len(body.split())
        meta.language = self._detect_language(body[:300])

        logger.debug("Markdown meta: title=%r author=%r date=%r tags=%s",
                     meta.title, meta.author, meta.date, meta.tags)
        return meta

    def _extract_from_pdf(self, path: Path) -> DocumentMetadata:
        """
        Extract from PDF metadata and body text using pypdf.

        Reads: /Title, /Author, /CreationDate, page count, word count.
        Requires: pip install pypdf
        """
        try:
            from pypdf import PdfReader
        except ImportError:
            logger.warning("pypdf not installed — returning minimal PDF metadata. "
                           "Run: pip install pypdf")
            return DocumentMetadata(source=path.name, doc_type="pdf")

        try:
            reader = PdfReader(str(path))
            info = reader.metadata or {}

            title = str(info.get("/Title", "")).strip() or path.stem
            author = str(info.get("/Author", "")).strip()
            raw_date = str(info.get("/CreationDate", ""))

            # PDF dates: "D:YYYYMMDDHHmmss" → YYYY-MM-DD
            date = ""
            dm = re.search(r"D:(\d{4})(\d{2})(\d{2})", raw_date)
            if dm:
                date = f"{dm.group(1)}-{dm.group(2)}-{dm.group(3)}"

            page_count = len(reader.pages)
            all_text = " ".join(p.extract_text() or "" for p in reader.pages)
            word_count = len(all_text.split())

            logger.info("PDF meta: title=%r pages=%d words=%d", title, page_count, word_count)
            return DocumentMetadata(
                source=path.name,
                title=title,
                author=author,
                date=date,
                doc_type="pdf",
                page_count=page_count,
                word_count=word_count,
                language=self._detect_language(all_text[:300]),
            )
        except Exception as exc:
            logger.error("PDF metadata extraction failed for %s: %s", path, exc)
            return DocumentMetadata(source=path.name, doc_type="pdf")

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _guess_title(self, lines: list[str]) -> str:
        for line in lines[:5]:
            if len(line) < 120 and (line.istitle() or line.isupper() or line.startswith("#")):
                return line.lstrip("#").strip()
        return lines[0][:80] if lines else ""

    def _guess_author(self, lines: list[str]) -> str:
        for line in lines:
            m = re.match(r"(?:author|by)[:\s]+(.+)", line, re.IGNORECASE)
            if m:
                return m.group(1).strip()
        return ""

    def _find_date(self, text: str) -> str:
        """Return the first ISO or US-style date found in text."""
        iso = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text)
        if iso:
            return iso.group(1)
        us = re.search(r"\b(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})\b", text)
        if us:
            return f"{us.group(3)}-{us.group(1).zfill(2)}-{us.group(2).zfill(2)}"
        return ""

    def _detect_language(self, sample: str) -> str:
        """
        Lightweight language heuristic using stop-word frequency.

        For production use langdetect (pip install langdetect) or fasttext.
        STUDENT TODO: swap this for a real language detector.
        """
        words = set(re.findall(r"\b\w+\b", sample.lower()))
        scores = {
            "fr": len(words & {"le", "la", "les", "est", "un", "une", "des", "et", "en"}),
            "es": len(words & {"el", "la", "los", "es", "un", "una", "de", "en", "y"}),
            "de": len(words & {"der", "die", "das", "ist", "ein", "und", "in", "zu"}),
        }
        best_lang, best_score = max(scores.items(), key=lambda x: x[1])
        return best_lang if best_score >= 2 else "en"
