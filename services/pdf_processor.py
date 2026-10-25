
import os
import re
from pathlib import Path

import pymupdf

from services.analysis_progress import report_stage


# Short selectable text with embedded images often means incomplete extraction.
_SUSPICIOUSLY_SHORT_CHARS = 40

# Garbled-text detection (added after evaluation on real protocols).
# Some PDFs use broken font encodings: text is selectable but scrambled,
# e.g. "3URWRFRO" instead of "Protocol". Readable English pages in our test
# protocols have >= 11% common function words; scrambled pages have <= 1.4%.
_COMMON_WORDS = frozenset(
    "the a an and or of to in on for with by is are be will was were as at "
    "from that this which not no all any if than then each who may must".split()
)
_GARBLED_MIN_WORDS = 30
_GARBLED_MAX_COMMON_RATIO = 0.04


class PDFProcessingError(Exception):
    """Raised when a PDF cannot be processed."""


def _has_meaningful_text(text: str) -> bool:
    """Return True when extracted text is non-empty after trimming."""
    return bool(text and text.strip())


def _looks_garbled(text: str) -> bool:
    """
    Return True when selectable text looks scrambled (broken font encoding).

    Uses the share of very common English words. Short pages are never
    judged, so headers, tables of numbers and title pages are not flagged.
    """
    words = re.findall(r"[A-Za-z]+", text or "")
    if len(words) < _GARBLED_MIN_WORDS:
        return False
    common = sum(1 for word in words if word.lower() in _COMMON_WORDS)
    return common / len(words) < _GARBLED_MAX_COMMON_RATIO


def _extract_selectable_text(page) -> str:
    """Extract selectable text in approximate reading order."""
    return page.get_text("text", sort=True).strip()


def _looks_tabular(text: str) -> bool:
    """Cheap gate: table pages have wide column gaps or many short lines."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 4:
        return False
    gapped = sum(1 for ln in lines if re.search(r"\S {3,}\S|\t", ln))
    short = sum(1 for ln in lines if len(ln) <= 25)
    return gapped >= 3 or (len(lines) >= 12 and short / len(lines) >= 0.5)


def _extract_tables(page, text: str) -> list[dict]:
    """
    Extract ruled tables as rows of cell text, in addition to the page text.

    The page "text" is never changed (it is the verification source); tables
    are an extra, optional view that keeps row/column structure. Any failure
    simply yields no tables.
    """
    if os.getenv("CTAI_EXTRACT_TABLES", "1").strip().lower() in {"0", "false", "off", "no"}:
        return []
    if not _looks_tabular(text) or not hasattr(page, "find_tables"):
        return []
    try:
        found = page.find_tables()
        tables = []
        for table in getattr(found, "tables", found):
            # Cell text is re-read from the page inside each cell's box:
            # table.extract() can drop spaces or split values like "+/- 3".
            rows = [
                [
                    " ".join(page.get_text("text", clip=pymupdf.Rect(cell)).split())
                    if cell else ""
                    for cell in row.cells
                ]
                for row in table.rows
            ]
            rows = [r for r in rows if any(c.strip() for c in r)]
            if len(rows) >= 2 and max(len(r) for r in rows) >= 2:
                tables.append({"rows": rows})
        return tables
    except Exception:
        return []


def _page_has_embedded_images(page) -> bool:
    """Return True when the page contains embedded images."""
    return bool(page.get_images(full=True))


def _looks_suspiciously_incomplete(page, text: str) -> bool:
    """
    Detect pages that likely contain more content than was extracted.

    Heuristic only: short text plus embedded images. Never invents text.
    """
    stripped = (text or "").strip()
    if not stripped:
        return True
    if len(stripped) >= _SUSPICIOUSLY_SHORT_CHARS:
        return False
    return _page_has_embedded_images(page)


def _load_ocr_dependencies():
    """
    Load local OCR dependencies.

    Raises PDFProcessingError with install guidance when unavailable.
    """
    try:
        import pytesseract
        from PIL import Image
    except ImportError as exc:
        raise PDFProcessingError(
            "OCR requires the Python packages 'pytesseract' and 'Pillow'. "
            "Install them with: pip install pytesseract Pillow"
        ) from exc

    try:
        pytesseract.get_tesseract_version()
    except Exception as exc:
        raise PDFProcessingError(
            "OCR requires the Tesseract OCR binary on PATH. "
            "Install it with: brew install tesseract (macOS) "
            "or apt-get install tesseract-ocr (Debian/Ubuntu)."
        ) from exc

    return pytesseract, Image


def _ocr_pdf_page(page, dpi: int = 200) -> str:
    """
    Render a PDF page to an image and run local OCR.

    Returns OCR text as-is (may be empty). Never invents content.
    """
    pytesseract, Image = _load_ocr_dependencies()

    zoom = dpi / 72.0
    pixmap = page.get_pixmap(
        matrix=pymupdf.Matrix(zoom, zoom),
        alpha=False,
    )
    image = Image.frombytes(
        "RGB",
        (pixmap.width, pixmap.height),
        pixmap.samples,
    )

    try:
        return pytesseract.image_to_string(image) or ""
    except Exception as exc:
        raise PDFProcessingError(
            f"OCR failed while reading page content: {exc}"
        ) from exc


def _choose_best_extracted_text(selectable_text: str, ocr_text: str) -> str:
    """
    Prefer the longer non-empty extraction without inventing content.

    When lengths are equal, prefer selectable text.
    """
    selectable = (selectable_text or "").strip()
    ocr = (ocr_text or "").strip()

    if not selectable:
        return ocr
    if not ocr:
        return selectable
    if len(ocr) > len(selectable):
        return ocr
    return selectable


def extract_pdf(file_path: str) -> dict:
    """
    Extract text from a PDF while preserving page numbers.

    Pages with selectable text are extracted in reading order. Pages with
    no meaningful text, or suspiciously short text plus embedded images,
    are rendered and processed with local OCR when available. Mixed
    selectable/scanned PDFs are supported. Missing content is never invented.
    """
    path = Path(file_path)

    if not path.is_file():
        raise PDFProcessingError(
            f"PDF file not found: {file_path}"
        )

    if path.suffix.lower() != ".pdf":
        raise PDFProcessingError(
            "Only PDF files are supported."
        )

    try:
        with pymupdf.open(path) as document:
            if not document.is_pdf:
                raise PDFProcessingError(
                    "The uploaded file is not a valid PDF."
                )

            pages = []
            warnings: list[str] = []

            page_total = len(document)
            report_stage("extract", 0, page_total)
            for index, page in enumerate(document):
                page_number = index + 1
                selectable_text = _extract_selectable_text(page)
                text = selectable_text
                ocr_applied = False
                attempted_ocr = False

                garbled = _looks_garbled(selectable_text)
                needs_ocr = (
                    not _has_meaningful_text(selectable_text)
                    or _looks_suspiciously_incomplete(page, selectable_text)
                    or garbled
                )

                if needs_ocr:
                    attempted_ocr = True
                    try:
                        ocr_text = _ocr_pdf_page(page).strip()
                        if (
                            garbled
                            and _has_meaningful_text(ocr_text)
                            and not _looks_garbled(ocr_text)
                        ):
                            # Scrambled selectable text is worse than any
                            # readable OCR text, even if OCR is shorter.
                            chosen = ocr_text
                        else:
                            chosen = _choose_best_extracted_text(
                                selectable_text,
                                ocr_text,
                            )
                        if (
                            _has_meaningful_text(ocr_text)
                            and chosen == ocr_text
                            and ocr_text != selectable_text
                        ):
                            text = chosen
                            ocr_applied = True
                        elif _has_meaningful_text(chosen):
                            text = chosen
                        else:
                            text = ""
                            warnings.append(
                                f"Page {page_number}: no selectable text "
                                "and OCR returned no text."
                            )
                    except PDFProcessingError as exc:
                        warnings.append(
                            f"Page {page_number}: {exc}"
                        )
                        text = selectable_text

                if garbled and not ocr_applied:
                    warnings.append(
                        f"Page {page_number}: selectable text looks "
                        "scrambled (broken PDF font encoding) and OCR did "
                        "not produce readable text (or Tesseract OCR is not "
                        "installed). Analysis of this page is unreliable."
                    )

                if (
                    attempted_ocr
                    and not ocr_applied
                    and _looks_suspiciously_incomplete(page, text)
                    and _has_meaningful_text(text)
                ):
                    warnings.append(
                        f"Page {page_number}: extracted text looks "
                        "suspiciously incomplete "
                        f"({len(text)} characters with embedded images)."
                    )

                if not _has_meaningful_text(text) and not attempted_ocr:
                    warnings.append(
                        f"Page {page_number}: empty page with no "
                        "extractable text."
                    )

                page_entry = {
                    "page_number": page_number,
                    "text": text,
                    "character_count": len(text),
                    "requires_ocr": not _has_meaningful_text(text),
                    "ocr_applied": ocr_applied,
                    "garbled_text_detected": garbled,
                }
                if not garbled and _has_meaningful_text(selectable_text):
                    tables = _extract_tables(page, selectable_text)
                    if tables:
                        page_entry["tables"] = tables
                pages.append(page_entry)
                report_stage("extract", page_number, page_total)

            return {
                "filename": path.name,
                "total_pages": len(document),
                "pages": pages,
                "warnings": warnings,
            }

    except PDFProcessingError:
        raise
    except Exception as exc:
        raise PDFProcessingError(
            f"Unable to process PDF: {exc}"
        ) from exc



def extract_structured_pdf(file_path: str) -> dict:
    """
    Extract document metadata and page-level text blocks.

    Each block preserves its page number and bounding box,
    making it easier to build source citations and RAG.
    """
    path = Path(file_path)

    if not path.is_file():
        raise PDFProcessingError(
            f"PDF file not found: {file_path}"
        )

    try:
        with pymupdf.open(path) as document:
            if not document.is_pdf:
                raise PDFProcessingError(
                    "The file is not a valid PDF."
                )

            pages = []

            for page_index, page in enumerate(document):
                blocks = []

                for block in page.get_text("blocks"):
                    x0, y0, x1, y1, text, *rest = block

                    # Skip image blocks and empty text.
                    if rest and rest[1] != 0:
                        continue

                    text = str(text).strip()

                    if not text:
                        continue

                    blocks.append({
                        "block_id": (
                            f"p{page_index + 1}_b{len(blocks) + 1}"
                        ),
                        "text": text,
                        "bbox": [x0, y0, x1, y1],
                    })

                pages.append({
                    "page_number": page_index + 1,
                    "blocks": blocks,
                    "requires_ocr": len(blocks) == 0,
                })

            return {
                "filename": path.name,
                "metadata": dict(document.metadata or {}),
                "total_pages": len(document),
                "pages": pages,
            }

    except PDFProcessingError:
        raise
    except Exception as exc:
        raise PDFProcessingError(
            f"Unable to process PDF: {exc}"
        ) from exc
