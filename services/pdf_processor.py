
from pathlib import Path

import pymupdf


# Short selectable text with embedded images often means incomplete extraction.
_SUSPICIOUSLY_SHORT_CHARS = 40


class PDFProcessingError(Exception):
    """Raised when a PDF cannot be processed."""


def _has_meaningful_text(text: str) -> bool:
    """Return True when extracted text is non-empty after trimming."""
    return bool(text and text.strip())


def _extract_selectable_text(page) -> str:
    """Extract selectable text in approximate reading order."""
    return page.get_text("text", sort=True).strip()


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

            for index, page in enumerate(document):
                page_number = index + 1
                selectable_text = _extract_selectable_text(page)
                text = selectable_text
                ocr_applied = False
                attempted_ocr = False

                needs_ocr = (
                    not _has_meaningful_text(selectable_text)
                    or _looks_suspiciously_incomplete(page, selectable_text)
                )

                if needs_ocr:
                    attempted_ocr = True
                    try:
                        ocr_text = _ocr_pdf_page(page).strip()
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

                pages.append({
                    "page_number": page_number,
                    "text": text,
                    "character_count": len(text),
                    "requires_ocr": not _has_meaningful_text(text),
                    "ocr_applied": ocr_applied,
                })

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
