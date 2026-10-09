
from pathlib import Path

import pymupdf


class PDFProcessingError(Exception):
    """Raised when a PDF cannot be processed."""


def extract_pdf(file_path: str) -> dict:
    """
    Extract text from a PDF while preserving page numbers.
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

            for index, page in enumerate(document):
                text = page.get_text("text").strip()

                pages.append({
                    "page_number": index + 1,
                    "text": text,
                    "character_count": len(text),
                    "requires_ocr": not bool(text),
                })

            return {
                "filename": path.name,
                "total_pages": len(document),
                "pages": pages,
            }

    except PDFProcessingError:
        raise
    except Exception as exc:
        raise PDFProcessingError(
            f"Unable to process PDF: {exc}"
        ) from exc
