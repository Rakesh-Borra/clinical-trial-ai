
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
