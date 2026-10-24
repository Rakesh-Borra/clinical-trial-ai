"""
Run the protocol analysis pipeline on a PDF (calls Ollama; manual use only).

    python -m scripts.test_pdf_pipeline path/to/protocol.pdf

Without a path, a small synthetic two-page protocol is generated in a
temporary folder and analysed instead.
"""
import json
import sys
import tempfile
from pathlib import Path

import pymupdf

from services.protocol_pipeline import analyze_protocol_pdf

SYNTHETIC_PAGE_1 = """PROTOCOL TITLE: Type 2 Diabetes Treatment Study
PROTOCOL ID: T2D-001

INCLUSION CRITERIA:
1. Participants must be between 18 and 65 years old.
2. Participants must have Type 2 Diabetes.

EXCLUSION CRITERIA:
1. Participants with severe kidney disease are excluded.
"""
SYNTHETIC_PAGE_2 = """STUDY PROCEDURES:
1. Blood glucose testing.
2. Blood pressure measurement.

VISIT SCHEDULE:
Screening Visit: Day 0.
Follow-up Visit: Day 30.

SAFETY INFORMATION:
Participants will be monitored for adverse events.
"""


def create_synthetic_pdf(path: Path) -> Path:
    """Write the synthetic two-page clinical trial protocol to `path`."""
    document = pymupdf.open()
    for text in (SYNTHETIC_PAGE_1, SYNTHETIC_PAGE_2):
        document.new_page().insert_text((50, 70), text, fontsize=10)
    document.save(path)
    document.close()
    return path


def resolve_pdf(argv: list) -> Path:
    """The PDF given on the command line, or a freshly generated synthetic one."""
    if argv:
        path = Path(argv[0]).expanduser()
        if not path.is_file():
            raise SystemExit(f"PDF not found: {path}")
        return path
    path = create_synthetic_pdf(Path(tempfile.mkdtemp(prefix="ctai_")) / "synthetic.pdf")
    print(f"Created synthetic protocol PDF: {path}")
    return path


def main(argv: list) -> None:
    pdf_path = resolve_pdf(argv)
    result = analyze_protocol_pdf(str(pdf_path))

    print("\n=== PDF PROTOCOL ANALYSIS ===\n")
    print(result["analysis"].model_dump_json(indent=2))

    print("\n=== SOURCE VERIFICATION ===\n")
    print(json.dumps(result["verification"], indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])

