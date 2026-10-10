from pathlib import Path
import pymupdf

from services.protocol_pipeline import analyze_protocol_pdf


pdf_path = Path("data/sample_clinical_protocol.pdf")

# Generate a synthetic two-page clinical trial protocol
document = pymupdf.open()

page1 = document.new_page()
page1.insert_text(
    (50, 70),
    """PROTOCOL TITLE: Type 2 Diabetes Treatment Study
PROTOCOL ID: T2D-001

INCLUSION CRITERIA:
1. Participants must be between 18 and 65 years old.
2. Participants must have Type 2 Diabetes.

EXCLUSION CRITERIA:
1. Participants with severe kidney disease are excluded.
""",
    fontsize=10,
)

page2 = document.new_page()
page2.insert_text(
    (50, 70),
    """STUDY PROCEDURES:
1. Blood glucose testing.
2. Blood pressure measurement.

VISIT SCHEDULE:
Screening Visit: Day 0.
Follow-up Visit: Day 30.

SAFETY INFORMATION:
Participants will be monitored for adverse events.
""",
    fontsize=10,
)

document.save(pdf_path)
document.close()

print(f"Created synthetic protocol PDF: {pdf_path}")

# Run our complete clinical trial analysis pipeline
analysis = analyze_protocol_pdf(str(pdf_path))

print("\n=== PDF PROTOCOL ANALYSIS ===\n")
print(analysis.model_dump_json(indent=2))