
import re
from models.protocol_models import ProtocolAnalysis


def normalize_text(text: str) -> str:
    """Normalize whitespace and case for comparison."""
    return re.sub(r"\s+", " ", text).strip().casefold()


def verify_source_pages(
    analysis: ProtocolAnalysis,
    pdf_result: dict,
) -> list[dict]:
    """
    Check whether each extracted item appears on its claimed PDF page.

    This first version uses exact text matching after normalization.
    It does not verify paraphrases or clinical meaning.
    """

    page_texts = {
        page["page_number"]: normalize_text(page["text"])
        for page in pdf_result["pages"]
    }

    results = []

    fields = {
        "inclusion_criteria": "description",
        "exclusion_criteria": "description",
        "procedures": "name",
        "visits": "visit_name",
        "safety_information": "description",
    }

    for category, text_field in fields.items():
        for item in getattr(analysis, category):
            statement = getattr(item, text_field)
            source_page = item.source_page

            if source_page is None:
                status = "missing_page"
            elif source_page not in page_texts:
                status = "invalid_page"
            elif normalize_text(statement) in page_texts[source_page]:
                status = "exact_match"
            else:
                status = "not_verified"

            results.append({
                "category": category,
                "statement": statement,
                "source_page": source_page,
                "status": status,
            })

    return results
