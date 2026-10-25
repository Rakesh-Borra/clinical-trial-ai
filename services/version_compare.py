"""
Version comparison: what an amended protocol adds, removes and changes.

The comparison is deterministic and makes no model calls. It works on the
statements each version already has (written by the existing plain-language
agent and verified against their own source page), matches them by their
protocol wording and reports additions, removals and modifications. The
patient-friendly explanation of every change reuses the statement wording that
agent produced.

Each change is then checked against the *original PDFs* of the versions
involved: the protocol excerpt must appear on the cited page. When either
original PDF is unavailable the comparison is marked as not source-backed and
is never presented as verified.
"""

from __future__ import annotations

import difflib
import hashlib
import os
import tempfile
from typing import Any, Callable, Dict, List, Mapping, Optional

from models.plain_language_models import GuideItem, PatientGuide
from services import review_store, version_store
from services.source_verifier import (
    authoritative_page_texts,
    compact,
    normalize_text,
    page_contains_excerpt,
    page_supports_paraphrase,
)

SIMILARITY = 0.55
PageTexts = Mapping[int, str]
PageTextLoader = Callable[[str], Optional[PageTexts]]

ADDED, REMOVED, MODIFIED = "added", "removed", "modified"
VERIFIED, UNVERIFIED, UNAVAILABLE = "verified", "unverified", "unavailable"

UNAVAILABLE_NOTICE = (
    "Full source-backed version comparison is unavailable because the original PDF "
    "for {versions} is not stored. This comparison uses the saved statements only "
    "and has NOT been verified against the protocol documents."
)


def live_items(guide: PatientGuide) -> List[GuideItem]:
    return [item for item in guide.all_items() if not item.rejected]


def guide_fingerprint(guide: PatientGuide) -> str:
    """Identifies the statements a comparison was made from (edits make it stale)."""
    rows = sorted(f"{i.section}|{i.original_text}|{i.plain_text}" for i in live_items(guide))
    return hashlib.sha256("\n".join(rows).encode()).hexdigest()[:20]


def _side(item: GuideItem) -> Dict[str, Any]:
    return {
        "plain_text": item.plain_text,
        "original_text": item.original_text,
        "page": item.source_page,
    }


def _match(old_items: List[GuideItem], new_items: List[GuideItem]):
    """Pair statements of one section: unchanged, modified, and the leftovers."""
    old_left = list(old_items)
    new_left: List[GuideItem] = []
    for item in new_items:
        key = compact(item.original_text)
        same = next((o for o in old_left if compact(o.original_text) == key), None)
        if same is not None:
            old_left.remove(same)  # identical protocol wording: unchanged
        else:
            new_left.append(item)
    pairs = []
    for item in list(new_left):
        scored = [
            (difflib.SequenceMatcher(
                None, normalize_text(o.original_text), normalize_text(item.original_text)
            ).ratio(), o)
            for o in old_left
        ]
        best = max(scored, key=lambda pair: pair[0], default=None)
        if best and best[0] >= SIMILARITY:
            pairs.append((best[1], item))
            old_left.remove(best[1])
            new_left.remove(item)
    return pairs, old_left, new_left


def diff_guides(old: PatientGuide, new: PatientGuide) -> List[Dict[str, Any]]:
    """Additions, removals and modifications between two guides, by section."""
    changes: List[Dict[str, Any]] = []
    for old_section, new_section in zip(old.sections(), new.sections()):
        old_items = [i for i in old_section.items if not i.rejected]
        new_items = [i for i in new_section.items if not i.rejected]
        pairs, removed, added = _match(old_items, new_items)
        title = new_section.title
        for before, after in pairs:
            changes.append({
                "change": MODIFIED, "section": title, "old": _side(before), "new": _side(after),
                "explanation": f"This has changed. Before: {before.plain_text} Now: {after.plain_text}",
            })
        for item in added:
            changes.append({
                "change": ADDED, "section": title, "old": None, "new": _side(item),
                "explanation": f"New in this version: {item.plain_text}",
            })
        for item in removed:
            changes.append({
                "change": REMOVED, "section": title, "old": _side(item), "new": None,
                "explanation": f"No longer in this version: {item.plain_text}",
            })
    return changes


def _supported(side: Optional[Mapping[str, Any]], pages: Optional[PageTexts]) -> Optional[bool]:
    """Is this statement's protocol excerpt on its cited page? None when no PDF text."""
    if side is None:
        return True
    if pages is None:
        return None
    page = side.get("page")
    text = pages.get(page) if page is not None else None
    if not text:
        return False
    excerpt = (side.get("original_text") or "").strip()
    if excerpt and page_contains_excerpt(excerpt, text):
        return True
    # Faithful plain-language paraphrase of the same fact (numbers stay strict).
    plain = (side.get("plain_text") or "").strip()
    if plain and page_supports_paraphrase(plain, text):
        return True
    return False


def default_page_texts(protocol_id: str) -> Optional[PageTexts]:
    """Authoritative page text of the stored original PDF, or None when unavailable."""
    data = version_store.stored_pdf_bytes(protocol_id)
    if data is None:
        return None
    from services.pdf_processor import PDFProcessingError, extract_pdf

    fd, path = tempfile.mkstemp(suffix=".pdf")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        return authoritative_page_texts(extract_pdf(path))
    except PDFProcessingError:
        return None
    finally:
        os.unlink(path)


def compare_versions(
    new_protocol_id: str,
    *,
    page_text_loader: PageTextLoader | None = None,
) -> Dict[str, Any]:
    """
    Compare a version with the one before it, check every change against both
    original PDFs, and save the result (which resets any earlier approval).
    """
    previous = version_store.previous_version(new_protocol_id)
    if previous is None:
        raise version_store.VersionError("This is the first version, so there is nothing to compare.")
    new_draft = review_store.load_draft(new_protocol_id)
    old_draft = review_store.load_draft(previous["protocol_id"])
    if new_draft is None or old_draft is None:
        raise version_store.VersionError("Both versions need a saved guide before they can be compared.")

    changes = diff_guides(old_draft["guide"], new_draft["guide"])
    page_text_loader = page_text_loader or default_page_texts
    old_pages = page_text_loader(previous["protocol_id"])
    new_pages = page_text_loader(new_protocol_id)
    missing = [
        f"Version {label}"
        for label, pages in ((previous["version_label"], old_pages),
                             (version_store.get_version(new_protocol_id)["version_label"], new_pages))
        if pages is None
    ]
    source_backed = not missing
    for change in changes:
        if not source_backed:
            change["evidence"] = UNAVAILABLE
            continue
        ok = _supported(change["old"], old_pages) and _supported(change["new"], new_pages)
        change["evidence"] = VERIFIED if ok else UNVERIFIED

    diff = {
        "previous_protocol_id": previous["protocol_id"],
        "previous_label": previous["version_label"],
        "new_label": version_store.get_version(new_protocol_id)["version_label"],
        "guide_fingerprint": guide_fingerprint(new_draft["guide"]),
        "source_backed": source_backed,
        "notice": "" if source_backed else UNAVAILABLE_NOTICE.format(versions=" and ".join(missing)),
        "changes": changes,
    }
    source_verified = source_backed and all(c["evidence"] == VERIFIED for c in changes)
    version_store.save_comparison(new_protocol_id, previous["protocol_id"], diff, source_verified)
    return diff


def comparison_block_reason(protocol_id: str, guide: Optional[PatientGuide]) -> Optional[str]:
    """Why publication must wait for the version comparison, or None."""
    previous = version_store.previous_version(protocol_id)
    if previous is None:
        return None
    label = previous["version_label"]
    saved = version_store.load_comparison(protocol_id)
    if saved is None or saved["previous_protocol_id"] != previous["protocol_id"]:
        return f"Compare this version with Version {label} and approve the changes before publishing."
    if guide is None or saved["diff"].get("guide_fingerprint") != guide_fingerprint(guide):
        return (
            f"The statements changed after the comparison with Version {label}. "
            "Run the comparison again and approve it before publishing."
        )
    if not saved["approved_at"]:
        return f"Approve the changes from Version {label} before publishing."
    return None


def approval_block_reason(saved: Mapping[str, Any]) -> Optional[str]:
    """Why a comparison cannot be approved (changes unsupported by the PDFs)."""
    diff = saved["diff"]
    if diff["source_backed"] and any(c["evidence"] != VERIFIED for c in diff["changes"]):
        return (
            "Some changes could not be found on their cited pages of the original PDFs. "
            "Fix or reject those statements and compare again."
        )
    return None
