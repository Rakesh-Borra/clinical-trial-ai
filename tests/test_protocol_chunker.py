import pytest

from services.protocol_chunker import chunk_protocol, detect_sections


def _source_page_text(chunks, page_number, *, include_overlap=False):
    return "".join(
        page.text
        for chunk in chunks
        for page in chunk.pages
        if page.page_number == page_number
        and (include_overlap or not page.overlap)
    )


def test_empty_input_returns_no_chunks_and_explicit_warning():
    result = chunk_protocol("")

    assert result.chunks == ()
    assert any("Input is empty" in warning for warning in result.warnings)


def test_pages_are_kept_in_order_with_original_numbers_and_text():
    protocol = (
        "[SOURCE PAGE 3]\nFirst page text.\n\n"
        "[SOURCE PAGE 7]\nSecond page text."
    )

    result = chunk_protocol(protocol, max_chars=35)

    source_pages = [
        page
        for chunk in result.chunks
        for page in chunk.pages
        if not page.overlap
    ]
    assert [page.page_number for page in source_pages] == [3, 7]
    assert "".join(page.text for page in source_pages) == (
        "First page text.\n\nSecond page text."
    )
    assert all(chunk.character_count <= 35 for chunk in result.chunks)


def test_unclassified_pages_are_retained_not_filtered():
    protocol = (
        "[SOURCE PAGE 1]\nPROTOCOL TITLE: Example\n"
        "[SOURCE PAGE 2]\nUnclassified trial detail.\n"
        "[SOURCE PAGE 3]\nSAFETY INFORMATION\nMonitor participants."
    )

    result = chunk_protocol(protocol, max_chars=100)

    assert [
        page.page_number
        for chunk in result.chunks
        for page in chunk.pages
        if not page.overlap
    ] == [1, 2, 3]
    assert any(
        "Unclassified trial detail." in page.text
        for chunk in result.chunks
        for page in chunk.pages
    )


def test_malformed_markers_are_reported_and_retained():
    protocol = (
        "[SOURCE PAGE 1]\nValid text.\n"
        "[SOURCE PAGE zero]\nMalformed text.\n"
        "[SOURCE PAGE 2]\nMore valid text."
    )

    result = chunk_protocol(protocol, max_chars=200)

    assert any("Malformed source-page marker" in warning for warning in result.warnings)
    assert [page.page_number for page in result.chunks[0].pages] == [1, 2]
    assert "[SOURCE PAGE zero]" in result.chunks[0].pages[0].text
    assert "Malformed text." in result.chunks[0].pages[0].text


def test_unmarked_input_is_retained_with_unknown_page_warning():
    result = chunk_protocol("No page marker; retain this text.")

    assert len(result.chunks) == 1
    assert result.chunks[0].pages[0].page_number is None
    assert result.chunks[0].text == "No page marker; retain this text."
    assert any("No valid source-page markers" in w for w in result.warnings)


def test_oversized_page_splits_into_bounded_fragments_without_loss():
    body = "0123456789" * 7
    result = chunk_protocol(
        f"[SOURCE PAGE 12]\n{body}",
        max_chars=24,
    )

    fragments = [
        page
        for chunk in result.chunks
        for page in chunk.pages
        if not page.overlap
    ]
    assert len(fragments) > 1
    assert all(page.page_number == 12 for page in fragments)
    assert fragments[0].continuation is False
    assert all(page.continuation for page in fragments[1:])
    assert "".join(page.text for page in fragments) == body
    assert all(chunk.character_count <= 24 for chunk in result.chunks)


def test_oversized_page_prefers_line_boundaries_and_marks_continuations():
    body = (
        "First complete line stays together.\n"
        "Second complete line stays together.\n"
        "Third line also stays together."
    )

    result = chunk_protocol(f"[SOURCE PAGE 4]\n{body}", max_chars=52)
    fragments = [
        page
        for chunk in result.chunks
        for page in chunk.pages
        if not page.overlap
    ]

    assert len(fragments) > 1
    assert "".join(page.text for page in fragments) == body
    assert fragments[0].text.endswith("\n")
    assert all(page.continuation for page in fragments[1:])
    assert all(chunk.character_count <= 52 for chunk in result.chunks)


def test_oversized_page_keeps_numeric_threshold_and_units_together():
    body = (
        "Eligibility requirement: HbA1c >= 7.0% (53 mmol/mol) "
        "and eGFR >= 60 mL/min/1.73m2."
    )

    result = chunk_protocol(f"[SOURCE PAGE 5]\n{body}", max_chars=58)
    fragments = [
        page
        for chunk in result.chunks
        for page in chunk.pages
        if not page.overlap
    ]

    assert "".join(page.text for page in fragments) == body
    joined_fragments = [page.text for page in fragments]
    assert any("HbA1c >= 7.0%" in fragment for fragment in joined_fragments)
    assert any("53 mmol/mol" in fragment for fragment in joined_fragments)
    assert any("60 mL/min/1.73m2" in fragment for fragment in joined_fragments)
    assert all(chunk.character_count <= 58 for chunk in result.chunks)


def test_unavoidable_hard_split_preserves_text_and_marks_continuation():
    body = "X" * 80
    result = chunk_protocol(f"[SOURCE PAGE 6]\n{body}", max_chars=25)
    fragments = [
        page
        for chunk in result.chunks
        for page in chunk.pages
        if not page.overlap
    ]

    assert "".join(page.text for page in fragments) == body
    assert fragments[0].continuation is False
    assert all(page.continuation for page in fragments[1:])
    assert all(chunk.character_count <= 25 for chunk in result.chunks)


def test_max_chars_is_configurable_and_invalid_sizes_are_explicit():
    result = chunk_protocol(
        "[SOURCE PAGE 1]\nA moderately long page body.",
        max_chars=22,
    )

    assert all(chunk.character_count <= 22 for chunk in result.chunks)
    with pytest.raises(ValueError, match="greater than zero"):
        chunk_protocol("text", max_chars=0)
    with pytest.raises(ValueError, match="larger than the source-page marker"):
        chunk_protocol("[SOURCE PAGE 1]\nText", max_chars=15)
    with pytest.raises(ValueError, match="cannot be negative"):
        chunk_protocol("text", overlap_pages=-1)


def test_optional_overlap_duplicates_complete_prior_page_in_order():
    protocol = (
        "[SOURCE PAGE 1]\nAAAA\n"
        "[SOURCE PAGE 2]\nBBBB\n"
        "[SOURCE PAGE 3]\nCCCC"
    )

    result = chunk_protocol(protocol, max_chars=44, overlap_pages=1)

    assert len(result.chunks) >= 2
    assert result.chunks[1].pages[0].page_number == 2
    assert result.chunks[1].pages[0].overlap is True
    primary_text = "".join(
        page.text
        for chunk in result.chunks
        for page in chunk.pages
        if not page.overlap
    )
    assert primary_text == "AAAA\nBBBB\nCCCC"
    assert all(chunk.character_count <= 44 for chunk in result.chunks)


def test_overlap_is_omitted_if_it_would_leave_no_room_for_new_text():
    protocol = (
        "[SOURCE PAGE 1]\n123456789012\n"
        "[SOURCE PAGE 2]\nB"
    )

    result = chunk_protocol(protocol, max_chars=38, overlap_pages=1)

    assert all(chunk.character_count <= 38 for chunk in result.chunks)
    assert any("overlap was omitted" in warning for warning in result.warnings)
    assert "".join(
        page.text
        for chunk in result.chunks
        for page in chunk.pages
        if not page.overlap
    ) == "123456789012\nB"


def test_overlap_unavailable_for_fragmented_page_warns_without_dropping_it():
    body = "Line one is deliberately long.\nLine two is also deliberately long."
    protocol = f"[SOURCE PAGE 8]\n{body}\n\n[SOURCE PAGE 9]\nNext page."

    result = chunk_protocol(protocol, max_chars=32, overlap_pages=1)

    assert any("overlap was unavailable or reduced" in warning for warning in result.warnings)
    assert "".join(
        page.text
        for chunk in result.chunks
        for page in chunk.pages
        if not page.overlap and page.page_number == 8
    ) == body + "\n\n"
    assert any(
        page.page_number == 9 and not page.overlap
        for chunk in result.chunks
        for page in chunk.pages
    )


def test_page_marker_input_order_is_preserved_without_sorting():
    protocol = (
        "[SOURCE PAGE 12]\nFirst in input.\n"
        "[SOURCE PAGE 3]\nSecond in input.\n"
        "[SOURCE PAGE 3]\nRepeated page marker."
    )

    result = chunk_protocol(protocol, max_chars=100)
    primary_pages = [
        page.page_number
        for chunk in result.chunks
        for page in chunk.pages
        if not page.overlap
    ]

    assert primary_pages == [12, 3, 3]


def test_section_detection_requires_heading_shaped_lines_and_keeps_all_pages():
    assert detect_sections(
        "INCLUSION CRITERIA\nEXCLUSION CRITERIA\n"
        "SAFETY INFORMATION\nVISIT SCHEDULE\nSTUDY PROCEDURES"
    ) == ("eligibility", "exclusions", "safety", "visits", "procedures")

    protocol = (
        "[SOURCE PAGE 1]\nINCLUSION CRITERIA\nAge 18+\n"
        "[SOURCE PAGE 2]\nThe inclusion criteria mention visit safety "
        "and study procedures in prose."
    )
    result = chunk_protocol(protocol, max_chars=100)

    assert {page.page_number for chunk in result.chunks for page in chunk.pages} == {1, 2}
    assert result.chunks[0].sections == ("eligibility",)
    assert result.chunks[1].sections == ()
