"""The manual PDF pipeline script no longer depends on a bundled sample PDF."""

import pytest

from scripts import test_pdf_pipeline as script


def test_script_uses_the_given_pdf(tmp_path):
    pdf = tmp_path / "mine.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    assert script.resolve_pdf([str(pdf)]) == pdf


def test_missing_pdf_is_a_clear_error(tmp_path):
    with pytest.raises(SystemExit, match="PDF not found"):
        script.resolve_pdf([str(tmp_path / "nope.pdf")])


def test_without_a_path_a_temporary_synthetic_pdf_is_created(tmp_path, capsys):
    path = script.resolve_pdf([])
    assert path.is_file() and "data" not in path.parts[-3:-1]
    import pymupdf
    text = "".join(page.get_text() for page in pymupdf.open(path))
    assert "Type 2 Diabetes Treatment Study" in text
