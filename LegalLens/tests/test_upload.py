from utils.upload import UploadValidationError, sanitize_filename, validate_pdf_bytes


def test_sanitize_filename_removes_path_and_unsafe_chars() -> None:
    assert sanitize_filename("../Bad Contract!!.PDF") == "Bad_Contract.pdf"


def test_validate_pdf_rejects_non_pdf_bytes() -> None:
    try:
        validate_pdf_bytes(b"not a pdf", "contract.pdf")
    except UploadValidationError as exc:
        assert "valid PDF" in str(exc)
    else:
        raise AssertionError("Expected UploadValidationError")

