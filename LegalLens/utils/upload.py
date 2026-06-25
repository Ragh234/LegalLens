import hashlib
import re
from pathlib import Path
from uuid import uuid4

import fitz

from config.settings import settings
from utils.models import UploadedContract


class UploadValidationError(ValueError):
    pass


def sanitize_filename(filename: str) -> str:
    name = Path(filename).name
    stem = Path(name).stem
    suffix = Path(name).suffix.lower()
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    if not safe_stem:
        safe_stem = "contract"
    return f"{safe_stem[:80]}{suffix}"


def validate_pdf_bytes(content: bytes, original_name: str) -> None:
    max_bytes = settings.upload_max_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise UploadValidationError(f"{original_name} exceeds the {settings.upload_max_mb} MB upload limit.")
    if Path(original_name).suffix.lower() != ".pdf":
        raise UploadValidationError("Only PDF files are supported.")
    if not content.startswith(b"%PDF"):
        raise UploadValidationError(f"{original_name} does not appear to be a valid PDF.")
    try:
        with fitz.open(stream=content, filetype="pdf") as document:
            if document.page_count == 0:
                raise UploadValidationError(f"{original_name} has no pages.")
    except UploadValidationError:
        raise
    except Exception as exc:
        raise UploadValidationError(f"{original_name} could not be opened as a PDF.") from exc


def content_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def save_validated_upload(uploaded_file) -> UploadedContract:
    content = uploaded_file.getvalue()
    validate_pdf_bytes(content, uploaded_file.name)

    settings.contracts_dir.mkdir(parents=True, exist_ok=True)
    digest = content_hash(content)
    existing = next(settings.contracts_dir.glob(f"{digest}_*.pdf"), None)
    if existing:
        return UploadedContract(uploaded_file.name, existing.name, existing, len(content), duplicate=True)

    safe_name = sanitize_filename(uploaded_file.name)
    stored_name = f"{digest}_{uuid4().hex[:8]}_{safe_name}"
    destination = settings.contracts_dir / stored_name
    destination.write_bytes(content)
    return UploadedContract(uploaded_file.name, stored_name, destination, len(content), duplicate=False)
