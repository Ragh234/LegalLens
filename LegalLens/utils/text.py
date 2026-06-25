import re


def normalize_whitespace(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def remove_repeated_headers_footers(page_text: str) -> str:
    lines = [line.strip() for line in page_text.splitlines() if line.strip()]
    if len(lines) <= 4:
        return page_text
    body = lines[:]
    if re.match(r"^(page\s*)?\d+(\s+of\s+\d+)?$", body[-1], re.IGNORECASE):
        body = body[:-1]
    return "\n".join(body)


def estimate_token_count(text: str) -> int:
    return max(1, int(len(text.split()) * 1.3))


def detect_heading(line: str) -> bool:
    stripped = line.strip()
    if len(stripped) < 4 or len(stripped) > 120:
        return False
    numbered = re.match(r"^(\d+\.|[A-Z]\.|ARTICLE\s+[IVXLC\d]+|SECTION\s+\d+)", stripped, re.IGNORECASE)
    title_case = stripped.istitle() and len(stripped.split()) <= 10
    upper = stripped.isupper() and len(stripped.split()) <= 12
    return bool(numbered or title_case or upper)

