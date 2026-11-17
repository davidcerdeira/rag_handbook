import json
import re
from pathlib import Path
from pypdf import PdfReader

RAW_DIR = Path("data/raw")
OUT_FILE = Path("data/processed/documents.jsonl")
MIN_CHARS = 200  # skip near-empty pages (e.g. index pages that only list links)

FRONT_MATTER = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)
SHORTCODE = re.compile(r"\{\{[<%].*?[>%]\}\}", re.DOTALL)
HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
SUMMARY = re.compile(r"<summary>(.*?)</summary>", re.DOTALL)
HTML_TAG = re.compile(r"</?(?:details|summary|div|span|br|p|sup|sub|img|a)\b[^>]*>", re.IGNORECASE)


def clean_markdown(raw: str) -> tuple[str | None, str]:
    """Strip website-only clutter. Returns (title, clean_text)."""
    title = None
    match = FRONT_MATTER.match(raw)
    if match:
        found = re.search(r'^title:\s*["\']?(.+?)["\']?\s*$', match.group(1), re.MULTILINE)
        if found:
            title = found.group(1)
        raw = raw[match.end():]
    text = SHORTCODE.sub("", raw)
    text = HTML_COMMENT.sub("", text)
    text = SUMMARY.sub(r"\1:", text) 
    text = HTML_TAG.sub("", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text) 
    text = re.sub(r"\n{3,}", "\n\n", text)
    return title, text.strip()


def fallback_title(path: Path) -> str:
    # "_index.md" files are named after their folder
    name = path.parent.name if path.stem == "_index" else path.stem
    return name.replace("-", " ").title()


def load_markdown(path: Path) -> list[dict]:
    title, text = clean_markdown(path.read_text(encoding="utf-8"))
    return [{"title": title or fallback_title(path), "text": text, "page": None}]


def load_pdf(path: Path) -> list[dict]:
    # one record per page, so citations can point to a page number
    pages = []
    for i, page in enumerate(PdfReader(path).pages, start=1):
        pages.append({"title": path.stem, "text": (page.extract_text() or "").strip(), "page": i})
    return pages


LOADERS = {".md": load_markdown, ".pdf": load_pdf}


def load_all() -> tuple[list[dict], int]:
    docs, skipped = [], 0
    for path in sorted(RAW_DIR.rglob("*")):
        loader = LOADERS.get(path.suffix.lower())
        if not loader:
            continue
        relative = path.relative_to(RAW_DIR)
        for record in loader(path):
            if len(record["text"]) < MIN_CHARS:
                skipped += 1
                continue
            docs.append({
                "id": f"{relative}#p{record['page']}" if record["page"] else str(relative),
                "source": str(relative),
                "section": relative.parts[0],  # people-policies / total-rewards
                "title": record["title"],
                "page": record["page"],
                "text": record["text"],
                "char_count": len(record["text"]),
            })
    return docs, skipped


if __name__ == "__main__":
    docs, skipped = load_all()
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUT_FILE.open("w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    total = sum(d["char_count"] for d in docs)
    print(f"Loaded {len(docs)} documents ({skipped} skipped as near-empty)")
    print(f"Total characters: {total:,} | average per doc: {total // max(len(docs), 1):,}")
    for section in sorted({d["section"] for d in docs}):
        print(f"  {section}: {sum(d['section'] == section for d in docs)} docs")
    biggest = max(docs, key=lambda d: d["char_count"])
    print(f"Largest: {biggest['title']} ({biggest['char_count']:,} chars)")
    print("\n--- Sample (first 600 chars) ---")
    print(docs[0]["title"], "|", docs[0]["source"])
    print(docs[0]["text"][:600])