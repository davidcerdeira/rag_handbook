import json
from pathlib import Path
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

DOCS_FILE = Path("data/processed/documents.jsonl")
OUT_DIR = Path("data/processed")
CHUNK_SIZE = 1000      # characters per chunk
CHUNK_OVERLAP = 150    # characters shared between neighbouring chunks
MIN_CHUNK_CHARS = 50   # drop tiny leftovers (e.g. a lone heading)

HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]
header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS, strip_headers=False)
size_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)


def load_documents() -> list[dict]:
    with DOCS_FILE.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def make_chunk(doc: dict, text: str, strategy: str, index: int, heading: str) -> dict:
    prefix = doc["title"] + (f" > {heading}" if heading else "")
    return {
        "chunk_id": f"{strategy}:{doc['id']}:{index}",
        "doc_id": doc["id"],
        "source": doc["source"],
        "section": doc["section"],
        "title": doc["title"],
        "page": doc["page"],
        "heading": heading,
        "strategy": strategy,
        "chunk_index": index,
        "text": text,                       # the raw chunk (shown to the user / cited)
        "content": f"{prefix}\n\n{text}",   # what we embed and search over
        "char_count": len(text),
    }


def fixed_chunks(doc: dict) -> list[dict]:
    """Baseline: slide a 1000-char window, stepping 850 chars each time."""
    text, step = doc["text"], CHUNK_SIZE - CHUNK_OVERLAP
    pieces = [text[i:i + CHUNK_SIZE] for i in range(0, len(text), step)]
    pieces = [p for p in pieces if len(p.strip()) >= MIN_CHUNK_CHARS]
    return [make_chunk(doc, p, "fixed", i, "") for i, p in enumerate(pieces)]


def header_chunks(doc: dict) -> list[dict]:
    """Structure-aware: split on headings, then split long sections by size."""
    sections = header_splitter.split_text(doc["text"])
    pieces = size_splitter.split_documents(sections)
    chunks = []
    for piece in pieces:
        if len(piece.page_content.strip()) < MIN_CHUNK_CHARS:
            continue
        heading = " > ".join(piece.metadata[k] for k in ("h1", "h2", "h3") if k in piece.metadata)
        chunks.append(make_chunk(doc, piece.page_content, "header", len(chunks), heading))
    return chunks


STRATEGIES = {"fixed": fixed_chunks, "header": header_chunks}


def save(chunks: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    docs = load_documents()
    for name, chunk_fn in STRATEGIES.items():
        chunks = [c for doc in docs for c in chunk_fn(doc)]
        save(chunks, OUT_DIR / f"chunks_{name}.jsonl")
        sizes = [c["char_count"] for c in chunks]
        print(f"[{name}] {len(chunks)} chunks | avg {sum(sizes) // len(sizes)} chars "
              f"| min {min(sizes)} | max {max(sizes)}")

        sample = chunks[len(chunks) // 2]  # a chunk from the middle, not the intro
        print(f"--- sample [{name}] ---")
        print(sample["content"][:500])
        print()