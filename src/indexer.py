import json
import sys
import time
from pathlib import Path

import chromadb
import numpy as np
import voyageai
from dotenv import load_dotenv

from collections import Counter
from langdetect import DetectorFactory, LangDetectException, detect

DetectorFactory.seed = 0  # same text -> same language guess, every run

load_dotenv()

PROCESSED = Path("data/processed")
CHROMA_DIR = "chroma_db"
EMBED_MODEL = "voyage-4"
BATCH_SIZE = 64
DUPLICATE_THRESHOLD = 0.95

vo = voyageai.Client()


def load_chunks(strategy: str) -> list[dict]:
    with (PROCESSED / f"chunks_{strategy}.jsonl").open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def embed_documents(texts: list[str]) -> np.ndarray:
    """Embed in batches, retrying if Voyage rate-limits us."""
    vectors = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]
        for attempt in range(5):
            try:
                result = vo.embed(batch, model=EMBED_MODEL, input_type="document")
                break
            except voyageai.error.RateLimitError:
                wait = 20 * (attempt + 1)
                print(f"  rate limited, waiting {wait}s...")
                time.sleep(wait)
        else:
            raise RuntimeError("Gave up after 5 rate-limit retries")
        vectors.extend(result.embeddings)
        print(f"  embedded {min(start + BATCH_SIZE, len(texts))}/{len(texts)}")
    vecs = np.array(vectors, dtype=np.float32)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)  # unit length -> dot product = cosine


def deduplicate(vectors: np.ndarray) -> tuple[list[int], list[tuple[int, int, float]]]:
    """Keep a chunk only if it's not >95% similar to one we already kept."""
    similarity = vectors @ vectors.T  # every chunk vs every chunk
    keep, duplicates = [], []
    for i in range(len(vectors)):
        if keep:
            scores = similarity[i, keep]
            j = int(scores.argmax())
            if scores[j] > DUPLICATE_THRESHOLD:
                duplicates.append((i, keep[j], float(scores[j])))
                continue
        keep.append(i)
    return keep, duplicates


def to_metadata(chunk: dict) -> dict:
    # Chroma metadata can't store None, and "content" is stored as the document itself
    return {k: v for k, v in chunk.items() if k != "content" and v is not None}

def detect_language(text: str) -> str:
    """Guess the language of a chunk ('en', 'fr', ...)."""
    try:
        return detect(text)
    except LangDetectException:  # chunk is mostly numbers/symbols
        return "unknown"

def index_strategy(strategy: str) -> None:
    print(f"\n=== {strategy} ===")
    chunks = load_chunks(strategy)
    for chunk in chunks:
        chunk["language"] = detect_language(chunk["text"])
    # English first: dedup keeps the FIRST copy it sees, so English wins
    chunks.sort(key=lambda c: c["language"] != "en")
    print(f"Languages: {dict(Counter(c['language'] for c in chunks))}")
    vectors = embed_documents([c["content"] for c in chunks])

    keep, duplicates = deduplicate(vectors)
    print(f"Duplicates removed: {len(duplicates)} of {len(chunks)}")
    for i, j, score in duplicates[:3]:
        print(f"  [{score:.3f}] '{chunks[i]['heading'] or chunks[i]['title']}' "
              f"duplicates '{chunks[j]['heading'] or chunks[j]['title']}'")
        print(f"     dropped: {chunks[i]['text'][:120]!r}")

    kept_chunks = [chunks[i] for i in keep]
    kept_vectors = vectors[keep]

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    name = f"handbook_{strategy}"
    try:
        client.delete_collection(name)  # rebuild from scratch every run
    except Exception:
        pass
    collection = client.create_collection(name, metadata={"hnsw:space": "cosine"})
    collection.add(
        ids=[c["chunk_id"] for c in kept_chunks],
        embeddings=kept_vectors.tolist(),
        documents=[c["content"] for c in kept_chunks],
        metadatas=[to_metadata(c) for c in kept_chunks],
    )

    with (PROCESSED / f"indexed_{strategy}.jsonl").open("w", encoding="utf-8") as f:
        for chunk in kept_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"Indexed {collection.count()} chunks into '{name}'")


if __name__ == "__main__":
    strategies = sys.argv[1:] or ["fixed", "header"]
    for strategy in strategies:
        index_strategy(strategy)