import sys
import json
import re

from rank_bm25 import BM25Okapi
from functools import lru_cache
from pathlib import Path

import chromadb
import voyageai
from dotenv import load_dotenv

load_dotenv()


CHROMA_DIR = "chroma_db"
EMBED_MODEL = "voyage-4"
PROCESSED = Path("data/processed")
RRF_K = 60  # standard RRF constant: dampens the advantage of a single #1 rank
RERANK_MODEL = "rerank-2.5"


vo = voyageai.Client()
chroma = chromadb.PersistentClient(path=CHROMA_DIR)


def embed_query(question: str) -> list[float]:
    # "query" (not "document"): Voyage optimizes questions differently from passages
    return vo.embed([question], model=EMBED_MODEL, input_type="query").embeddings[0]


def dense_search(question: str, strategy: str = "header", k: int = 20) -> list[dict]:
    """Find the k chunks closest in MEANING to the question."""
    collection = chroma.get_collection(f"handbook_{strategy}")
    result = collection.query(query_embeddings=[embed_query(question)], n_results=k)
    return [
        {"chunk_id": chunk_id, "score": 1 - distance, **metadata, "content": document}
        for chunk_id, metadata, document, distance in zip(
            result["ids"][0], result["metadatas"][0], result["documents"][0], result["distances"][0]
        )
    ]

def tokenize(text: str) -> list[str]:
    """'What is PRSI?' -> ['what', 'is', 'prsi']"""
    return re.findall(r"\w+", text.lower())


@lru_cache(maxsize=None)  # build each BM25 index once, then reuse it
def load_bm25(strategy: str) -> tuple[BM25Okapi, list[dict]]:
    # same chunk list as ChromaDB (saved by the indexer), so both searches see the same pieces
    with (PROCESSED / f"indexed_{strategy}.jsonl").open(encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]
    bm25 = BM25Okapi([tokenize(c["content"]) for c in chunks])
    return bm25, chunks


def sparse_search(question: str, strategy: str = "header", k: int = 20) -> list[dict]:
    """Find the k chunks that best match the question's exact WORDS."""
    bm25, chunks = load_bm25(strategy)
    scores = bm25.get_scores(tokenize(question))
    top = scores.argsort()[::-1][:k]  # indexes of the highest scores
    return [{**chunks[i], "score": float(scores[i])} for i in top if scores[i] > 0]

def rrf_fuse(ranked_lists: dict[str, list[dict]], weights: dict[str, float], k: int = RRF_K) -> list[dict]:
    """Merge ranked lists by POSITION (not score): each appearance earns weight / (k + rank)."""
    fused: dict[str, dict] = {}
    for name, results in ranked_lists.items():
        for rank, r in enumerate(results, start=1):
            entry = fused.setdefault(r["chunk_id"], {**r, "score": 0.0, "found_by": []})
            entry["score"] += weights[name] / (k + rank)
            entry["found_by"].append(f"{name}#{rank}")
    return sorted(fused.values(), key=lambda r: r["score"], reverse=True)


def hybrid_search(question: str, strategy: str = "header", k: int = 20, dense_weight: float = 0.5) -> list[dict]:
    """Dense + sparse, fused with RRF. dense_weight lets us tune meaning vs exact words."""
    dense = dense_search(question, strategy, k)
    sparse = sparse_search(question, strategy, k)
    fused = rrf_fuse({"dense": dense, "sparse": sparse}, {"dense": dense_weight, "sparse": 1 - dense_weight})
    return fused[:k]

def rerank(question: str, candidates: list[dict], top_n: int = 5) -> list[dict]:
    """The careful librarian: read question + each chunk together, keep the best top_n."""
    if not candidates:
        return []
    result = vo.rerank(
        question,
        [c["content"] for c in candidates],
        model=RERANK_MODEL,
        top_k=top_n,
    )
    # r.index points back to the candidate; relevance_score is 0-1 (higher = more relevant)
    return [{**candidates[r.index], "score": r.relevance_score} for r in result.results]


def search(question: str, strategy: str = "header", top_n: int = 5) -> list[dict]:
    """The full retrieval pipeline: hybrid search (top 20) -> rerank (best 5)."""
    candidates = hybrid_search(question, strategy, k=20)
    return rerank(question, candidates, top_n)


def show(results: list[dict], n: int = 5) -> None:
    for rank, r in enumerate(results[:n], start=1):
        found = f"  (found by: {', '.join(r['found_by'])})" if "found_by" in r else ""
        print(f"{rank}. [{r['score']:.4f}] {r['title']} > {r['heading'] or '-'}{found}")
        print(f"     {r['text'][:150].replace(chr(10), ' ')}")


if __name__ == "__main__":
    args = sys.argv[1:]
    debug = "--debug" in args  # show every stage, not just the final result
    question = " ".join(a for a in args if a != "--debug") or "How much paternity leave do I get in Ireland?"
    print(f"Q: {question}\n")

    if debug:
        print("=== Dense search (meaning) ===")
        show(dense_search(question))
        print("\n=== Sparse search (exact words, BM25) ===")
        show(sparse_search(question))
        print("\n=== Hybrid (RRF fusion) ===")
        show(hybrid_search(question))
        print()

    print("=== Final results (hybrid + rerank) ===")
    show(search(question))