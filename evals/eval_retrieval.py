"""Phase 4b: retrieval eval.

1. hit@5 for 2 chunking strategies x 4 search methods.
   A question is a hit if every evidence quote appears in one of the top-5 chunks.
2. Gate tuning: the best reranker score of every question (in-scope vs out-of-scope),
   and how many each threshold would block.
"""
import json
import re
import time
from pathlib import Path

from voyageai.error import RateLimitError

from src.retriever import dense_search, hybrid_search, search, sparse_search

GOLDEN = Path("evals/golden.json")
OUT = Path("evals/results_retrieval.json")
STRATEGIES = ["fixed", "header"]
METHODS = ["dense", "bm25", "hybrid", "hybrid+rerank"]
TOP_N = 5
THRESHOLDS = [round(0.20 + 0.05 * i, 2) for i in range(9)]  # 0.20 → 0.60


def norm(text: str) -> str:
    """Lowercase and collapse whitespace, so line breaks don't break a match."""
    return re.sub(r"\s+", " ", text).strip().lower()


def is_hit(evidence: list[str], chunks: list[dict]) -> bool:
    """Every quote must be inside a single chunk (no matching across chunk boundaries)."""
    texts = [norm(c["content"]) for c in chunks]
    return all(any(norm(e) in t for t in texts) for e in evidence)


def with_retry(fn, *args, **kwargs):
    """Call a Voyage-backed function; wait and retry if rate-limited."""
    for wait in (20, 40, 60, None):
        try:
            return fn(*args, **kwargs)
        except RateLimitError:
            if wait is None:
                raise
            print(f"   rate-limited, waiting {wait}s…")
            time.sleep(wait)


def run_setups(question: str, strategy: str) -> dict:
    """Top 5 chunks from each search method."""
    return {
        "dense": with_retry(dense_search, question, strategy=strategy, k=TOP_N),
        "bm25": sparse_search(question, strategy=strategy, k=TOP_N),
        "hybrid": with_retry(hybrid_search, question, strategy=strategy)[:TOP_N],
        "hybrid+rerank": with_retry(search, question, strategy=strategy, top_n=TOP_N),
    }


def print_hit_table(hits: dict) -> None:
    print(f"\n=== hit@{TOP_N} (questions with evidence: {len(hits['header']['dense'])}) ===")
    print(f"{'':10}" + "".join(f"{m:>16}" for m in METHODS))
    for s in STRATEGIES:
        row = f"{s:10}"
        for m in METHODS:
            n_hit = sum(r["hit"] for r in hits[s][m])
            row += f"{f'{n_hit / len(hits[s][m]):.0%} ({n_hit}/{len(hits[s][m])})':>16}"
        print(row)


def print_breakdown(rows: list[dict], label: str) -> None:
    print(f"\n=== {label}: by question type ===")
    for t in sorted({r["type"] for r in rows}):
        group = [r for r in rows if r["type"] == t]
        print(f"  {t:12} {sum(r['hit'] for r in group)}/{len(group)}")
    misses = [r["id"] for r in rows if not r["hit"]]
    print(f"  misses: {misses or 'none'}")


def tune_gate(questions: list[dict], best: dict) -> list[dict]:
    in_scope = sorted(best[q["id"]] for q in questions if q["type"] != "out_of_scope")
    out_scope = sorted(best[q["id"]] for q in questions if q["type"] == "out_of_scope")
    print("\n=== Gate: best reranker score (header, hybrid+rerank) ===")
    print(f"  out-of-scope:          {[round(s, 2) for s in out_scope]}")
    print(f"  in-scope, lowest 5:    {[round(s, 2) for s in in_scope[:5]]}")
    print(f"\n  {'threshold':>9} | {'off-topic blocked':>17} | {'false refusals':>14}")
    rows = []
    for t in THRESHOLDS:
        blocked = sum(s < t for s in out_scope)
        refused = sum(s < t for s in in_scope)
        rows.append({"threshold": t, "off_topic_blocked": blocked, "false_refusals": refused})
        print(f"  {t:>9.2f} | {f'{blocked}/{len(out_scope)}':>17} | {f'{refused}/{len(in_scope)}':>14}")
    return rows


def main() -> None:
    questions = json.loads(GOLDEN.read_text())
    hits = {s: {m: [] for m in METHODS} for s in STRATEGIES}
    best = {}  # question id -> best reranker score (header strategy = production setup)

    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q['id']} ({q['type']})")
        if q["evidence"]:
            for s in STRATEGIES:
                results = run_setups(q["question"], s)
                for m in METHODS:
                    hits[s][m].append({"id": q["id"], "type": q["type"],
                                       "hit": is_hit(q["evidence"], results[m])})
                if s == "header":
                    best[q["id"]] = results["hybrid+rerank"][0]["score"]
        else:  # ambiguous / out-of-scope: no evidence, only needed for the gate
            top = with_retry(search, q["question"], strategy="header", top_n=TOP_N)
            best[q["id"]] = top[0]["score"]

    print_hit_table(hits)
    print_breakdown(hits["header"]["hybrid+rerank"], "header + hybrid+rerank")
    gate = tune_gate(questions, best)

    OUT.write_text(json.dumps({"hits": hits, "best_scores": best, "gate": gate}, indent=2))
    print(f"\nSaved → {OUT}")


if __name__ == "__main__":
    main()