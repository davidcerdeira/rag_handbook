import json
import random
import re
from collections import defaultdict
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from src.citation_checker import parse_json

load_dotenv()

DRAFT_MODEL = "claude-sonnet-5-5"   # question quality matters more than a fraction of a cent
SEED = 42                           # same "random" picks every run = reproducible
INDEXED = Path("data/processed/indexed_header.jsonl")
OUT_FILE = Path("evals/draft_questions.json")

TYPE_PLAN = ["direct"] * 15 + ["exact_term"] * 4 + ["paraphrased"] * 5   # 24 single-source questions
N_MULTI_HOP = 3

TYPE_INSTRUCTIONS = {
    "direct": "a straightforward question an employee would ask, fully answered by this excerpt",
    "exact_term": "a question that uses a specific term, acronym, amount or name that appears in the excerpt "
                  "(e.g. a benefit name, a euro amount, an acronym)",
    "paraphrased": "a question about the excerpt's content that uses DIFFERENT everyday words than the excerpt "
                   "(do not reuse its key terms)",
}

SINGLE_PROMPT = """You write evaluation questions for an HR assistant that answers from GitLab's handbook.
Given ONE handbook excerpt, write ONE question of the requested type.

Rules:
- The excerpt alone must fully answer the question.
- If the excerpt is specific to a country or entity, name it in the question (otherwise the question is ambiguous).
- expected_answer: the correct answer in 1-2 sentences, using only the excerpt.
- evidence: a SHORT phrase (3-12 words) copied EXACTLY, character for character, from the excerpt,
  that contains the key fact.

Respond with ONLY this JSON:
{"question": "...", "expected_answer": "...", "evidence": "..."}"""

MULTI_PROMPT = """You write evaluation questions for an HR assistant that answers from GitLab's handbook.
Given TWO handbook excerpts on the same topic for different countries/entities, write ONE comparison question
that needs BOTH excerpts to answer.

Rules:
- Name both countries/entities in the question.
- expected_answer: the correct answer in 1-3 sentences, using only the excerpts.
- evidence: two SHORT phrases (3-12 words each), one copied EXACTLY from each excerpt, containing the key facts.

Respond with ONLY this JSON:
{"question": "...", "expected_answer": "...", "evidence": ["<from excerpt 1>", "<from excerpt 2>"]}"""

claude = anthropic.Anthropic()


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def label(chunk: dict) -> str:
    return chunk["title"] + (f" > {chunk['heading']}" if chunk.get("heading") else "")


def ask_claude(system: str, user: str) -> dict:
    response = claude.messages.create(
        model=DRAFT_MODEL, max_tokens=1000, system=system,
        messages=[{"role": "user", "content": user}],
    )
    text = "".join(block.text for block in response.content if block.type == "text")
    return parse_json(text)


def load_candidates() -> list[dict]:
    """English chunks long enough to hold a real fact, in a reproducible random order."""
    with INDEXED.open(encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]
    chunks = [c for c in chunks if c.get("language") == "en" and len(c["text"]) >= 300 and c.get("heading")]
    random.Random(SEED).shuffle(chunks)
    return chunks


def pick_spread(chunks: list[dict], n: int) -> list[dict]:
    """Coverage: at most one chunk per document first, then fill up if there aren't enough documents."""
    picked, seen_docs = [], set()
    for c in chunks:
        if c["doc_id"] not in seen_docs:
            picked.append(c)
            seen_docs.add(c["doc_id"])
        if len(picked) == n:
            return picked
    for c in chunks:
        if c not in picked:
            picked.append(c)
        if len(picked) == n:
            break
    return picked


def pick_pairs(chunks: list[dict], n: int) -> list[tuple[dict, dict]]:
    """Same topic (last heading part, e.g. 'Paternity Leave'), different documents."""
    by_topic = defaultdict(list)
    for c in chunks:
        by_topic[c["heading"].split(" > ")[-1].strip().lower()].append(c)
    pairs = []
    for group in by_topic.values():
        docs = {}
        for c in group:
            docs.setdefault(c["doc_id"], c)  # one chunk per document
        if len(docs) >= 2:
            pairs.append(tuple(list(docs.values())[:2]))
    random.Random(SEED).shuffle(pairs)
    return pairs[:n]


def make_record(qtype: str, draft: dict, sources: list[dict]) -> dict:
    evidence = draft["evidence"] if isinstance(draft["evidence"], list) else [draft["evidence"]]
    # each evidence quote must really appear in its source chunk: if not, flag it for review
    evidence_ok = len(evidence) == len(sources) and all(
        normalize(e) in normalize(s["text"]) for e, s in zip(evidence, sources)
    )
    return {
        "type": qtype,
        "question": draft["question"],
        "expected_answer": draft["expected_answer"],
        "evidence": evidence,
        "expected_sources": [label(s) for s in sources],
        "evidence_ok": evidence_ok,
        "source_texts": [s["text"] for s in sources],   # for your review; removed from the final golden set
    }


if __name__ == "__main__":
    candidates = load_candidates()
    records = []

    for qtype, chunk in zip(TYPE_PLAN, pick_spread(candidates, len(TYPE_PLAN))):
        user = f"Question type: {qtype}: {TYPE_INSTRUCTIONS[qtype]}\n\nExcerpt ({label(chunk)}):\n{chunk['text']}"
        try:
            records.append(make_record(qtype, ask_claude(SINGLE_PROMPT, user), [chunk]))
            print(f"  ✓ {qtype:<12} {label(chunk)[:70]}")
        except Exception as error:   # one bad draft shouldn't stop the whole run
            print(f"  ✗ {qtype:<12} skipped ({error})")

    for a, b in pick_pairs(candidates, N_MULTI_HOP):
        user = f"Excerpt 1 ({label(a)}):\n{a['text']}\n\nExcerpt 2 ({label(b)}):\n{b['text']}"
        try:
            records.append(make_record("multi_hop", ask_claude(MULTI_PROMPT, user), [a, b]))
            print(f"  ✓ multi_hop    {label(a)[:32]} + {label(b)[:32]}")
        except Exception as error:
            print(f"  ✗ multi_hop    skipped ({error})")

    for i, record in enumerate(records, start=1):
        record["id"] = f"q{i:02d}"

    OUT_FILE.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    bad = [r["id"] for r in records if not r["evidence_ok"]]
    print(f"\nDrafted {len(records)} questions → {OUT_FILE}")
    print(f"Evidence not found verbatim (check these first): {bad or 'none'}")