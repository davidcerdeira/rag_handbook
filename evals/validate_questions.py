import json
import random
from collections import Counter
from pathlib import Path

import anthropic
from dotenv import load_dotenv

from src.citation_checker import parse_json

load_dotenv()

REVIEW_MODEL = "claude-haiku-5-5"   # a different model from the drafter, so it doesn't just agree with itself
DRAFT_FILE = Path("evals/draft_questions.json")
GOLDEN_FILE = Path("evals/golden.json")
REJECTED_FILE = Path("evals/rejected_questions.json")

REVIEW_PROMPT = """You are a strict reviewer of evaluation questions for an HR assistant built on GitLab's handbook.
You get a test question, its expected answer, and the handbook excerpt(s) it was written from.

Check each of these:
- answerable: the excerpt(s) fully answer the question
- answer_correct: the expected answer is correct and complete according to the excerpt(s), with nothing added
- specific: the question names the country/entity when the answer depends on it
- realistic: an employee could plausibly ask this
- needs_all_sources: if there are 2 excerpts, answering really requires BOTH; if there is 1 excerpt, true

Respond with ONLY this JSON:
{"answerable": true, "answer_correct": true, "specific": true, "realistic": true,
 "needs_all_sources": true, "reason": "<one short sentence if anything is false, otherwise empty>"}"""

CHECKS = ["answerable", "answer_correct", "specific", "realistic", "needs_all_sources"]

# Hand-written: their "correct answer" is a behaviour, so there's nothing to check against the documents
MANUAL_QUESTIONS = [
    {"type": "ambiguous", "question": "How much parental leave do I get?",
     "expected_answer": "Should explain it depends on the country/entity, mention the company-wide 16 weeks of paid parental leave, and/or ask where the person is based."},
    {"type": "ambiguous", "question": "How many sick days do I get?",
     "expected_answer": "Should explain it depends on the country/entity and not give a single number without saying which entity it applies to."},
    {"type": "ambiguous", "question": "Do I get a pension?",
     "expected_answer": "Should explain pension benefits depend on the country/entity, and describe the options per entity or ask where the person is based."},
    {"type": "out_of_scope", "question": "What's the best pizza place in Lisbon?",
     "expected_answer": "Should say the handbook doesn't cover this."},
    {"type": "out_of_scope", "question": "What will the weather be like in Seoul tomorrow?",
     "expected_answer": "Should say the handbook doesn't cover this."},
    {"type": "out_of_scope", "question": "How do I reset my laptop password?",
     "expected_answer": "Should say the handbook doesn't cover this."},
    {"type": "out_of_scope", "question": "How do I set up two-factor authentication on my GitLab account?",
     "expected_answer": "Should say the handbook doesn't cover this."},
    {"type": "out_of_scope", "question": "How do I get access to the company VPN?",
     "expected_answer": "Should say the handbook doesn't cover this."},
]

claude = anthropic.Anthropic()


def review(q: dict) -> dict:
    excerpts = "\n\n".join(
        f"Excerpt {i} ({label}):\n{text}"
        for i, (label, text) in enumerate(zip(q["expected_sources"], q["source_texts"]), start=1)
    )
    response = claude.messages.create(
        model=REVIEW_MODEL, max_tokens=500, system=REVIEW_PROMPT,
        messages=[{"role": "user", "content":
                   f"Question: {q['question']}\nExpected answer: {q['expected_answer']}\n\n{excerpts}"}],
    )
    return parse_json("".join(b.text for b in response.content if b.type == "text"))


def clean(q: dict) -> dict:
    """The final golden set doesn't need the review-only fields."""
    return {k: v for k, v in q.items() if k not in ("source_texts", "evidence_ok", "id")}


if __name__ == "__main__":
    drafts = json.loads(DRAFT_FILE.read_text(encoding="utf-8"))
    kept, rejected = [], []

    for q in drafts:
        try:
            verdict = review(q)
        except Exception as error:
            verdict = {"reason": f"review failed: {error}"}
        failed = [check for check in CHECKS if not verdict.get(check, False)]
        if not q["evidence_ok"]:
            failed.append("evidence_not_verbatim")

        if failed:
            rejected.append({**clean(q), "failed": failed, "reason": verdict.get("reason", "")})
            print(f"  ✗ {q['id']} {q['type']:<12} {', '.join(failed)}: {verdict.get('reason', '')}")
        else:
            kept.append(clean(q))
            print(f"  ✓ {q['id']} {q['type']:<12} {q['question'][:70]}")

    manual = [{**m, "evidence": [], "expected_sources": []} for m in MANUAL_QUESTIONS]
    golden = kept + manual
    for i, q in enumerate(golden, start=1):
        q["id"] = f"q{i:02d}"

    GOLDEN_FILE.write_text(json.dumps(golden, indent=2, ensure_ascii=False), encoding="utf-8")
    REJECTED_FILE.write_text(json.dumps(rejected, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nKept {len(kept)}/{len(drafts)} drafted + {len(manual)} hand-written = {len(golden)} questions → {GOLDEN_FILE}")
    print(f"By type: {dict(Counter(q['type'] for q in golden))}")

    print("\n--- Spot check: 3 random kept questions ---")
    for q in random.sample(kept, k=min(3, len(kept))):
        print(f"\n[{q['type']}] {q['question']}\n  expected: {q['expected_answer']}\n  evidence: {q['evidence']}")