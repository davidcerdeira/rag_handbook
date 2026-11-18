"""Phase 4c: answer eval.

Runs the full pipeline (search → gate → answer → citation check → confidence) on every
golden question, then a Haiku judge grades each answer against expected_answer.
Reports correctness, faithfulness, abstention, cost/speed and confidence calibration.
"""
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

from evals.eval_retrieval import with_retry
from src.citation_checker import parse_json
from src.generator import answer_question

load_dotenv()
client = Anthropic()

GOLDEN = Path("evals/golden.json")
OUT = Path("evals/results_answers.json")
JUDGE_MODEL = "claude-haiku-5-5"
FACTUAL = {"direct", "exact_term", "paraphrased", "multi_hop"}
TYPE_ORDER = ["direct", "exact_term", "paraphrased", "multi_hop", "ambiguous", "out_of_scope"]
VERDICTS = {"correct", "partial", "incorrect"}

JUDGE_PROMPT = """You are grading an HR assistant's answer.

Question: {question}

Expected answer (for factual questions: the key facts; otherwise: the expected behaviour):
{expected}

Actual answer:
{answer}

Grade by MEANING, not wording:
- "correct": contains the key facts (or shows the expected behaviour) and contradicts nothing. Extra correct detail is fine.
- "partial": some key facts missing or slightly off, but the main point is right.
- "incorrect": wrong, contradicts the expected answer, misses the main point, or refuses to answer a question it should answer.

Return ONLY JSON: {{"verdict": "correct" | "partial" | "incorrect", "reason": "<one sentence>"}}"""


def judge(question: str, expected: str, answer: str) -> dict:
    """Haiku grades one answer. Retries once if the JSON comes back broken (lesson from bug #7)."""
    prompt = JUDGE_PROMPT.format(question=question, expected=expected, answer=answer)
    text = ""
    for _ in range(2):
        response = client.messages.create(
            model=JUDGE_MODEL, max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in response.content if b.type == "text")
        try:
            result = parse_json(text)
            if result.get("verdict") in VERDICTS:
                return result
        except ValueError:
            pass
    return {"verdict": "error", "reason": f"Judge output unreadable: {text[:100]}"}


def grade(q: dict, result: dict) -> dict:
    """Gated questions are graded by rule (no judge call); everything else by the judge."""
    if result["gated"]:
        if q["type"] == "out_of_scope":
            return {"verdict": "correct", "reason": "Blocked by the gate (no LLM call)."}
        return {"verdict": "incorrect", "reason": "False refusal: blocked by the gate."}
    return judge(q["question"], q["expected_answer"], result["answer"])


def run_one(q: dict) -> dict:
    start = time.perf_counter()
    result = with_retry(answer_question, q["question"])
    seconds = time.perf_counter() - start
    verdict = grade(q, result)
    check = result.get("check") or {}
    checked = bool(check) and not check.get("skipped")
    return {
        "id": q["id"], "type": q["type"], "question": q["question"],
        "expected_answer": q["expected_answer"], "answer": result["answer"],
        "verdict": verdict["verdict"], "reason": verdict.get("reason", ""),
        "gated": result["gated"], "best_score": round(result["best_score"], 3),
        "confidence": result["confidence"]["label"],
        "claims_supported": check.get("supported") if checked else None,
        "claims_total": check.get("total") if checked else None,
        "truncated": result["truncated"],
        "input_tokens": result["input_tokens"], "output_tokens": result["output_tokens"],
        "seconds": round(seconds, 1),
        "sources": [f"{c.get('title')} > {c.get('heading')}" for c in result["chunks"]],
    }


def pct(a: int, b: int) -> str:
    return f"{a / b:.0%} ({a}/{b})" if b else "n/a"


def report(rows: list[dict]) -> None:
    print("\n=== Verdicts by question type ===")
    print(f"{'type':14}{'correct':>9}{'partial':>9}{'incorrect':>11}{'error':>7}")
    for t in TYPE_ORDER:
        c = Counter(r["verdict"] for r in rows if r["type"] == t)
        if c:
            print(f"{t:14}{c['correct']:>9}{c['partial']:>9}{c['incorrect']:>11}{c['error']:>7}")

    factual = [r for r in rows if r["type"] in FACTUAL]
    n_correct = sum(r["verdict"] == "correct" for r in factual)
    n_ok = sum(r["verdict"] in ("correct", "partial") for r in factual)
    print("\n=== Headline numbers ===")
    print(f"Correctness (factual):     {pct(n_correct, len(factual))} correct, "
          f"{pct(n_ok, len(factual))} correct or partial")
    for t in ("ambiguous", "out_of_scope"):
        group = [r for r in rows if r["type"] == t]
        print(f"Abstention, {t:13} {pct(sum(r['verdict'] == 'correct' for r in group), len(group))}")

    checked = [r for r in rows if r.get("claims_total")]
    supported = sum(r["claims_supported"] for r in checked)
    total = sum(r["claims_total"] for r in checked)
    fully = sum(r["claims_supported"] == r["claims_total"] for r in checked)
    print(f"Faithfulness:              {pct(supported, total)} of claims supported, "
          f"{pct(fully, len(checked))} of answers fully supported")

    answered = [r for r in rows if r.get("gated") is False]
    if answered:
        avg = lambda k: sum(r[k] for r in answered) / len(answered)
        print(f"Per answered question:     ~{avg('input_tokens'):.0f} in / {avg('output_tokens'):.0f} out "
              f"tokens (answer call), {avg('seconds'):.1f}s avg, "
              f"{max(r['seconds'] for r in answered):.1f}s max (full pipeline)")
        print(f"Truncated answers:         {sum(r['truncated'] for r in answered)}")

    print("\n=== Confidence calibration (factual questions) ===")
    by_label = defaultdict(list)
    for r in factual:
        if "confidence" in r:
            by_label[r["confidence"]].append(r["verdict"] == "correct")
    for label in ("High", "Medium", "Low", "None"):
        if by_label[label]:
            print(f"  {label:7} {pct(sum(by_label[label]), len(by_label[label]))} correct")

    print("\n=== Not fully correct (read only these) ===")
    for r in rows:
        if r["verdict"] != "correct":
            print(f"\n{r['id']} [{r['type']}] {r['verdict'].upper()}: {r['question']}")
            print(f"  judge:  {r['reason']}")
            if r.get("answer"):
                print(f"  answer: {r['answer'][:300]}…")


def main() -> None:
    questions = json.loads(GOLDEN.read_text())
    rows = []
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q['id']} ({q['type']}) … ", end="", flush=True)
        try:
            row = run_one(q)
        except Exception as e:  # one broken question shouldn't stop the whole eval
            row = {"id": q["id"], "type": q["type"], "question": q["question"],
                   "verdict": "error", "reason": f"{type(e).__name__}: {e}"}
        rows.append(row)
        print(row["verdict"])

    report(rows)
    OUT.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"\nSaved → {OUT}")


if __name__ == "__main__":
    main()