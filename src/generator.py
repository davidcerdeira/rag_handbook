import sys

import anthropic
from dotenv import load_dotenv

from src.retriever import search
from src.citation_checker import check_citations, summarize
from src.confidence import compute_confidence

load_dotenv()

ANSWER_MODEL = "claude-sonnet-5-5"
claude = anthropic.Anthropic()

MIN_RELEVANCE = 0.5  # gate: below this best-chunk score, don't call Claude (tune in evals)
NOT_COVERED = (
    "I couldn't find anything in the handbook that answers this. "
    "Try rephrasing your question, or contact People Ops directly."
)

SYSTEM_PROMPT = """You are an HR assistant for GitLab team members.
Answer ONLY using the numbered handbook excerpts provided. Never use outside knowledge.

Rules:
- End every factual sentence with the excerpt number(s) it comes from, like [1] or [2][3].
- If the excerpts don't contain the answer, say so plainly. Do not guess.
- If the answer depends on something the question doesn't say (for example the country or entity),
  say what it depends on and summarise the options the excerpts cover.
- Be concise: a short direct answer first, then details if needed."""


def build_context(chunks: list[dict]) -> str:
    """AUGMENTATION: number each chunk so Claude can cite it: [1], [2], ..."""
    blocks = []
    for i, c in enumerate(chunks, start=1):
        label = c["title"] + (f" > {c['heading']}" if c.get("heading") else "")
        blocks.append(f"[{i}] {label}\n{c['text']}")
    return "\n\n".join(blocks)


def generate_answer(question: str, chunks: list[dict]) -> dict:
    """GENERATION: send rules + numbered chunks + question to Claude."""
    response = claude.messages.create(
        model=ANSWER_MODEL,
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": f"Handbook excerpts:\n\n{build_context(chunks)}\n\nQuestion: {question}",
        }],
    )
    return {
        "answer": "".join(block.text for block in response.content if block.type == "text"),
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "truncated": response.stop_reason == "max_tokens",
    }

def answer_question(question: str) -> dict:
    """The full pipeline: retrieve -> gate -> (augment + generate) -> verify -> confidence."""
    chunks = search(question)                                  # RETRIEVAL
    best_score = chunks[0]["score"] if chunks else 0.0

    if best_score < MIN_RELEVANCE:                             # GATE: skip Claude entirely
        result = {"answer": NOT_COVERED, "gated": True, "best_score": best_score,
                  "chunks": chunks, "input_tokens": 0, "output_tokens": 0,
                  "truncated": False, "check": None}
    else:
        result = generate_answer(question, chunks)             # AUGMENTATION + GENERATION
        check = check_citations(result["answer"], build_context(chunks), len(chunks))  # VERIFY
        result = {**result, "gated": False, "best_score": best_score, "chunks": chunks, "check": check}

    result["confidence"] = compute_confidence(result)          # CONFIDENCE (no API call)
    return result


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "How much paternity leave do I get in Ireland?"
    result = answer_question(question)

    print(f"Q: {question}\n")
    print(result["answer"])
    if result["truncated"]:
        print("\n⚠️ Answer was cut off (hit max_tokens).")

    if result["gated"]:
        print(f"\n🚫 Gate: best score {result['best_score']:.2f} < {MIN_RELEVANCE} → Claude not called")
        print("Closest chunks (not used):")
    else:
        print("\nSources:")
    for i, c in enumerate(result["chunks"], start=1):
        print(f"  [{i}] ({c['score']:.2f}) {c['title']} > {c.get('heading') or '-'}")
    print(f"\nTokens: {result['input_tokens']} in / {result['output_tokens']} out")
    if result["check"]:
        print("\n" + summarize(result["check"]))
    conf = result["confidence"]
    print(f"\nConfidence: {conf['label']} ({conf['score']:.2f})")
    for reason in conf["reasons"]:
        print(f"  - {reason}")
    