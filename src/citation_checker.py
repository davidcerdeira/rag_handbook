import json
import re

import anthropic
from dotenv import load_dotenv

load_dotenv()

JUDGE_MODEL = "claude-haiku-5-5"  # cheapest model: checking is easier than writing
claude = anthropic.Anthropic()

CITATION = re.compile(r"\[(\d+)\]")  # finds [1], [2], ... in the answer

JUDGE_PROMPT = """You are a strict fact-checker for an HR assistant.
You receive numbered handbook excerpts and an answer that cites them like [1] or [2][3].

For every factual claim in the answer that carries a citation, check whether the CITED excerpt(s) support it:
- "supported": the cited excerpt states this fact (paraphrasing is fine: judge meaning, not wording)
- "partial": the cited excerpt supports part of it, but the claim adds details the excerpt doesn't contain
- "unsupported": the cited excerpt doesn't state this fact

If a claim is unsupported by its cited excerpt but ANOTHER excerpt supports it, put that excerpt's number in "better_source".
Skip sentences that aren't factual claims (e.g. "I can't answer that from the excerpts").

Respond with ONLY this JSON, nothing else:
{"claims": [{"claim": "...", "cited": [1], "verdict": "supported", "better_source": null}]}"""


def find_invalid_citations(answer: str, n_chunks: int) -> list[int]:
    """Free check, no LLM: citations pointing to excerpts that don't exist (e.g. [7] when there are 5)."""
    cited = {int(n) for n in CITATION.findall(answer)}
    return sorted(n for n in cited if n < 1 or n > n_chunks)


def parse_json(text: str) -> dict:
    """Models sometimes wrap JSON in ```json fences or add a sentence; grab the {...} part."""
    start, end = text.find("{"), text.rfind("}")
    return json.loads(text[start:end + 1])


def check_citations(answer: str, context: str, n_chunks: int) -> dict:
    """LLM-as-judge: does each cited excerpt actually support its claim?"""
    invalid = find_invalid_citations(answer, n_chunks)

    if not CITATION.search(answer):  # nothing cited (e.g. "the handbook doesn't cover this")
        return {"skipped": True, "claims": [], "invalid": invalid, "supported": 0, "total": 0,
                "input_tokens": 0, "output_tokens": 0}

    response = claude.messages.create(
        model=JUDGE_MODEL,
        max_tokens=2000,
        system=JUDGE_PROMPT,
        messages=[{"role": "user", "content": f"Excerpts:\n\n{context}\n\nAnswer to check:\n{answer}"}],
    )
    text = "".join(block.text for block in response.content if block.type == "text")
    claims = parse_json(text)["claims"]

    return {
        "skipped": False,
        "claims": claims,
        "invalid": invalid,
        "supported": sum(c["verdict"] == "supported" for c in claims),
        "total": len(claims),
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }


def summarize(check: dict) -> str:
    """Human-readable report of a citation check."""
    if check["skipped"]:
        return "Citation check: nothing to check (no citations)"
    lines = [f"Citation check: {check['supported']}/{check['total']} claims supported"]
    for c in check["claims"]:
        if c["verdict"] != "supported":
            better = f", better source [{c['better_source']}]" if c.get("better_source") else ""
            lines.append(f"  ⚠️ {c['verdict'].upper()}: \"{c['claim']}\" (cited {c['cited']}{better})")
    if check["invalid"]:
        lines.append(f"  ⚠️ Citations to excerpts that don't exist: {check['invalid']}")
    lines.append(f"  (judge tokens: {check['input_tokens']} in / {check['output_tokens']} out)")
    return "\n".join(lines)


if __name__ == "__main__":
    # TEST THE CHECKER ON A KNOWN-BAD ANSWER: we plant errors and confirm the judge catches them
    from src.generator import build_context
    from src.retriever import search

    chunks = search("How much paternity leave do I get in Ireland?")
    planted = (
        "You get 2 consecutive weeks of paternity leave within the first 6 months [1]. "   # correct
        "GitLab tops up the state benefit to 100% of your salary [1]. "                     # correct
        "You also get 9 weeks of Parent's Leave in the child's first 2 years [1]. "         # wrong number: it's in [2]
        "Paternity Benefit is paid by the Department of Social Protection [1]. "            # from memory, not in [1]
        "You must apply at least 12 weeks in advance [7]."                                  # excerpt [7] doesn't exist
    )
    print("Planted answer:\n" + planted + "\n")
    print(summarize(check_citations(planted, build_context(chunks), len(chunks))))