HIGH = 0.75    # starting thresholds: tuned in phase 4 (evals)
MEDIUM = 0.5


def compute_confidence(result: dict) -> dict:
    """Weakest link: confidence is limited by the weakest signal. Red flags force Low."""
    if result["gated"]:
        return {"label": "None", "score": 0.0, "weakest": "retrieval", "signals": {},
                "reasons": ["The handbook doesn't seem to cover this question"]}

    check = result["check"]
    reasons = []

    # Signal 1: did we find a strong source? (best reranker score, 0-1)
    signals = {"retrieval": result["best_score"]}

    # Signal 2: is the answer faithful to its sources? (share of claims the judge marked supported)
    if check["skipped"]:
        signals["citations"] = 0.0
        reasons.append("No cited facts in the answer, so nothing could be verified")
    else:
        signals["citations"] = check["supported"] / check["total"] if check["total"] else 0.0
        not_supported = check["total"] - check["supported"]
        if not_supported:
            reasons.append(f"{not_supported} of {check['total']} claims not fully supported by their sources")

    # Weakest link: the lowest signal decides the score
    weakest = min(signals, key=signals.get)
    score = signals[weakest]

    # Red flags override everything
    red_flags = []
    if result["truncated"]:
        red_flags.append("Answer was cut off (hit max_tokens)")
    if check["invalid"]:
        red_flags.append(f"Cites excerpts that don't exist: {check['invalid']}")

    if red_flags:
        label = "Low"
        reasons += red_flags
    elif score >= HIGH:
        label = "High"
    elif score >= MEDIUM:
        label = "Medium"
    else:
        label = "Low"

    reasons.insert(0, f"Weakest signal: {weakest} ({score:.2f})")
    return {"label": label, "score": round(score, 2), "weakest": weakest,
            "signals": signals, "reasons": reasons}