def get_attribution(combined_ai_score):
    if combined_ai_score >= 0.70:
        return "likely_ai"

    if combined_ai_score <= 0.30:
        return "likely_human"

    return "uncertain"

def get_confidence(combined_ai_score, attribution):
    if attribution == "likely_ai":
        return round(combined_ai_score, 2)

    if attribution == "likely_human":
        return round(1 - combined_ai_score, 2)

    return round(0.40 + abs(combined_ai_score - 0.50), 2)

def get_transparency_label(attribution, verified=False):
    if verified:
        return "Verified creator: this creator completed a provenance check for this content."

    if attribution == "likely_ai":
        return (
            "This content appears likely to be AI-generated. "
            "This label is based on writing-pattern signals and may be appealed by the creator."
        )

    if attribution == "likely_human":
        return (
            "This content appears likely to be written by a person. "
            "No AI-generated label is being shown."
        )

    return (
        "We are not confident enough to say whether this was written by a person "
        "or generated with AI."
    )
