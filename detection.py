from labels import get_attribution, get_confidence, get_transparency_label
from signals import groq_signal, stylometric_signal, formulaic_signal

def classify_text(text):
    groq_result = groq_signal(text)
    stylometric_result = stylometric_signal(text)
    formulaic_result = formulaic_signal(text)

    groq_score = groq_result["score"]
    stylometric_score = stylometric_result["score"]
    formulaic_score = formulaic_result["score"]

    combined_ai_score = round(
        0.50 * groq_score
        + 0.30 * stylometric_score
        + 0.20 * formulaic_score,
        2
    )

    attribution = get_attribution(combined_ai_score)
    confidence = get_confidence(combined_ai_score, attribution)
    label = get_transparency_label(attribution)

    return {
        "attribution": attribution,
        "confidence": confidence,
        "combined_ai_score": combined_ai_score,
        "label": label,
        "signals": {
            "groq": groq_result,
            "stylometric": stylometric_result,
            "formulaic": formulaic_result,
        }
    }

def classify_metadata(description, metadata):
    combined_text = description + " " + " ".join(
        f"{key}: {value}" for key, value in metadata.items()
    )

    result = classify_text(combined_text)
    result["content_type"] = "metadata"

    return result
