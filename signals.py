import json
import os
import re
import statistics
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

MODEL = "llama-3.3-70b-versatile"

def clamp(value, minimum=0.0, maximum=1.0):
    return max(minimum, min(maximum, value))

def extract_json(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None

    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None

def groq_signal(text):
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return {
            "score": 0.5,
            "reasoning": "Groq API key missing, so this signal returned a neutral score."
        }

    client = Groq(api_key=api_key)

    prompt = f"""
You are one signal in a provenance detection system.

Your task is to estimate whether this text appears AI-generated or human-written.

Return ONLY valid JSON with this exact shape:
{{
  "score": 0.0,
  "reasoning": "brief explanation"
}}

The score must be from 0.0 to 1.0:
- 0.0 means very human-like
- 0.5 means uncertain
- 1.0 means very AI-like

Text:
{text}
"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You return only valid JSON. Do not include markdown."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1,
        )

        raw = response.choices[0].message.content
        parsed = extract_json(raw)

        if not parsed:
            return {
                "score": 0.5,
                "reasoning": "Groq response could not be parsed, so this signal returned a neutral score."
            }

        return {
            "score": clamp(float(parsed.get("score", 0.5))),
            "reasoning": parsed.get("reasoning", "No reasoning provided.")
        }

    except Exception as error:
        return {
            "score": 0.5,
            "reasoning": f"Groq signal failed, so this signal returned a neutral score. Error: {error}"
        }

def stylometric_signal(text):
    sentences = [
        sentence.strip()
        for sentence in re.split(r"[.!?]+", text)
        if sentence.strip()
    ]

    words = re.findall(r"\b[a-zA-Z']+\b", text.lower())
    total_words = len(words)

    if total_words == 0:
        return {
            "score": 0.5,
            "metrics": {},
            "reasoning": "No words found, so this signal returned a neutral score."
        }

    sentence_lengths = [
        len(re.findall(r"\b[a-zA-Z']+\b", sentence))
        for sentence in sentences
    ]

    avg_sentence_length = sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0
    sentence_variance = statistics.pvariance(sentence_lengths) if len(sentence_lengths) > 1 else 0
    type_token_ratio = len(set(words)) / total_words
    punctuation_density = len(re.findall(r"[,.!?;:]", text)) / max(total_words, 1)

    casual_markers = [
        "i ", "i'm", "honestly", "lol", "idk", "kinda", "sorta",
        "really", "way too", "won't", "can't", "don't", "my "
    ]
    casual_count = sum(text.lower().count(marker) for marker in casual_markers)

    # More uniform sentence lengths can look more AI-like.
    variance_score = 1 - clamp(sentence_variance / 80)

    # Very casual writing is less AI-like for this project.
    casual_score = 1 - clamp(casual_count / 5)

    # Moderate/high sentence length can look more polished and AI-like.
    length_score = clamp(avg_sentence_length / 25)

    # Extremely low word variety can be repetitive; extremely high can happen in short text.
    if type_token_ratio < 0.35:
        diversity_score = 0.75
    elif type_token_ratio > 0.85 and total_words < 80:
        diversity_score = 0.45
    else:
        diversity_score = 0.55

    score = (
        0.35 * variance_score
        + 0.25 * length_score
        + 0.20 * diversity_score
        + 0.20 * casual_score
    )

    # Short text is harder to judge with stylometrics, so avoid extreme scores.
    # This prevents two polished human sentences from being treated as strongly AI-like.
    if total_words < 50:
        score = min(score, 0.55)

    # First-person personal writing is usually less AI-like in this simple system.
    personal_markers = [" i ", " my ", " me ", " myself ", " coffee ", " friend ", " honestly"]
    personal_count = sum(f" {text.lower()} ".count(marker) for marker in personal_markers)

    if personal_count >= 2:
        score = max(0.0, score - 0.15)

    return {
        "score": round(clamp(score), 2),
        "metrics": {
            "avg_sentence_length": round(avg_sentence_length, 2),
            "sentence_length_variance": round(sentence_variance, 2),
            "type_token_ratio": round(type_token_ratio, 2),
            "punctuation_density": round(punctuation_density, 2),
            "casual_marker_count": casual_count,
        },
        "reasoning": "Stylometric score based on sentence consistency, sentence length, word variety, and casual language markers."
    }

def formulaic_signal(text):
    lower_text = text.lower()

    phrases = [
        "it is important to note",
        "it is essential to",
        "in conclusion",
        "furthermore",
        "moreover",
        "overall",
        "in today's world",
        "plays a crucial role",
        "transformative",
        "paradigm shift",
        "stakeholders",
        "responsible deployment",
        "various sectors",
        "ethical implications",
        "benefits are numerous",
    ]

    matched = [phrase for phrase in phrases if phrase in lower_text]
    words = re.findall(r"\b[a-zA-Z']+\b", lower_text)
    total_words = max(len(words), 1)

    phrase_density = len(matched) / total_words

    # A few formulaic phrases in a short paragraph should matter,
    # but this signal should not dominate the ensemble.
    score = clamp(len(matched) / 5)

    return {
        "score": round(score, 2),
        "matched_phrases": matched,
        "phrase_density": round(phrase_density, 4),
        "reasoning": "Formulaic score based on generic transition phrases and repeated AI-like wording."
    }
