import uuid
from datetime import datetime, timezone

from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from database import (
    init_db,
    save_content,
    get_content,
    update_content_status,
    verify_content,
    save_appeal,
    log_event,
    read_log,
    get_analytics,
    get_latest_classification_log,
)
from detection import classify_text, classify_metadata
from labels import get_transparency_label

app = Flask(__name__)

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[],
    storage_uri="memory://",
)

init_db()

def now_utc():
    return datetime.now(timezone.utc).isoformat()

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Provenance Guard is running.",
        "endpoints": [
            "POST /submit",
            "POST /appeal",
            "GET /log",
            "GET /content/<content_id>",
            "GET /analytics",
            "POST /verify/<content_id>",
            "POST /submit-metadata"
        ]
    })

@app.route("/submit", methods=["POST"])
@limiter.limit("10 per minute;100 per day")
def submit():
    data = request.get_json() or {}

    text = data.get("text", "").strip()
    creator_id = data.get("creator_id", "").strip()

    if not text:
        return jsonify({"error": "Missing required field: text"}), 400

    if not creator_id:
        return jsonify({"error": "Missing required field: creator_id"}), 400

    content_id = str(uuid.uuid4())
    result = classify_text(text)

    content = {
        "content_id": content_id,
        "creator_id": creator_id,
        "text": text,
        "attribution": result["attribution"],
        "confidence": result["confidence"],
        "combined_ai_score": result["combined_ai_score"],
        "label": result["label"],
        "status": "classified",
        "verified": 0,
        "created_at": now_utc(),
    }

    save_content(content)

    log_event({
        "event_type": "classification",
        "content_id": content_id,
        "creator_id": creator_id,
        "attribution": result["attribution"],
        "confidence": result["confidence"],
        "combined_ai_score": result["combined_ai_score"],
        "groq_score": result["signals"]["groq"]["score"],
        "stylometric_score": result["signals"]["stylometric"]["score"],
        "formulaic_score": result["signals"]["formulaic"]["score"],
        "status": "classified",
    })

    return jsonify({
        "content_id": content_id,
        "creator_id": creator_id,
        "attribution": result["attribution"],
        "confidence": result["confidence"],
        "combined_ai_score": result["combined_ai_score"],
        "label": result["label"],
        "signals": {
            "groq_score": result["signals"]["groq"]["score"],
            "groq_reasoning": result["signals"]["groq"]["reasoning"],
            "stylometric_score": result["signals"]["stylometric"]["score"],
            "stylometric_metrics": result["signals"]["stylometric"]["metrics"],
            "formulaic_score": result["signals"]["formulaic"]["score"],
            "matched_formulaic_phrases": result["signals"]["formulaic"]["matched_phrases"],
        },
        "status": "classified"
    })

@app.route("/appeal", methods=["POST"])
def appeal():
    data = request.get_json() or {}

    content_id = data.get("content_id", "").strip()
    creator_reasoning = data.get("creator_reasoning", "").strip()

    if not content_id:
        return jsonify({"error": "Missing required field: content_id"}), 400

    if not creator_reasoning:
        return jsonify({"error": "Missing required field: creator_reasoning"}), 400

    content = get_content(content_id)

    if not content:
        return jsonify({"error": "No content found with that content_id"}), 404

    classification_scores = get_latest_classification_log(content_id)

    save_appeal(content_id, creator_reasoning)
    update_content_status(content_id, "under_review")

    log_event({
        "event_type": "appeal",
        "content_id": content_id,
        "creator_id": content["creator_id"],
        "attribution": content["attribution"],
        "confidence": content["confidence"],
        "combined_ai_score": content["combined_ai_score"],
        "groq_score": classification_scores.get("groq_score"),
        "stylometric_score": classification_scores.get("stylometric_score"),
        "formulaic_score": classification_scores.get("formulaic_score"),
        "status": "under_review",
        "appeal_reasoning": creator_reasoning,
    })

    return jsonify({
        "content_id": content_id,
        "status": "under_review",
        "message": "Your appeal was received and is under review."
    })

@app.route("/log", methods=["GET"])
def view_log():
    limit = request.args.get("limit", default=20, type=int)
    return jsonify({"entries": read_log(limit=limit)})

@app.route("/content/<content_id>", methods=["GET"])
def view_content(content_id):
    content = get_content(content_id)

    if not content:
        return jsonify({"error": "No content found with that content_id"}), 404

    return jsonify(content)

@app.route("/analytics", methods=["GET"])
def analytics():
    return jsonify(get_analytics())

@app.route("/verify/<content_id>", methods=["POST"])
def verify(content_id):
    content = get_content(content_id)

    if not content:
        return jsonify({"error": "No content found with that content_id"}), 404

    verify_content(content_id)

    verified_label = get_transparency_label(content["attribution"], verified=True)

    log_event({
        "event_type": "verification",
        "content_id": content_id,
        "creator_id": content["creator_id"],
        "attribution": content["attribution"],
        "confidence": content["confidence"],
        "combined_ai_score": content["combined_ai_score"],
        "status": "verified",
    })

    return jsonify({
        "content_id": content_id,
        "verified": True,
        "label": verified_label,
        "message": "Creator verification completed for this content."
    })

@app.route("/submit-metadata", methods=["POST"])
@limiter.limit("10 per minute;100 per day")
def submit_metadata():
    data = request.get_json() or {}

    creator_id = data.get("creator_id", "").strip()
    content_type = data.get("content_type", "metadata").strip()
    description = data.get("description", "").strip()
    metadata = data.get("metadata", {})

    if not creator_id:
        return jsonify({"error": "Missing required field: creator_id"}), 400

    if not description and not metadata:
        return jsonify({
            "error": "Provide at least a description or metadata."
        }), 400

    content_id = str(uuid.uuid4())
    result = classify_metadata(description, metadata)

    stored_text = f"content_type: {content_type}\ndescription: {description}\nmetadata: {metadata}"

    content = {
        "content_id": content_id,
        "creator_id": creator_id,
        "text": stored_text,
        "attribution": result["attribution"],
        "confidence": result["confidence"],
        "combined_ai_score": result["combined_ai_score"],
        "label": result["label"],
        "status": "classified",
        "verified": 0,
        "created_at": now_utc(),
    }

    save_content(content)

    log_event({
        "event_type": "metadata_classification",
        "content_id": content_id,
        "creator_id": creator_id,
        "attribution": result["attribution"],
        "confidence": result["confidence"],
        "combined_ai_score": result["combined_ai_score"],
        "groq_score": result["signals"]["groq"]["score"],
        "stylometric_score": result["signals"]["stylometric"]["score"],
        "formulaic_score": result["signals"]["formulaic"]["score"],
        "status": "classified",
    })

    return jsonify({
        "content_id": content_id,
        "creator_id": creator_id,
        "content_type": content_type,
        "attribution": result["attribution"],
        "confidence": result["confidence"],
        "combined_ai_score": result["combined_ai_score"],
        "label": result["label"],
        "signals": {
            "groq_score": result["signals"]["groq"]["score"],
            "stylometric_score": result["signals"]["stylometric"]["score"],
            "formulaic_score": result["signals"]["formulaic"]["score"],
        },
        "status": "classified"
    })

if __name__ == "__main__":
    app.run(port=5000, debug=True)
