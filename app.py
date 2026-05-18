"""Flask demo for car damage severity prediction."""

from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from car_damage_cv import DamageSeverityPredictor

app = Flask(__name__)

_predictor: DamageSeverityPredictor | None = None


def get_predictor() -> DamageSeverityPredictor:
    global _predictor
    if _predictor is None:
        _predictor = DamageSeverityPredictor(
            damage_checkpoint=os.getenv("DAMAGE_MODEL_PATH", "model_damage.pth"),
            parts_checkpoint=os.getenv("PARTS_MODEL_PATH", "model_parts.pth"),
            device=os.getenv("DEVICE"),
            image_size=int(os.getenv("IMAGE_SIZE", "512")),
        )
    return _predictor


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    payload = request.get_json(silent=True) or {}
    image_url = payload.get("image_url", "").strip()
    if not image_url:
        return jsonify({"error": "image_url is required"}), 400

    try:
        result = get_predictor().predict_url(image_url)
    except FileNotFoundError as exc:
        return jsonify({"error": f"Model checkpoint not found: {Path(exc.filename).name}"}), 500
    except Exception as exc:
        return jsonify({"error": f"Could not analyze image: {exc}"}), 400

    return jsonify({"score": result.grade, "raw_score": result.raw_score, "url": image_url}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)
