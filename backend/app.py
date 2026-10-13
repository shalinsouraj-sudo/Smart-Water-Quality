from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import pandas as pd
import cv2
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "MULTI_ML_PIPELINE.pkl")
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")

# -----------------------------
# Load ML pipeline
# -----------------------------
pipeline = joblib.load(MODEL_PATH)

models = pipeline["models"]
features = pipeline["features"]

# -----------------------------
# Safe limits from ML notebook
# -----------------------------
LIMITS = {
    "Nitrate": 50,
    "Boron": 2.4,
    "Fluoride": 1.5,
    "Chlorine": 5,
    "Copper": 2,
    "Chromium": 0.05,
    "Zinc": 3,
    "Sodium": 200,
    "Iron": 0.3,
    "Sulphate": 250,
    "Magnesium": 50,
    "Cadmium": 0.003,
    "Arsenic": 0.01,
    "Chloride": 250,
    "Lead": 0.01,
    "Mercury": 0.006
}

# -----------------------------
# Prediction function
# -----------------------------
def predict_from_rgb(R, G, B):

    R_model = float(R) / 255.0
    G_model = float(G) / 255.0
    B_model = float(B) / 255.0

    X_input = pd.DataFrame(
        [[R_model, G_model, B_model]],
        columns=features
    )

    results = []

    for contaminant, model in models.items():

        predicted_ppm = float(model.predict(X_input)[0])

        # Same logic as notebook
        predicted_ppm = max(0, predicted_ppm)

        limit = LIMITS[contaminant]

        status = "UNSAFE" if predicted_ppm > limit else "SAFE"

        results.append({
            "Contaminant": contaminant,
            "Predicted_PPM": round(predicted_ppm, 4),
            "Safe_Limit": limit,
            "Status": status
        })

    overall_status = (
        "UNSAFE"
        if any(item["Status"] == "UNSAFE" for item in results)
        else "SAFE"
    )

    return {
        "RGB": {
            "R": round(float(R), 2),
            "G": round(float(G), 2),
            "B": round(float(B), 2)
        },
        "results": results,
        "overall_status": overall_status
    }


# -----------------------------
# Health check
# -----------------------------
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": True,
        "model_count": len(models)
    })


# -----------------------------
# RAW RGB prediction
# Useful for testing now
# -----------------------------
@app.route("/api/predict", methods=["POST"])
def predict():

    data = request.get_json()

    if not data:
        return jsonify({"error": "JSON body required"}), 400

    required = ["R", "G", "B"]

    if not all(key in data for key in required):
        return jsonify({
            "error": "R, G and B are required"
        }), 400

    try:
        result = predict_from_rgb(
            data["R"],
            data["G"],
            data["B"]
        )

        return jsonify(result)

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


# -----------------------------
# IMAGE → ROI → RGB → ML
# -----------------------------
@app.route("/api/analyze", methods=["POST"])
def analyze():

    if "image" not in request.files:
        return jsonify({
            "error": "No image uploaded"
        }), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({
            "error": "Empty filename"
        }), 400

    try:
        # Read uploaded image
        image_bytes = np.frombuffer(
            file.read(),
            np.uint8
        )

        image = cv2.imdecode(
            image_bytes,
            cv2.IMREAD_COLOR
        )

        if image is None:
            return jsonify({
                "error": "Could not read image"
            }), 400

        # Same ROI coordinates used in notebook
        x = 245
        y = 70
        width = 655
        height = 510

        image_height, image_width = image.shape[:2]

        if (
            x + width > image_width
            or y + height > image_height
        ):
            return jsonify({
                "error": (
                    f"Image is too small for the configured ROI. "
                    f"Image size: {image_width}x{image_height}, "
                    f"required at least: {x + width}x{y + height}"
                )
            }), 400

        roi = image[
            y:y + height,
            x:x + width
        ]

        # Notebook calculates average BGR
        average_bgr = np.mean(
            roi,
            axis=(0, 1)
        )

        B = average_bgr[0]
        G = average_bgr[1]
        R = average_bgr[2]

        result = predict_from_rgb(R, G, B)

        result["processing"] = {
            "roi": {
                "x": x,
                "y": y,
                "width": width,
                "height": height
            },
            "method": "OpenCV average RGB from ROI"
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


# -----------------------------
# Serve frontend
# -----------------------------
@app.route("/")
def index():
    return send_from_directory(
        FRONTEND_DIR,
        "water-quality-dashboard.html"
    )


if __name__ == "__main__":
    print("======================================")
    print("Smart Water Quality Backend")
    print("======================================")
    print(f"Loaded models: {len(models)}")
    print(f"Features: {features}")
    print("Server: http://127.0.0.1:5000")
    print("======================================")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )