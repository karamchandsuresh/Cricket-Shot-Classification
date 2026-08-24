from pathlib import Path
import os
import uuid

from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.utils import secure_filename

from inference.analyzer import CricketShotAnalyzer


app = Flask(__name__)
CORS(app)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
UPLOAD_FOLDER = PROJECT_ROOT / "backend" / "uploads"

UPLOAD_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


ALLOWED_EXTENSIONS = {
    ".avi",
    ".mp4",
    ".mov",
    ".mkv",
}


# Load the model once when the backend starts.
analyzer = CricketShotAnalyzer(
    model_path=(
        PROJECT_ROOT
        / "models"
        / "cricket_shot_3dcnn_epoch15.keras"
    )
)


def allowed_file(filename):
    """
    Check whether the uploaded file has
    a supported video extension.
    """

    extension = Path(filename).suffix.lower()

    return extension in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET"])
def home():
    return jsonify(
        {
            "project": (
                "Cricket Shot Classification "
                "and Analytics"
            ),
            "message": (
                "Backend is running successfully."
            ),
            "status": "ok",
        }
    )


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "healthy",
            "backend": "Flask",
            "model": (
                "cricket_shot_3dcnn_epoch15.keras"
            ),
        }
    )


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Receive a cricket video, analyze it,
    and return prediction + pose information.
    """

    if "video" not in request.files:
        return jsonify(
            {
                "error": "No video file provided."
            }
        ), 400

    uploaded_file = request.files["video"]

    if uploaded_file.filename == "":
        return jsonify(
            {
                "error": "No file selected."
            }
        ), 400

    if not allowed_file(
        uploaded_file.filename
    ):
        return jsonify(
            {
                "error": (
                    "Unsupported file type. "
                    "Use AVI, MP4, MOV or MKV."
                )
            }
        ), 400

    original_filename = secure_filename(
        uploaded_file.filename
    )

    extension = Path(
        original_filename
    ).suffix.lower()

    unique_filename = (
        f"{uuid.uuid4().hex}{extension}"
    )

    upload_path = (
        UPLOAD_FOLDER
        / unique_filename
    )

    try:
        uploaded_file.save(
            upload_path
        )

        result = analyzer.analyze_video(
            upload_path
        )

        response = {
            "success": True,

            "filename": (
                original_filename
            ),

            "prediction": (
                result["prediction"]
            ),

            "confidence": (
                result["confidence"]
            ),

            "confidence_percent": (
                result[
                    "confidence_percent"
                ]
            ),

            "confidence_level": (
                result[
                    "confidence_level"
                ]
            ),

            "pose": (
                result["pose"]
            ),

            "probabilities": (
                result["probabilities"]
            ),
        }

        return jsonify(
            response
        ), 200

    except Exception as error:

        return jsonify(
            {
                "success": False,
                "error": str(error),
            }
        ), 500

    finally:

        if upload_path.exists():

            try:
                os.remove(
                    upload_path
                )

            except OSError:
                pass


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
        use_reloader=False,
    )