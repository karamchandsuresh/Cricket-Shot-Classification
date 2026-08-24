from pathlib import Path
import os
import uuid

from flask import (
    Flask,
    jsonify,
    request,
)
from flask_cors import CORS
from werkzeug.utils import (
    secure_filename,
)

from inference.analyzer import (
    CricketShotAnalyzer,
)


# =========================================================
# FLASK APPLICATION
# =========================================================

app = Flask(__name__)

CORS(app)


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

UPLOAD_FOLDER = (
    PROJECT_ROOT
    / "backend"
    / "uploads"
)

UPLOAD_FOLDER.mkdir(
    parents=True,
    exist_ok=True,
)


# =========================================================
# SUPPORTED VIDEO TYPES
# =========================================================

ALLOWED_EXTENSIONS = {
    ".avi",
    ".mp4",
    ".mov",
    ".mkv",
}


# =========================================================
# FINAL MODEL
#
# MobileNetV2 pretrained spatial features
# + GRU temporal classifier
#
# Test Accuracy: 69.60%
# =========================================================

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "precomputed_mobilenet_gru_best.keras"
)


# Load model once when backend starts.
analyzer = CricketShotAnalyzer(
    model_path=MODEL_PATH
)


# =========================================================
# FILE VALIDATION
# =========================================================

def allowed_file(
    filename,
):
    """
    Check whether the uploaded file
    has a supported video extension.
    """

    extension = (
        Path(filename)
        .suffix
        .lower()
    )

    return (
        extension
        in ALLOWED_EXTENSIONS
    )


# =========================================================
# HOME ENDPOINT
# =========================================================

@app.route(
    "/",
    methods=["GET"],
)
def home():

    return jsonify(
        {
            "project": (
                "Cricket Shot "
                "Classification "
                "and Analytics"
            ),

            "message": (
                "Backend is running "
                "successfully."
            ),

            "status": "ok",

            "model": (
                "MobileNetV2 "
                "Features + GRU"
            ),
        }
    )


# =========================================================
# HEALTH ENDPOINT
# =========================================================

@app.route(
    "/health",
    methods=["GET"],
)
def health():

    return jsonify(
        {
            "status": "healthy",

            "backend": "Flask",

            "model": (
                "precomputed_"
                "mobilenet_"
                "gru_best.keras"
            ),

            "test_accuracy": (
                69.60
            ),
        }
    )


# =========================================================
# VIDEO ANALYSIS ENDPOINT
# =========================================================

@app.route(
    "/analyze",
    methods=["POST"],
)
def analyze():
    """
    Receive a cricket video,
    classify the shot,
    run pose analysis,
    and return the result.
    """

    # -----------------------------------------------------
    # Validate upload field
    # -----------------------------------------------------

    if (
        "video"
        not in request.files
    ):

        return jsonify(
            {
                "success": False,

                "error": (
                    "No video file "
                    "provided."
                ),
            }
        ), 400


    uploaded_file = (
        request.files[
            "video"
        ]
    )


    # -----------------------------------------------------
    # Validate filename
    # -----------------------------------------------------

    if (
        uploaded_file.filename
        == ""
    ):

        return jsonify(
            {
                "success": False,

                "error": (
                    "No file selected."
                ),
            }
        ), 400


    # -----------------------------------------------------
    # Validate file extension
    # -----------------------------------------------------

    if not allowed_file(
        uploaded_file.filename
    ):

        return jsonify(
            {
                "success": False,

                "error": (
                    "Unsupported file "
                    "type. Use AVI, "
                    "MP4, MOV or MKV."
                ),
            }
        ), 400


    # -----------------------------------------------------
    # Secure filename
    # -----------------------------------------------------

    original_filename = (
        secure_filename(
            uploaded_file.filename
        )
    )

    extension = (
        Path(
            original_filename
        )
        .suffix
        .lower()
    )


    # -----------------------------------------------------
    # Generate temporary filename
    # -----------------------------------------------------

    unique_filename = (
        f"{uuid.uuid4().hex}"
        f"{extension}"
    )

    upload_path = (
        UPLOAD_FOLDER
        / unique_filename
    )


    # -----------------------------------------------------
    # Analyze video
    # -----------------------------------------------------

    try:

        uploaded_file.save(
            upload_path
        )


        result = (
            analyzer.analyze_video(
                upload_path
            )
        )


        response = {
            "success": True,

            "filename": (
                original_filename
            ),

            "prediction": (
                result[
                    "prediction"
                ]
            ),

            "confidence": (
                result[
                    "confidence"
                ]
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
                result[
                    "pose"
                ]
            ),

            "probabilities": (
                result[
                    "probabilities"
                ]
            ),
        }


        return jsonify(
            response
        ), 200


    # -----------------------------------------------------
    # Error handling
    # -----------------------------------------------------

    except Exception as error:

        return jsonify(
            {
                "success": False,

                "error": str(
                    error
                ),
            }
        ), 500


    # -----------------------------------------------------
    # Remove temporary upload
    # -----------------------------------------------------

    finally:

        if upload_path.exists():

            try:

                os.remove(
                    upload_path
                )

            except OSError:

                pass


# =========================================================
# START DEVELOPMENT SERVER
# =========================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
        use_reloader=False,
    )