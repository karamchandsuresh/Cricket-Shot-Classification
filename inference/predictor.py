from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2

from preprocessing.video_preprocessor import preprocess_video


CLASS_NAMES = [
    "cover",
    "defense",
    "flick",
    "hook",
    "late_cut",
    "lofted",
    "pull",
    "square_cut",
    "straight",
    "sweep",
]


NUM_FRAMES = 16
FRAME_HEIGHT = 112
FRAME_WIDTH = 112
CHANNELS = 3
FEATURE_SIZE = 1280


class CricketShotPredictor:
    """
    Cricket shot classifier using:

    Video
    -> 16 sampled frames
    -> pretrained MobileNetV2 features
    -> GRU classifier
    -> shot prediction
    """

    def __init__(
        self,
        model_path=(
            "models/"
            "precomputed_mobilenet_gru_best.keras"
        ),
    ):
        self.model_path = Path(
            model_path
        )

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: "
                f"{self.model_path}"
            )

        print(
            "Loading MobileNetV2 "
            "+ GRU cricket shot model..."
        )

        self.model = (
            tf.keras.models.load_model(
                self.model_path
            )
        )

        print(
            "GRU classifier loaded successfully."
        )

        print(
            "Loading pretrained "
            "MobileNetV2 feature extractor..."
        )

        self.feature_extractor = MobileNetV2(
            weights="imagenet",
            include_top=False,
            pooling="avg",
            input_shape=(
                FRAME_HEIGHT,
                FRAME_WIDTH,
                CHANNELS,
            ),
        )

        self.feature_extractor.trainable = False

        print(
            "MobileNetV2 feature extractor "
            "loaded successfully."
        )

    def get_confidence_level(
        self,
        confidence,
    ):
        """
        Convert prediction probability into
        a user-facing confidence level.
        """

        if confidence >= 0.60:
            return "high"

        if confidence >= 0.30:
            return "medium"

        return "low"

    def extract_features(
        self,
        processed_video,
    ):
        """
        Convert 16 processed frames into
        MobileNetV2 feature vectors.

        Input:
            (16, 112, 112, 3)

        Output:
            (1, 16, 1280)
        """

        video = np.asarray(
            processed_video,
            dtype=np.float32,
        )

        if video.shape != (
            NUM_FRAMES,
            FRAME_HEIGHT,
            FRAME_WIDTH,
            CHANNELS,
        ):
            raise ValueError(
                "Unexpected video shape: "
                f"{video.shape}"
            )

        # Existing preprocessing gives
        # pixel values in the 0-1 range.
        #
        # Convert to MobileNetV2's
        # expected approximately -1 to 1.
        video = (
            video * 2.0
        ) - 1.0

        frame_features = (
            self.feature_extractor.predict(
                video,
                batch_size=16,
                verbose=0,
            )
        )

        video_features = (
            frame_features.reshape(
                1,
                NUM_FRAMES,
                FEATURE_SIZE,
            )
        )

        return video_features

    def predict(
        self,
        video_path,
    ):
        """
        Predict one cricket-shot video.
        """

        video_path = Path(
            video_path
        )

        if not video_path.exists():
            raise FileNotFoundError(
                f"Video not found: "
                f"{video_path}"
            )

        # -------------------------------------------------
        # Video preprocessing
        # -------------------------------------------------

        processed_video = preprocess_video(
            str(video_path)
        )

        # -------------------------------------------------
        # Pretrained MobileNetV2 features
        # -------------------------------------------------

        video_features = (
            self.extract_features(
                processed_video
            )
        )

        # -------------------------------------------------
        # GRU prediction
        # -------------------------------------------------

        probabilities = (
            self.model.predict(
                video_features,
                verbose=0,
            )[0]
        )

        predicted_index = int(
            np.argmax(
                probabilities
            )
        )

        confidence = float(
            probabilities[
                predicted_index
            ]
        )

        predicted_class = (
            CLASS_NAMES[
                predicted_index
            ]
        )

        confidence_level = (
            self.get_confidence_level(
                confidence
            )
        )

        probability_dict = {
            class_name: float(
                probability
            )
            for (
                class_name,
                probability,
            ) in zip(
                CLASS_NAMES,
                probabilities,
            )
        }

        return {
            "predicted_class": (
                predicted_class
            ),

            "confidence": (
                confidence
            ),

            "confidence_percent": (
                confidence * 100
            ),

            "confidence_level": (
                confidence_level
            ),

            "probabilities": (
                probability_dict
            ),
        }