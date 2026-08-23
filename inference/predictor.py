from pathlib import Path

import numpy as np
import tensorflow as tf

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


class CricketShotPredictor:
    """
    Load the trained 3D CNN and predict cricket shot classes
    from short cricket video clips.
    """

    def __init__(
        self,
        model_path="models/cricket_shot_3dcnn_epoch15.keras",
    ):
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}"
            )

        print("Loading cricket shot model...")

        self.model = tf.keras.models.load_model(
            self.model_path
        )

        print("Model loaded successfully.")

    def get_confidence_level(self, confidence):
        """
        Convert raw model confidence into a simple
        user-facing confidence level.
        """

        if confidence >= 0.60:
            return "high"

        if confidence >= 0.30:
            return "medium"

        return "low"

    def predict(self, video_path):
        """
        Predict the cricket shot type from one video.

        Returns:
            predicted_class
            confidence
            confidence_level
            probabilities
        """

        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(
                f"Video not found: {video_path}"
            )

        processed_video = preprocess_video(
            str(video_path)
        )

        model_input = np.expand_dims(
            processed_video,
            axis=0
        )

        predictions = self.model.predict(
            model_input,
            verbose=0
        )[0]

        predicted_index = int(
            np.argmax(predictions)
        )

        confidence = float(
            predictions[predicted_index]
        )

        predicted_class = CLASS_NAMES[
            predicted_index
        ]

        confidence_level = (
            self.get_confidence_level(
                confidence
            )
        )

        probability_dict = {
            class_name: float(probability)
            for class_name, probability
            in zip(
                CLASS_NAMES,
                predictions
            )
        }

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "confidence_percent": confidence * 100,
            "confidence_level": confidence_level,
            "probabilities": probability_dict,
        }