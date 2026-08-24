from pathlib import Path

import cv2

from inference.predictor import CricketShotPredictor
from pose_estimation.pose_detector import PoseDetector


class CricketShotAnalyzer:
    """
    Combine cricket shot classification
    with MediaPipe pose estimation.

    Final classification pipeline:

    Video
    -> preprocessing
    -> MobileNetV2 feature extraction
    -> GRU classifier
    -> shot prediction

    Pose estimation is performed separately
    for analytics.
    """

    def __init__(
        self,
        model_path=(
            "models/"
            "precomputed_mobilenet_gru_best.keras"
        ),
    ):
        self.predictor = CricketShotPredictor(
            model_path=model_path
        )

    def analyze_video(
        self,
        video_path,
    ):
        """
        Analyze one cricket-shot video.

        Returns:
            prediction
            confidence
            confidence level
            pose statistics
            class probabilities
        """

        video_path = Path(
            video_path
        )

        if not video_path.exists():
            raise FileNotFoundError(
                f"Video not found: "
                f"{video_path}"
            )

        # =================================================
        # SHOT CLASSIFICATION
        # =================================================

        prediction_result = (
            self.predictor.predict(
                video_path
            )
        )

        # =================================================
        # OPEN VIDEO FOR POSE ANALYSIS
        # =================================================

        cap = cv2.VideoCapture(
            str(video_path)
        )

        if not cap.isOpened():
            raise ValueError(
                f"Could not open video: "
                f"{video_path}"
            )

        fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        if fps <= 0:
            fps = 25.0

        detector = PoseDetector()

        total_frames = 0
        pose_detected_frames = 0
        total_landmarks = 0

        # =================================================
        # POSE ANALYSIS
        # =================================================

        while True:

            success, frame = cap.read()

            if not success:
                break

            timestamp_ms = int(
                (
                    total_frames
                    / fps
                )
                * 1000
            )

            _, landmarks = (
                detector.detect_pose(
                    frame,
                    timestamp_ms,
                )
            )

            if landmarks:
                pose_detected_frames += 1

                total_landmarks += len(
                    landmarks
                )

            total_frames += 1

        # =================================================
        # CLEANUP
        # =================================================

        cap.release()

        detector.close()

        # =================================================
        # POSE STATISTICS
        # =================================================

        if total_frames > 0:

            pose_detection_rate = (
                pose_detected_frames
                / total_frames
            )

        else:

            pose_detection_rate = 0.0

        if pose_detected_frames > 0:

            average_landmarks = (
                total_landmarks
                / pose_detected_frames
            )

        else:

            average_landmarks = 0.0

        # =================================================
        # FINAL COMBINED RESULT
        # =================================================

        return {
            "video": str(
                video_path
            ),

            "prediction": (
                prediction_result[
                    "predicted_class"
                ]
            ),

            "confidence": (
                prediction_result[
                    "confidence"
                ]
            ),

            "confidence_percent": (
                prediction_result[
                    "confidence_percent"
                ]
            ),

            "confidence_level": (
                prediction_result[
                    "confidence_level"
                ]
            ),

            "pose": {
                "total_frames": (
                    total_frames
                ),

                "frames_with_pose": (
                    pose_detected_frames
                ),

                "pose_detection_rate": (
                    pose_detection_rate
                ),

                "pose_detection_percent": (
                    pose_detection_rate
                    * 100
                ),

                "average_landmarks": (
                    average_landmarks
                ),
            },

            "probabilities": (
                prediction_result[
                    "probabilities"
                ]
            ),
        }