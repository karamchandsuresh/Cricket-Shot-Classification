from pathlib import Path

import cv2
import mediapipe as mp


class PoseDetector:
    """
    Pose detector using the MediaPipe Tasks Pose Landmarker API.
    """

    def __init__(self):
        project_root = Path(__file__).resolve().parent

        model_path = (
            project_root
            / "models"
            / "pose_landmarker.task"
        )

        if not model_path.exists():
            raise FileNotFoundError(
                f"Pose model not found: {model_path}"
            )

        base_options = mp.tasks.BaseOptions(
            model_asset_path=str(model_path)
        )

        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=(
                mp.tasks.vision.RunningMode.VIDEO
            ),
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.detector = (
            mp.tasks.vision.PoseLandmarker.create_from_options(
                options
            )
        )

    def detect_pose(
        self,
        frame,
        timestamp_ms,
    ):
        """
        Detect pose landmarks from one OpenCV video frame.

        Parameters:
            frame:
                OpenCV BGR frame.

            timestamp_ms:
                Timestamp of the frame in milliseconds.

        Returns:
            annotated_frame:
                Frame with pose skeleton drawn.

            landmarks:
                List of detected landmark dictionaries.
        """

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        result = self.detector.detect_for_video(
            mp_image,
            timestamp_ms
        )

        annotated_frame = frame.copy()

        landmarks = []

        if result.pose_landmarks:

            pose_landmarks = result.pose_landmarks[0]

            height, width, _ = frame.shape

            for landmark_id, landmark in enumerate(
                pose_landmarks
            ):
                landmarks.append(
                    {
                        "id": landmark_id,
                        "x": landmark.x,
                        "y": landmark.y,
                        "z": landmark.z,
                        "visibility": (
                            landmark.visibility
                            if landmark.visibility is not None
                            else 0.0
                        ),
                    }
                )

            self._draw_pose(
                annotated_frame,
                pose_landmarks,
                width,
                height
            )

        return annotated_frame, landmarks

    def _draw_pose(
        self,
        frame,
        landmarks,
        width,
        height,
    ):
        """
        Draw pose landmarks and skeleton connections.
        """

        # BlazePose / MediaPipe 33-landmark connections
        pose_connections = [
            (0, 1), (1, 2), (2, 3), (3, 7),
            (0, 4), (4, 5), (5, 6), (6, 8),
            (9, 10),

            (11, 12),

            (11, 13),
            (13, 15),

            (12, 14),
            (14, 16),

            (11, 23),
            (12, 24),
            (23, 24),

            (23, 25),
            (25, 27),

            (24, 26),
            (26, 28),

            (27, 29),
            (29, 31),

            (28, 30),
            (30, 32),

            (15, 17),
            (15, 19),
            (15, 21),

            (16, 18),
            (16, 20),
            (16, 22),
        ]

        points = []

        for landmark in landmarks:

            x = int(
                landmark.x * width
            )

            y = int(
                landmark.y * height
            )

            points.append((x, y))

        # Draw skeleton connections
        for start_id, end_id in pose_connections:

            if (
                start_id < len(points)
                and end_id < len(points)
            ):
                cv2.line(
                    frame,
                    points[start_id],
                    points[end_id],
                    (0, 255, 0),
                    2,
                )

        # Draw landmark points
        for point in points:

            cv2.circle(
                frame,
                point,
                4,
                (0, 0, 255),
                -1,
            )

    def close(self):
        """
        Release MediaPipe resources.
        """

        self.detector.close()