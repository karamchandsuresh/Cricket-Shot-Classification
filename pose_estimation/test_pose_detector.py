import cv2

from pose_estimation.pose_detector import PoseDetector


VIDEO_PATH = (
    "dataset/cricketshot/train/cover/cover_0001.avi"
)


def main():
    cap = cv2.VideoCapture(
        VIDEO_PATH
    )

    if not cap.isOpened():
        raise ValueError(
            f"Could not open video: {VIDEO_PATH}"
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 25.0

    detector = PoseDetector()

    frame_count = 0

    while True:

        success, frame = cap.read()

        if not success:
            break

        timestamp_ms = int(
            (
                frame_count
                / fps
            )
            * 1000
        )

        annotated_frame, landmarks = (
            detector.detect_pose(
                frame,
                timestamp_ms
            )
        )

        cv2.putText(
            annotated_frame,
            f"Landmarks: {len(landmarks)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        cv2.imshow(
            "Cricket Pose Detection",
            annotated_frame,
        )

        frame_count += 1

        if (
            cv2.waitKey(30) & 0xFF
            == ord("q")
        ):
            break

    cap.release()

    detector.close()

    cv2.destroyAllWindows()

    print(
        "Processed frames:",
        frame_count
    )


if __name__ == "__main__":
    main()