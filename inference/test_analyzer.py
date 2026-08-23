from inference.analyzer import CricketShotAnalyzer


VIDEO_PATH = (
    "dataset/cricketshot/test/cover/cover_0001.avi"
)


def main():

    analyzer = CricketShotAnalyzer()

    result = analyzer.analyze_video(
        VIDEO_PATH
    )

    print(
        "\n========== COMBINED ANALYSIS ==========\n"
    )

    print(
        "Video:",
        result["video"]
    )

    print(
        "Predicted Shot:",
        result["prediction"]
    )

    print(
        "Confidence:",
        f"{result['confidence_percent']:.2f}%"
    )

    print(
        "Confidence Level:",
        result["confidence_level"]
    )

    print("\n--- Pose Analysis ---")

    print(
        "Total frames:",
        result["pose"]["total_frames"]
    )

    print(
        "Frames with pose:",
        result["pose"]["frames_with_pose"]
    )

    print(
        "Pose detection rate:",
        f"{result['pose']['pose_detection_percent']:.2f}%"
    )

    print(
        "Average landmarks:",
        f"{result['pose']['average_landmarks']:.2f}"
    )

    print("\n--- Class Probabilities ---")

    for class_name, probability in (
        result["probabilities"].items()
    ):

        print(
            f"{class_name:12s}: "
            f"{probability * 100:.2f}%"
        )

    print(
        "\n=======================================\n"
    )


if __name__ == "__main__":
    main()