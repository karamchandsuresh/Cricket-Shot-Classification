from inference.predictor import CricketShotPredictor


VIDEO_PATH = (
    "dataset/cricketshot/test/cover/cover_0001.avi"
)


def main():
    predictor = CricketShotPredictor()

    result = predictor.predict(
        VIDEO_PATH
    )

    print("\n========== PREDICTION RESULT ==========\n")

    print(
        "Video:",
        VIDEO_PATH
    )

    print(
        "Prediction:",
        result["predicted_class"]
    )

    print(
        "Raw predicted class:",
        result["raw_predicted_class"]
    )

    print(
        "Confidence:",
        f"{result['confidence'] * 100:.2f}%"
    )

    print(
        "Accepted:",
        result["accepted"]
    )

    print(
        "Confidence threshold:",
        f"{result['confidence_threshold'] * 100:.0f}%"
    )

    print("\nClass probabilities:")

    for class_name, probability in (
        result["probabilities"].items()
    ):
        print(
            f"{class_name:12s}: "
            f"{probability * 100:.2f}%"
        )

    print("\n=======================================")


if __name__ == "__main__":
    main()