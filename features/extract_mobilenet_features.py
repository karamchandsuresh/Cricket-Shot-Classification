from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2

from preprocessing.dataset_loader import CricketShotDataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_ROOT = (
    PROJECT_ROOT
    / "dataset"
    / "cricketshot"
)

FEATURE_ROOT = (
    PROJECT_ROOT
    / "features"
    / "mobilenetv2"
)

FEATURE_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


NUM_FRAMES = 16
FRAME_HEIGHT = 112
FRAME_WIDTH = 112
CHANNELS = 3


def build_feature_extractor():
    """
    Build frozen MobileNetV2 feature extractor.
    """

    model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(
            FRAME_HEIGHT,
            FRAME_WIDTH,
            CHANNELS,
        ),
    )

    model.trainable = False

    return model


def preprocess_for_mobilenet(video_batch):
    """
    Our dataset loader provides values in 0-1.

    MobileNetV2 expects approximately -1 to 1.
    """

    return (
        video_batch * 2.0
    ) - 1.0


def extract_split_features(
    split_name,
    feature_extractor,
):
    """
    Extract MobileNetV2 features for one dataset split.
    """

    split_path = (
        DATASET_ROOT
        / split_name
    )

    loader = CricketShotDataset(
        dataset_dir=split_path,
        batch_size=2,
        shuffle=False,
    )

    print(
        f"\nExtracting {split_name} features..."
    )

    print(
        "Videos:",
        loader.num_samples
    )

    all_features = []
    all_labels = []

    for batch_index in range(
        len(loader)
    ):
        X_batch, y_batch = loader[
            batch_index
        ]

        X_batch = preprocess_for_mobilenet(
            X_batch
        )

        batch_size = X_batch.shape[0]

        # -------------------------------------------------
        # Reshape video batch:
        #
        # (B, 16, 112, 112, 3)
        #
        # into individual frames:
        #
        # (B * 16, 112, 112, 3)
        # -------------------------------------------------

        frames = X_batch.reshape(
            (
                batch_size
                * NUM_FRAMES,
                FRAME_HEIGHT,
                FRAME_WIDTH,
                CHANNELS,
            )
        )

        # -------------------------------------------------
        # MobileNetV2 feature extraction
        #
        # Each frame becomes a 1280-dimensional vector.
        # -------------------------------------------------

        frame_features = (
            feature_extractor.predict(
                frames,
                verbose=0,
            )
        )

        # -------------------------------------------------
        # Restore video structure:
        #
        # (B * 16, 1280)
        #
        # becomes:
        #
        # (B, 16, 1280)
        # -------------------------------------------------

        video_features = (
            frame_features.reshape(
                (
                    batch_size,
                    NUM_FRAMES,
                    1280,
                )
            )
        )

        all_features.append(
            video_features
        )

        all_labels.append(
            np.asarray(
                y_batch,
                dtype=np.int64,
            )
        )

        if (
            (batch_index + 1) % 50 == 0
            or
            (batch_index + 1)
            == len(loader)
        ):
            print(
                f"{batch_index + 1}"
                f"/{len(loader)} batches"
            )

    features = np.concatenate(
        all_features,
        axis=0,
    )

    labels = np.concatenate(
        all_labels,
        axis=0,
    )

    feature_path = (
        FEATURE_ROOT
        / f"{split_name}_features.npy"
    )

    label_path = (
        FEATURE_ROOT
        / f"{split_name}_labels.npy"
    )

    np.save(
        feature_path,
        features,
    )

    np.save(
        label_path,
        labels,
    )

    print(
        f"{split_name} feature shape:",
        features.shape
    )

    print(
        f"{split_name} label shape:",
        labels.shape
    )

    print(
        "Saved:",
        feature_path
    )

    print(
        "Saved:",
        label_path
    )


def main():
    print(
        "Building MobileNetV2 feature extractor..."
    )

    feature_extractor = (
        build_feature_extractor()
    )

    print(
        "Feature extractor ready."
    )

    for split_name in [
        "train",
        "val",
        "test",
    ]:
        extract_split_features(
            split_name,
            feature_extractor,
        )

    print(
        "\nFeature extraction completed."
    )


if __name__ == "__main__":
    main()