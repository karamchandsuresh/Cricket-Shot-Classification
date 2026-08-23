from pathlib import Path

import numpy as np
from tensorflow.keras.utils import Sequence

from preprocessing.video_preprocessor import preprocess_video


# Dataset classes in a fixed order.
# The numerical label for each class is its position in this list.
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


class CricketShotDataset(Sequence):
    """
    Keras dataset loader for the Cricket Shot Classification dataset.

    Each video is:
        1. Loaded from disk.
        2. Converted into 16 processed frames.
        3. Assigned its cricket-shot class label.
        4. Returned as part of a training batch.
    """

    def __init__(
        self,
        dataset_dir,
        batch_size=4,
        shuffle=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dataset_dir = Path(dataset_dir)
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.video_paths = []
        self.labels = []

        # Find all videos belonging to each class.
        for class_index, class_name in enumerate(CLASS_NAMES):

            class_dir = self.dataset_dir / class_name

            if not class_dir.exists():
                raise FileNotFoundError(
                    f"Class folder not found: {class_dir}"
                )

            # Dataset videos are AVI files.
            class_videos = sorted(class_dir.glob("*.avi"))

            for video_path in class_videos:
                self.video_paths.append(video_path)
                self.labels.append(class_index)

        self.video_paths = np.array(
            self.video_paths,
            dtype=object,
        )

        self.labels = np.array(
            self.labels,
            dtype=np.int32,
        )

        # Indices allow us to shuffle without changing
        # the relationship between paths and labels.
        self.indices = np.arange(len(self.video_paths))

        self.on_epoch_end()

    def __len__(self):
        """
        Return the number of batches in one epoch.
        """

        return int(
            np.ceil(
                len(self.video_paths)
                / self.batch_size
            )
        )

    def __getitem__(self, batch_index):
        """
        Load and return one batch.
        """

        start = batch_index * self.batch_size

        end = min(
            start + self.batch_size,
            len(self.video_paths),
        )

        batch_indices = self.indices[start:end]

        batch_videos = []
        batch_labels = []

        for index in batch_indices:

            video_path = str(
                self.video_paths[index]
            )

            label = self.labels[index]

            video = preprocess_video(video_path)

            batch_videos.append(video)
            batch_labels.append(label)

        X = np.asarray(
            batch_videos,
            dtype=np.float32,
        )

        y = np.asarray(
            batch_labels,
            dtype=np.int32,
        )

        return X, y

    def on_epoch_end(self):
        """
        Shuffle video order after every epoch.
        """

        if self.shuffle:
            np.random.shuffle(self.indices)

    @property
    def num_samples(self):
        """
        Number of videos contained in this dataset split.
        """

        return len(self.video_paths)

    @property
    def class_names(self):
        """
        Return the class names.
        """

        return CLASS_NAMES