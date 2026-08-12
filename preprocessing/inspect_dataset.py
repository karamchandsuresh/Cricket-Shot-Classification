import cv2
from pathlib import Path
from collections import Counter

# Path to our cricket video dataset
DATASET_PATH = Path("dataset/cricketshot")

# Dataset splits
SPLITS = ["train", "val", "test"]

# Store information about all videos
durations = []
frame_counts = []
fps_values = []
resolutions = []

broken_videos = []
class_counts = Counter()

total_videos = 0


for split in SPLITS:

    split_path = DATASET_PATH / split

    # Each folder represents one cricket shot class
    for class_folder in split_path.iterdir():

        if not class_folder.is_dir():
            continue

        # Read all AVI videos inside the class
        for video_path in class_folder.glob("*.avi"):

            total_videos += 1

            # Example key: train/cover
            class_counts[f"{split}/{class_folder.name}"] += 1

            cap = cv2.VideoCapture(str(video_path))

            # Check whether OpenCV can read the video
            if not cap.isOpened():
                broken_videos.append(str(video_path))
                continue

            fps = cap.get(cv2.CAP_PROP_FPS)

            frames = int(
                cap.get(cv2.CAP_PROP_FRAME_COUNT)
            )

            width = int(
                cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            )

            height = int(
                cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            )

            # Calculate duration
            if fps > 0:
                duration = frames / fps
                durations.append(duration)

            frame_counts.append(frames)
            fps_values.append(fps)
            resolutions.append((width, height))

            cap.release()


# -------------------------
# Display Dataset Statistics
# -------------------------

print("\n========== DATASET PROFILE ==========\n")

print("Total videos:", total_videos)

print("\n--- Videos per class ---")

for key, value in sorted(class_counts.items()):
    print(f"{key}: {value}")


print("\n--- Video Duration ---")

print("Minimum duration:", round(min(durations), 2), "seconds")
print("Maximum duration:", round(max(durations), 2), "seconds")
print(
    "Average duration:",
    round(sum(durations) / len(durations), 2),
    "seconds"
)


print("\n--- Frame Count ---")

print("Minimum frames:", min(frame_counts))
print("Maximum frames:", max(frame_counts))


print("\n--- FPS Values ---")

print(
    sorted(set(round(fps, 2) for fps in fps_values))
)


print("\n--- Resolutions ---")

for resolution, count in Counter(resolutions).items():
    print(f"{resolution}: {count} videos")


print("\n--- Broken Videos ---")

print("Broken/unreadable videos:", len(broken_videos))

for video in broken_videos:
    print(video)


print("\n=====================================")