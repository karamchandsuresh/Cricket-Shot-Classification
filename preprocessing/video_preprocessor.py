import cv2
import numpy as np


# Number of frames we want from each video
NUM_FRAMES = 16

# Size of each frame
FRAME_HEIGHT = 112
FRAME_WIDTH = 112


def preprocess_video(video_path):
    """
    Read a video and convert it into a fixed sequence
    of 16 processed frames.

    Output shape:
    (16, 112, 112, 3)
    """

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        raise ValueError(f"No frames found in video: {video_path}")

    # Select 16 evenly spaced frame positions
    frame_indices = np.linspace(
        0,
        total_frames - 1,
        NUM_FRAMES
    ).astype(int)

    processed_frames = []

    for frame_index in frame_indices:

        # Move directly to the required frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        success, frame = cap.read()

        if not success:
            continue

        # Resize frame
        frame = cv2.resize(
            frame,
            (FRAME_WIDTH, FRAME_HEIGHT)
        )

        # OpenCV reads images as BGR
        # Convert them to RGB
        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Convert pixel values from 0-255 to 0-1
        frame = frame.astype(np.float32) / 255.0

        processed_frames.append(frame)

    cap.release()

    # Make sure exactly 16 frames were extracted
    if len(processed_frames) != NUM_FRAMES:
        raise ValueError(
            f"Expected {NUM_FRAMES} frames, "
            f"but extracted {len(processed_frames)}"
        )

    return np.array(processed_frames, dtype=np.float32)