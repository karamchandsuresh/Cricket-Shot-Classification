import cv2
import numpy as np


# Number of frames used from each video
NUM_FRAMES = 16

# Size of each frame
FRAME_HEIGHT = 112
FRAME_WIDTH = 112


def process_frame(frame):
    """
    Process one video frame.

    Steps:
    1. Resize to 112 x 112.
    2. Convert BGR to RGB.
    3. Convert to float32.
    4. Normalize pixel values to 0-1.
    """

    frame = cv2.resize(
        frame,
        (FRAME_WIDTH, FRAME_HEIGHT)
    )

    frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    frame = frame.astype(np.float32) / 255.0

    return frame


def preprocess_video(video_path):
    """
    Convert a video into exactly 16 uniformly sampled frames.

    AVI videos can sometimes fail when OpenCV jumps directly
    to a specific frame. Therefore, this version reads the video
    sequentially and selects frames as it moves through the clip.

    Output shape:
    (16, 112, 112, 3)
    """

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(
            f"Unable to open video: {video_path}"
        )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if total_frames <= 0:
        cap.release()

        raise ValueError(
            f"No frames found in video: {video_path}"
        )

    # Create 16 uniformly distributed target positions.
    target_indices = np.linspace(
        0,
        total_frames - 1,
        NUM_FRAMES
    ).astype(int)

    processed_frames = []

    current_frame_index = 0
    target_position = 0

    last_valid_frame = None

    while True:

        success, frame = cap.read()

        if not success:
            break

        last_valid_frame = frame

        # Collect all target positions that have now been reached.
        while (
            target_position < NUM_FRAMES
            and current_frame_index >= target_indices[target_position]
        ):

            processed_frames.append(
                process_frame(frame)
            )

            target_position += 1

        if target_position >= NUM_FRAMES:
            break

        current_frame_index += 1

    cap.release()

    # Some AVI files report more frames than can actually
    # be decoded. If the video ends slightly early, use the
    # final valid frame to complete the 16-frame sequence.
    if (
        len(processed_frames) < NUM_FRAMES
        and last_valid_frame is not None
    ):

        final_frame = process_frame(
            last_valid_frame
        )

        while len(processed_frames) < NUM_FRAMES:

            processed_frames.append(
                final_frame.copy()
            )

    if len(processed_frames) != NUM_FRAMES:

        raise ValueError(
            f"Expected {NUM_FRAMES} frames, "
            f"but extracted {len(processed_frames)} "
            f"from video: {video_path}"
        )

    video_array = np.asarray(
        processed_frames,
        dtype=np.float32
    )

    return video_array