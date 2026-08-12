import matplotlib.pyplot as plt

from video_preprocessor import preprocess_video


VIDEO_PATH = "dataset/cricketshot/train/cover/cover_0001.avi"

# Preprocess video into 16 frames
video = preprocess_video(VIDEO_PATH)

# Frames we want to display
frame_numbers = [0, 4, 8, 12, 15]

plt.figure(figsize=(15, 4))

for i, frame_number in enumerate(frame_numbers):

    plt.subplot(1, 5, i + 1)

    # Display frame
    plt.imshow(video[frame_number])

    # Show human-friendly frame number
    plt.title(f"Frame {frame_number + 1}")

    plt.axis("off")


plt.suptitle("Uniformly Sampled Frames - Cover Shot")

plt.tight_layout()

plt.show()