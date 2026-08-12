from video_preprocessor import preprocess_video


VIDEO_PATH = "dataset/cricketshot/train/cover/cover_0001.avi"

video = preprocess_video(VIDEO_PATH)

print("Processed video shape:", video.shape)
print("Data type:", video.dtype)
print("Minimum pixel value:", video.min())
print("Maximum pixel value:", video.max())