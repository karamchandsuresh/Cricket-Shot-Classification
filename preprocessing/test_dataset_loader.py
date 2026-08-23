from preprocessing.dataset_loader import CricketShotDataset


TRAIN_DIR = "dataset/cricketshot/train"


train_loader = CricketShotDataset(
    dataset_dir=TRAIN_DIR,
    batch_size=4,
    shuffle=True,
)


print("\n========== DATASET LOADER TEST ==========\n")

print(
    "Total training videos:",
    train_loader.num_samples,
)

print(
    "Number of batches:",
    len(train_loader),
)

print("\nClass mapping:")

for index, class_name in enumerate(
    train_loader.class_names
):
    print(f"{index}: {class_name}")


print("\nLoading one batch...")

X_batch, y_batch = train_loader[0]


print("\nVideo batch shape:", X_batch.shape)
print("Label batch shape:", y_batch.shape)

print(
    "Labels:",
    y_batch,
)

print(
    "Data type:",
    X_batch.dtype,
)

print(
    "Pixel range:",
    X_batch.min(),
    "to",
    X_batch.max(),
)

print("\n=========================================")