import os
import cv2

video_path = "AICity_data/AICity_data/train/S03/c010/vdo.avi"
gt_path = "AICity_data/AICity_data/train/S03/c010/gt/gt.txt"
frames_dir = "data_all_frames/images/"
labels_dir = "data_all_frames/labels/"
train_file = "train_all.txt"
val_file = "val_all.txt"
test_file = "test_all.txt"

os.makedirs(frames_dir, exist_ok=True)
os.makedirs(labels_dir, exist_ok=True)

annotations = {}
with open(gt_path, "r") as f:
    for line in f.readlines():
        parts = line.strip().split(",")
        frame_id = int(parts[0])
        x, y, w, h = map(int, parts[2:6])

        # Convert to YOLO format
        img_w, img_h = 1920, 1080
        x_center = (x + w / 2) / img_w
        y_center = (y + h / 2) / img_h
        w_norm = w / img_w
        h_norm = h / img_h

        if frame_id not in annotations:
            annotations[frame_id] = []
        annotations[frame_id].append(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#  Training and validation split based on total frames
train_val_cutoff = int(total_frames * 0.25)
train_val_frames = list(range(train_val_cutoff))
test_frames = list(range(train_val_cutoff, total_frames))

frame_paths = []
frame_id = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_filename = f"{frame_id:06d}.jpg"
    frame_path = os.path.join(frames_dir, frame_filename)
    cv2.imwrite(frame_path, frame)
    frame_paths.append((frame_id, frame_path))

    # Save annotations
    label_path = os.path.join(labels_dir, f"{frame_id:06d}.txt")
    if frame_id in annotations:
        with open(label_path, "w") as f:
            f.write("\n".join(annotations[frame_id]))
    else:
        open(label_path, "w").close()

    frame_id += 1

cap.release()

# Split first 25% of frames into 80% train / 20% validation
split_idx = int(len(train_val_frames) * 0.8)
train_images = [path for frame_id, path in frame_paths if frame_id in train_val_frames[:split_idx]]
val_images = [path for frame_id, path in frame_paths if frame_id in train_val_frames[split_idx:]]

# Remaining 75% is for testing
test_images = [path for frame_id, path in frame_paths if frame_id in test_frames]

with open(train_file, "w") as f:
    f.write("\n".join(train_images))

with open(val_file, "w") as f:
    f.write("\n".join(val_images))

with open(test_file, "w") as f:
    f.write("\n".join(test_images))

print(f"Extracted {len(frame_paths)} frames")
print(f"Train: {len(train_images)} frames → {train_file}")
print(f"Validation: {len(val_images)} frames → {val_file}")
print(f"Test: {len(test_images)} frames → {test_file}")
print(f"Saved labels (including empty ones) in {labels_dir}")
