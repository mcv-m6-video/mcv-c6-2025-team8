import os
import numpy as np
from sklearn.model_selection import train_test_split

images_dir = "data_all_frames/images/"
labels_dir = "data_all_frames/labels/"

all_images = sorted([os.path.join(images_dir, img) for img in os.listdir(images_dir) if img.endswith(".jpg")])

K = 4

for fold in range(K):
    print(f"Creating Fold {fold}...")

    train_images, test_images = train_test_split(all_images, test_size=0.75, random_state=fold)

    train_images, val_images = train_test_split(train_images, test_size=0.2, random_state=fold)

    np.savetxt(f"train_kC{fold}_all_frames.txt", train_images, fmt="%s")
    np.savetxt(f"val_kC{fold}_all_frames.txt", val_images, fmt="%s")
    np.savetxt(f"test_kC{fold}_all_frames.txt", test_images, fmt="%s")

    print(f"Fold {fold} - Train: {len(train_images)}, Val: {len(val_images)}, Test: {len(test_images)}")
