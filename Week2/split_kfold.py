import os
import numpy as np
from sklearn.model_selection import KFold, train_test_split


images_dir = "data_all_frames/images/"
labels_dir = "data_all_frames/labels/"

all_images = sorted([os.path.join(images_dir, img) for img in os.listdir(images_dir) if img.endswith(".jpg")])

kf = KFold(n_splits=4, shuffle=False)
all_images = np.array(all_images)

for fold, (trainval_idx, test_idx) in enumerate(kf.split(all_images)):
    trainval_images = all_images[trainval_idx]
    test_images = all_images[test_idx]

    train_images, val_images = train_test_split(trainval_images, test_size=0.1, random_state=42)

    np.savetxt(f"train_k{fold}_all_frames.txt", train_images, fmt="%s")
    np.savetxt(f"val_k{fold}_all_frames.txt", val_images, fmt="%s")
    np.savetxt(f"test_k{fold}_all_frames.txt", test_images, fmt="%s")

    print(f"Fold {fold}: Train={len(train_images)}, Val={len(val_images)}, Test={len(test_images)}")
