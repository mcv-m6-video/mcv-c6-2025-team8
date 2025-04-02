import os
import json
from tqdm import tqdm
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('yolov8s.pt')

# Class IDs for 'person' and 'sports ball'
TARGET_CLASSES = [0, 32]
CONF_THRESHOLD = 0.5

# Define absolute paths
source_dir = '/ghome/c5mcv08/C6/week5/CVMasterActionRecognitionSpotting/tracking/398x224'
output_dir = '/ghome/c5mcv08/C6/week5/CVMasterActionRecognitionSpotting/tracking_results/398x224'

# Recursively find all leaf directories (those that contain .jpg frames)
match_dirs = []
for root, dirs, files in os.walk(source_dir):
    if any(f.endswith('.jpg') for f in files):
        match_dirs.append(root)

# Process each match
for match_path in tqdm(match_dirs, desc="Tracking matches"):
    rel_path = os.path.relpath(match_path, source_dir)
    output_path = os.path.join(output_dir, rel_path)
    os.makedirs(output_path, exist_ok=True)

    results = model.track(
        source=match_path,
        imgsz=640,
        conf=CONF_THRESHOLD,
        iou=0.5,
        classes=TARGET_CLASSES,
        tracker='bytetrack.yaml',
        persist=True,
        save=False,
        stream=True  # IMPORTANT to avoid RAM overload
    )

    tracking_data = []
    for result in results:
        frame_name = os.path.basename(result.path)
        frame_idx = int(''.join(filter(str.isdigit, frame_name)))

        for box in result.boxes:
            if box.id is None:
                continue  # skip untracked objects

            class_id = int(box.cls)
            conf = float(box.conf)
            if class_id in TARGET_CLASSES and conf > CONF_THRESHOLD:
                tracking_data.append({
                    'frame': frame_idx,
                    'track_id': int(box.id),
                    'class_id': class_id,
                    'class_name': model.names[class_id],
                    'bbox': list(map(float, box.xywh[0].tolist())),
                    'confidence': conf
                })

    output_file = os.path.join(output_path, 'Tracking.json')
    with open(output_file, 'w') as f:
        json.dump(tracking_data, f, indent=4)

    print(f"✅ Saved: {output_file}")
