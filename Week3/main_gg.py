import os
import numpy as np
import cv2
import torch
import torchreid
import sys
import time

from PIL import Image
from ultralytics import YOLO
from sort import Sort  # SORT tracker
from sklearn.metrics.pairwise import cosine_similarity

import logging
import contextlib

from collections import defaultdict

# Suppress torchreid logging
logging.getLogger("torchreid").setLevel(logging.ERROR)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
yolo_model = YOLO('yolov8n.pt').to(device)

def save_detections(detections, output_file, score, frame_id):
    """
    [frame, ID, left, top, width, height, score, -1, -1, -1]
    """

    with open(output_file, "a") as f:

        left, top, right, bottom, obj_id, _ = detections
        width = right - left
        height = bottom - top

        f.write(f"{frame_id},{obj_id},{int(left)},{int(top)},{int(width)},{int(height)},{score:.3f},-1,-1,-1\n")



# Dummy ReID feature extractor (Replace with a real model)
# Load pretrained ReID model
model = torchreid.models.build_model(
    name='osnet_x1_0', num_classes=1000, pretrained=True
)
model = model.to(device)
model.eval()  # Set model to evaluation mode

# Define function to extract ReID features
def extract_features(crop):
    _, transform_test = torchreid.data.transforms.build_transforms(
        height=256,
        width=128,
        norm_mean=[0.485, 0.456, 0.406], 
        norm_std=[0.229, 0.224, 0.225]
    )
    
    transform = transform_test  # ✅ Use test transform

    # 🔹 Convert NumPy array to PIL Image
    if isinstance(crop, np.ndarray):
        crop = Image.fromarray(crop)

    img_tensor = transform(crop).unsqueeze(0).to(device)  # ✅ Move to GPU

    with torch.no_grad():
        features = model(img_tensor)  # Extract features
    return features.cpu()  # Move back to CPU if needed



# Run YOLO detection with ReID feature extraction
def detect_with_features(frame):

    results = yolo_model(frame)
    detections = []
    
    for r in results:
        for i, box in enumerate(r.boxes.xyxy):
            x1, y1, x2, y2 = map(int, box[:4])
            score = r.boxes.conf[i].item()
            
            # Extract cropped vehicle
            if score > 0.3:
                obj_crop = frame[y1:y2, x1:x2] 
                
                # Extract appearance features
                with open('NUL', 'w') as fnull:  # Use 'NUL' for Windows
                    with contextlib.redirect_stdout(fnull):
                        feature_vector = extract_features(obj_crop)
                
                detections.append([x1, y1, x2, y2, score, feature_vector])
    
    return detections

# Run SORT tracker for a single camera
def run_sort_tracker(video_path, cam, sequence):
    cap = cv2.VideoCapture(video_path)

    tracker = Sort()
    tracking_results = {}    
        
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) 

        detections = detect_with_features(frame)

        dets = np.array([d[:5] for d in detections])  # Ignore features for SORT
        
        with open('NUL', 'w') as fnull:  # Use 'NUL' for Windows
            with contextlib.redirect_stdout(fnull):
                tracks = tracker.update(dets)

        score = detections[0][4]
        feature_vector = detections[0][5]

        tracking_results[frame_id] = [(t[0], t[1], t[2], t[3], int(t[4]), feature_vector) for t in tracks]

        output_path = f"filtered_detections_{sequence}_{cam}.txt"

        for track in tracking_results[frame_id]:
                x1, y1, x2, y2, track_id = map(int, track[:5])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        save_detections(tracking_results[frame_id][0], output_path, score, frame_id)
                
        cv2.imshow("Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    # out.release()
    cv2.destroyAllWindows()

    return tracking_results

# Match objects across cameras using ReID
def match_objects_across_cameras(detections_cam1, detections_cam2, threshold=0.8):
    matched_objects = []
    
    for obj1 in detections_cam1:
        best_match = None
        best_score = 0
        
        for obj2 in detections_cam2:

            similarity = cosine_similarity(detections_cam1[obj1][0][5].reshape(1, -1), detections_cam2[obj2][0][5].reshape(1, -1))[0][0]
            if similarity > best_score and similarity > threshold:
                best_score = similarity
                best_match = obj2

                print(obj1, obj2, similarity)

        if best_match:
            matched_objects.append((obj1, best_match))
    
    return matched_objects

# Assign global IDs across multiple cameras
global_id_map = {}  # Maps global ID → dictionary of object detections
object_to_global = {}  # Maps object ID → global ID
global_id_counter = 1  # Start global ID from 1

def assign_global_id(obj, detections, cam_id):

    global global_id_counter

    # If the object is already assigned a global ID, return it
    if obj in object_to_global:
        return object_to_global[obj]

    # Compare against existing global ID objects
    for global_id, known_detections in global_id_map.items():
        matched = match_objects_across_cameras(detections[cam_id], known_detections)
        if any(obj == match[0] for match in matched):
            object_to_global[obj] = global_id
            return global_id

    # If no match is found, assign a new global ID
    global_id_map[global_id_counter] = detections[cam_id]
    object_to_global[obj] = global_id_counter
    global_id_counter += 1

    return global_id_counter - 1  # Return the assigned global ID


def process_sequence(sequence, cams):
    camera_tracks = {}
    for cam in cams:
            if cam > 9:
                video_path = f'Week3/aic19-track1-mtmc-train/train/{sequence}/c0{cam}/vdo.avi'
            else:
                video_path = f'Week3/aic19-track1-mtmc-train/train/{sequence}/c00{cam}/vdo.avi'

            camera_tracks[cam]= run_sort_tracker(video_path, cam, sequence=sequence)


    for cam1 in cams:
        for cam2 in cams:
            if cam1 != cam2:
                matched_objects = match_objects_across_cameras(camera_tracks[cam1], camera_tracks[cam2])

                print(matched_objects)
                for obj1, obj2 in matched_objects:
                    global_id = assign_global_id(obj1, camera_tracks, cam1)
                    assign_global_id(obj2, camera_tracks, cam2)


if __name__ == "__main__":

    start = time.time()

    old_stdout = sys.stdout
    log_file = open("MyOutput.log", "w")
    sys.stdout = log_file
    
    sequences = {
        'S01': range(1, 6),  # Cameras 1-5
        'S03': range(10, 16),  # Cameras 10-15
        'S04': range(16, 41)  # Cameras 16-40
    }

    for seq, cams in sequences.items():
        process_sequence(seq, cams)


    end = time.time()
    elapsed_time = end - start

    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)

    print(f"Code execution time: {hours} hours, {minutes} minutes, and {seconds} seconds")

    sys.stdout = old_stdout
    log_file.close()
