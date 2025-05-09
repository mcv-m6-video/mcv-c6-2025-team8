import torch
import torchreid
import cv2
import numpy as np
import time
from collections import defaultdict
from scipy.spatial.distance import cosine
from torchvision import transforms

# Load a pre-trained ReID model
model = torchreid.models.build_model(
    name='osnet_x1_0',  # Options: 'osnet_x1_0', 'resnet50_fc512', etc.
    num_classes=1,
    pretrained=True
)
model.eval()
model = model.cuda() if torch.cuda.is_available() else model

# Define image pre-processing transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 128)),  # Standard ReID input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_reid_features(image):
    """ Extract ReID features from an image crop """
    image = transform(image).unsqueeze(0)
    image = image.cuda() if torch.cuda.is_available() else image
    with torch.no_grad():
        feature = model(image)
    return feature.cpu().numpy().flatten()

def read_yolo_detections(file_path):
    """ Reads YOLO detection file: frame_id, object_id, x, y, w, h, score """
    detections = defaultdict(list)
    with open(file_path, 'r') as f:
        for line in f:
            values = line.strip().split(",")
            frame_id = int(values[0])
            object_id = int(values[1])
            bbox = list(map(float, values[2:6]))  # x, y, w, h
            score = float(values[6])
            detections[frame_id].append((object_id, bbox, score))
    return detections

def match_objects_across_cameras(sequence, output_file="matched_objects.txt", threshold=0.8, TOLERANCE=100):
    """
    Matches objects across multiple cameras using a pre-trained ReID model.
    
    sequence: The sequence ID (e.g., "S03")
    threshold: Cosine similarity threshold for matching
    output_file: Path to save the results
    TOLERANCE: Maximum frame difference allowed for matching across cameras
    """
    matched_objects = []
    reid_features = {}

    if sequence =='S01':
        camera_ids = [1, 2, 3, 4, 5]
    elif sequence == 'S03':
        camera_ids = [10, 11, 12, 13, 14, 15]
    else:
        camera_ids = [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
    
    if sequence =='S01':
        video_paths = {cam: f'Week3/aic19-track1-mtmc-train/train/{sequence}/c00{cam}/vdo.avi' for cam in camera_ids}
    else:
        video_paths = {cam: f'Week3/aic19-track1-mtmc-train/train/{sequence}/c0{cam}/vdo.avi' for cam in camera_ids}
        
    detection_files = {cam: f"output/detection/{sequence}/detections_{cam}.txt" for cam in camera_ids}

    detections_dict = {cam: read_yolo_detections(detection_files[cam]) for cam in camera_ids}

    video_captures = {cam_id: cv2.VideoCapture(video_paths[cam_id]) for cam_id in camera_ids}

    total_inference_time = 0  # Total time for all objects
    inference_times = []  # Store inference times per object

    # Extract ReID features
    for cam_id, detections in detections_dict.items():
        cap = video_captures[cam_id]
        for frame_id, objects in detections.items():
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = cap.read()
            if not ret:
                continue

            # print(cam_id, frame_id, objects)
            for object_id, bbox, _ in objects:
                x, y, w, h = map(int, bbox)
                crop = frame[y:y+h, x:x+w]
                if crop.shape[0] == 0 or crop.shape[1] == 0:
                    continue
                
                # Measure inference time for the cropped object
                start_time = time.perf_counter()
                feature = extract_reid_features(crop)
                end_time = time.perf_counter()
                
                inference_time = end_time - start_time
                inference_times.append(inference_time)
                total_inference_time += inference_time

                reid_features[(cam_id, object_id, frame_id)] = feature

    # Match objects across cameras
    for (c1, obj1, frame1), feat1 in reid_features.items():
        for (c2, obj2, frame2), feat2 in reid_features.items():
            if c1 == c2: 
                continue  # Avoid matching within the same camera
            
            # Allow matching across different frames, but within a time tolerance
            if abs(frame1 - frame2) > TOLERANCE:  
                continue

            similarity = 1 - cosine(feat1, feat2)
            if similarity > threshold:
                matched_objects.append((frame1, c1, obj1, frame2, c2, obj2, similarity))

                with open(output_file, "a") as f:
                    f.write(f"{frame1}, {c1}, {obj1}, {frame2}, {c2}, {obj2}, {similarity:.4f}\n")

    # Release video resources
    for cap in video_captures.values():
        cap.release()

    # Print statistics
    print(f"Total inference time: {total_inference_time:.4f} seconds")
    if inference_times:
        print(f"Average inference time per object: {sum(inference_times) / len(inference_times):.4f} seconds")
    else:
        print("No objects processed.")

    print(f"Matching results saved in {output_file}")

# Example usage
sequence = "S03"  # Change as needed
match_objects_across_cameras(sequence, output_file=f"matched_objects_{sequence}.txt", TOLERANCE=500)

sequence = "S04"  # Change as needed
match_objects_across_cameras(sequence, output_file=f"matched_objects_{sequence}.txt", TOLERANCE=500)
