import cv2
import numpy as np
import os

file_path = "results_avi_2/morph"
output_path = file_path  # Save outputs in the same folder

# Ensure output directory exists
os.makedirs(output_path, exist_ok=True)

def detect_objects_from_segmented_video(video_path, output_txt_path, min_area=100, min_width=10, min_height=10):
    cap = cv2.VideoCapture(video_path)
    detections = {}
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert frame to grayscale if not already
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        # Threshold to ensure binary mask (if needed)
        _, fg_mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Find connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg_mask, connectivity=8)
        
        detections[frame_id] = []
        
        for i in range(1, num_labels):  # Skip background
            x, y, w, h, area = stats[i]
            
            if area >= min_area and w >= min_width and h >= min_height:
                detections[frame_id].append((x, y, w, h, 1.0000))  # Confidence score 1.0000
        
        frame_id += 1
    
    cap.release()
    
    # Save detections to file
    with open(output_txt_path, "w") as f:
        for frame_id, objects in detections.items():
            for (x, y, w, h, score) in objects:
                f.write(f"{frame_id} {x} {y} {w} {h} {score:.4f}\n")
    
    print(f"Detections saved to {output_txt_path}")

# Process all .avi files in the directory
for file in os.listdir(file_path):
    if file.endswith(".avi"):
        video_path = os.path.join(file_path, file)
        output_txt_path = os.path.join(output_path, os.path.splitext(file)[0] + ".txt")
        detect_objects_from_segmented_video(video_path, output_txt_path)

