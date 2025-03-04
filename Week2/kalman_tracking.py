import cv2
import os
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from sort import Sort

# Load YOLO model
model = YOLO("models/yolov8n.pt")

# Initialize SORT tracker
tracker = Sort()

video_path = "C:/Users/saran/OneDrive/Documenten/C6_videoanalysis/mcv-c6-2025-team8/AICity_data/train/S03/c010/vdo.avi"
output_txt_path = "Week2/detections.txt"
output_frames_dir = "Week2/tracked_frames"
output_video_path = "Week2/tracked_video.avi"

# Create directory to save frames
os.makedirs(output_frames_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Processing video: {video_path}")
print(f"Frame rate: {fps} FPS, Size: {width}x{height}")

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Get class index for "car"
car_class_indices = [
    idx for idx, name in model.names.items() if "car" in name.lower()
]

# Open output file for writing detections
frame_number = 0
with open(output_txt_path, "w") as f:
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO on frame
        results = model(frame)

        # Process YOLO detections
        dets = []
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                
                if cls in car_class_indices:  # Filter only cars
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
                    conf = box.conf[0].item()  # Confidence score

                    dets.append([x1, y1, x2, y2, conf])

                    # Write to file
                    f.write(f"{frame_number}, {x1}, {y1}, {x2}, {y2}, {conf:.4f}\n")

        # Convert detections to NumPy array for SORT
        dets = np.array(dets)

        # Update SORT tracker
        tracked_objects = tracker.update(dets)

        id_colors = {}

        # Draw tracked bounding boxes with object IDs
        for track in tracked_objects:
            x1, y1, x2, y2, track_id = map(int, track[:5])
            w, h = x2 - x1, y2 - y1
            if track_id not in id_colors:
                id_colors[track_id] = (int(track_id * 50 % 255), int(track_id * 150 % 255), int(track_id * 200 % 255))

            color = id_colors[track_id]

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.6, color, 2)

        # Save frame with tracking info
        frame_path = os.path.join(output_frames_dir, f"frame_{frame_number:04d}.png")
        cv2.imwrite(frame_path, frame)

        out_video.write(frame)

        cv2.imshow("YOLO + SORT Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_number += 1

cap.release()
out_video.release()
cv2.destroyAllWindows()

print(f"Detections saved to {output_txt_path}")
print(f"Tracked frames saved in {output_frames_dir}")
print(f"Final tracked video saved as {output_video_path}")
