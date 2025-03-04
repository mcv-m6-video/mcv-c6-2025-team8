import cv2
from sort import Sort
from utils import load_detections
import numpy as np


# Initialize SORT tracker
tracker = Sort()

# File paths
video_path = "C:/Users/saran/OneDrive/Documenten/C6_videoanalysis/mcv-c6-2025-team8/AICity_data/train/S03/c010/vdo.avi"
detection_file = "Week2/detections.txt"
output_file = "Week2/tracked_objects.txt"

# Load detections
detections = load_detections(detection_file)

# Open video
cap = cv2.VideoCapture(video_path)
frame_number = 0  # OpenCV uses 0-based indexing

# Open output file for writing
with open(output_file, "w") as f:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Exit when video ends

        # Get detections for the current frame
        dets = []
        if frame_number in detections:
            for x1, y1, x2, y2, conf in detections[frame_number]:
                dets.append([x1, y1, x2, y2, conf])  # SORT format

        # Convert to NumPy array
        dets = np.array(dets)

        # Update tracker with new detections
        tracks = tracker.update(dets)

        # Process tracked objects
        for track in tracks:
            x1, y1, x2, y2, track_id = map(int, track[:5])
            w, h = x2 - x1, y2 - y1  # Convert to width/height

            # Write to file: frame_id, track_id, bbox, confidence (-1 for missing values)
            f.write(f"{frame_number}, {track_id}, {x1}, {y1}, {w}, {h}, -1, -1, -1, -1\n")

            # Draw tracking bounding boxes
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Show frame (optional)
        cv2.imshow("SORT Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_number += 1  # Increment frame count

# Release resources
cap.release()
cv2.destroyAllWindows()

print(f"Tracking results saved to {output_file}")