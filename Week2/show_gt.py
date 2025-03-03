import cv2
from utils import load_annotations

# File paths
video_path = "C:/Users/Vincent Heuer/OneDrive - Berode GmbH/Dokumente/Master/C6_project/AICity_data/AICity_data/train/S03/c010/vdo.avi"
gt_path = "C:/Users/Vincent Heuer/OneDrive - Berode GmbH/Dokumente/Master/C6_project/AICity_data/AICity_data/train/S03/c010/gt/gt.txt"

# Load video
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("Frame rate:", fps, "\nSize:", width, "x", height, "\nFrame count: ", frame_count)

# Load ground truth annotations
gt_annotations = load_annotations(gt_path)

# Convert to a dictionary {frame_number: [bboxes]}
gt_data = {}
for ann in gt_annotations:
    frame_id = ann["frame_id"]
    x, y, w, h = ann["bbox"]
    x2, y2 = x + w, y + h  # Convert to (x1, y1, x2, y2)

    if frame_id not in gt_data:
        gt_data[frame_id] = []
    gt_data[frame_id].append((x, y, x2, y2))

# Loop over video frames
frame_number = 0  # OpenCV uses 0-based indexing
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Exit when video ends

    # Draw ground truth bounding boxes if they exist for this frame
    if (frame_number + 1) in gt_data:  # Adjust for 1-based annotation
        for (x1, y1, x2, y2) in gt_data[frame_number + 1]:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
            cv2.putText(frame, "GT", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show frame
    cv2.imshow("Video with Ground Truth", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_number += 1  # Move to next frame

# Release resources
cap.release()
cv2.destroyAllWindows()
