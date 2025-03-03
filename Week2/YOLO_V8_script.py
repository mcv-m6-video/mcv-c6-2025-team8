from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("models/yolov8n.pt")

# Define paths
video_path = "C:/Users/Vincent Heuer/OneDrive - Berode GmbH/Dokumente/Master/C6_project/AICity_data/AICity_data/train/S03/c010/vdo.avi"
output_txt_path = "Week2/detections.txt"  # Save detection results

# Open video
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("Frame rate:", fps, "\nSize:", width, "x", height)

# Get class index for "car"
car_class_indices = [
    idx for idx, name in model.names.items() if "car" in name.lower()
]

# Open output file for writing detections
with open(output_txt_path, "w") as f:
    frame_number = 0  # OpenCV frames start at 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Exit when video ends

        # Run YOLO on frame
        results = model(frame)

        # Process results
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])  # Class index
                
                if cls in car_class_indices:  # Filter only cars
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
                    conf = box.conf[0].item()  # Confidence score

                    # Write to file: frame_id, x1, y1, x2, y2, confidence
                    f.write(f"{frame_number}, {x1}, {y1}, {x2}, {y2}, {conf:.4f}\n")

                    # Draw rectangle and label on frame
                    label = f"Car {conf:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Show frame (optional)
        cv2.imshow("YOLOv8 Car Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_number += 1  # Increment frame count

# Release resources
cap.release()
cv2.destroyAllWindows()

print(f"Detections saved to {output_txt_path}")

