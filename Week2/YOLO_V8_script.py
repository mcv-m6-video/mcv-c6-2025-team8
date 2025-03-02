from ultralytics import YOLO
import cv2

model = YOLO("models/yolov8n.pt")

video_path = "C:/Users/Vincent Heuer/OneDrive - Berode GmbH/Dokumente/Master/C6_project/AICity_data/AICity_data/train/S03/c010/vdo.avi"
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("frame rate: ", fps, "\nsize:", width, "x", height)

# Save output video
#out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Exit when video ends

    # Run YOLO on frame
    results = model(frame)

    # Draw results on the frame
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box
            conf = box.conf[0]  # Confidence score
            cls = int(box.cls[0])  # Class index
            label = f"{model.names[cls]} {conf:.2f}"
            
            # Draw rectangle and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Write frame to output video
    #out.write(frame)

    # Show frame (optional)
    cv2.imshow("YOLOv8 Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
