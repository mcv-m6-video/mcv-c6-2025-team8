import cv2
import numpy as np

def load_detection_data(file_path):
    """Loads detection data into a dictionary."""
    detections = {}
    with open(file_path, 'r') as f:
        for line in f:
            values = list(map(float, line.strip().split()))
            frame_num = int(values[0])
            x, y, w, h = int(values[1]), int(values[2]), int(values[3]), int(values[4])

            if frame_num not in detections:
                detections[frame_num] = []
            detections[frame_num].append((x, y, w, h))
    
    return detections

def overlay_bboxes_on_video(video_path, detections, output_path):
    """Overlays bounding boxes on video frames and saves the result."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return
    
    # Get video properties for saving the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create VideoWriter object to save the output video
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Or 'MJPG' for example
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    frame_num = 1
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # End of video
        
        # Overlay detection bounding boxes
        if frame_num in detections:
            for (x, y, w, h) in detections[frame_num]:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red box
                cv2.putText(frame, f"Det", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Show the frame
        cv2.imshow("Video with Bounding Boxes", frame)

        # Write the frame to the output video
        out.write(frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        
        frame_num += 1

    # Release the video objects
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Video playback finished and output saved.")

# Example Usage
video_path = "C:/Users/Vincent Heuer/OneDrive - Berode GmbH/Dokumente/Master/C6_project/AICity_data/AICity_data/train/S03/c010/vdo.avi"
det_path = "Week2/det_yolo_v8n_fine_tuned.txt"
output_path = "../det.avi"

# Load detections from file
detections = load_detection_data(det_path)

# Overlay bounding boxes and save the video
overlay_bboxes_on_video(video_path, detections, output_path)
