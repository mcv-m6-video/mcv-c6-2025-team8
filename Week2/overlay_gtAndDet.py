import cv2
import numpy as np

def load_mot_txt(file_path, is_detection=False):
    """Loads a MOT-style txt file into a dictionary format."""
    data = {}
    with open(file_path, 'r') as f:
        for line in f:
            values = list(map(float, line.strip().split(',')))
            frame = int(values[0])
            track_id = int(values[1])

            if is_detection:
                # Detection format: (x1, y1, x2, y2) -> Convert to (x, y, w, h)
                x1, y1, x2, y2 = values[2:6]
                x, y = x1, y1
                w, h = abs(x2 - x1), abs(y2 - y1)  # Ensure width and height are positive
            else:
                # GT format is already (x, y, w, h)
                x, y, w, h = values[2:6]

            if frame not in data:
                data[frame] = []
            data[frame].append((track_id, x, y, w, h))
    return data


def show_video_with_bboxes(video_path, gt_path, det_path, output_path):
    """Displays video with overlaid GT and Detection bounding boxes and saves it."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return
    
    # Load ground truth and detections
    gt_data = load_mot_txt(gt_path)
    det_data = load_mot_txt(det_path, is_detection=True)

    # Get video properties for saving the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create VideoWriter object to save the output video
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # or 'MJPG' for example
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    frame_num = 1
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # End of video
        
        # Draw GT boxes (Green)
        if frame_num in gt_data:
            for track_id, x, y, w, h in gt_data[frame_num]:
                x, y, w, h = int(x), int(y), int(w), int(h)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"GT {track_id}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        ## Draw Detection boxes (Red)
        if frame_num in det_data:
            for track_id, x, y, w, h in det_data[frame_num]:
                x, y, w, h = int(x), int(y), int(w), int(h)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, f"Det {track_id}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Show the frame
        cv2.imshow("Tracking Visualization", frame)

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
gt_path = "Week2/tracking_results/gt/gt.txt"
det_path = "Week2/tracking_results/trackers/det_yolo_v8n_fine_tuned.txt"
output_path = "../det.avi"
show_video_with_bboxes(video_path, gt_path, det_path, output_path)
