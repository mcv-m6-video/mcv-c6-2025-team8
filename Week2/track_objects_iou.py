import cv2
from utils import load_detections, track_objects

def save_tracks(tracks, output_file):
    with open(output_file, "w") as f:
        for track_id, track_data in tracks.items():
            for frame_id, x1, y1, x2, y2, conf in track_data:
                # Ensure width and height are positive
                w = abs(x2 - x1)
                h = abs(y2 - y1)
                f.write(f"{frame_id}, {track_id}, {x1}, {y1}, {w}, {h}, {conf:.4f}, -1, -1, -1\n")



# file paths
detection_file = "Week2/det_yolo_v8n_fine_tuned.txt"
output_file = "Week2/tracked_objects_det_yolo_v8n_fine_tuned.txt"

# main
detections = load_detections(detection_file)
tracked_objects = track_objects(detections, iou_threshold=0.5)

save_tracks(tracked_objects, output_file)
