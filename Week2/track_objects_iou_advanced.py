import cv2
import numpy as np
import os
#from utils import load_detections, track_objects

def visualize_tracks(detections, tracks):
    """
    Visualizes the tracks on the video frames.
    """    
    for frame_id, detections_in_frame in detections.items():
        frame = np.zeros((500, 500, 3), dtype=np.uint8)  # Dummy image for visualization (use actual frame if available)

        # Draw the detections
        for x1, y1, x2, y2, conf in detections_in_frame:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        
        # Draw the tracked objects (if any)
        for track_id, track_data in tracks.items():
            for _, x1, y1, x2, y2, _ in track_data:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        
        cv2.imshow(f"Frame {frame_id}", frame)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

def iou(box1, box2):
    """
    Compute IOU between two bounding boxes.
    box1 and box2 are in the format (x1, y1, width, height), NOT (x1, y1, x2, y2).
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Convert to (x1, y1, x2, y2) format
    x1b, y1b, x2b, y2b = x1, y1, x1 + w1, y1 + h1
    x2b2, y2b2, x2b3, y2b3 = x2, y2, x2 + w2, y2 + h2

    # Compute intersection
    xi1, yi1 = max(x1b, x2b2), max(y1b, y2b2)
    xi2, yi2 = min(x2b, x2b3), min(y2b, y2b3)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    # Compute union
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    # Compute IOU
    iou_value = inter_area / union_area if union_area > 0 else 0

    #print(f"Box1: {box1}, Box2: {box2}, Intersection: {inter_area}, Union: {union_area}, IOU: {iou_value}")  # Debugging line

    return iou_value


def load_detections(detection_file):
    """
    Loads detections from a file and organizes them by frame.
    Returns: {frame_id: [(x1, y1, x2, y2, confidence)]}
    """
    detections = {}
    with open(detection_file, "r") as f:
        for line in f:
            frame_id, x1, y1, x2, y2, conf = map(float, line.strip().split())
            frame_id = int(frame_id)

            if frame_id not in detections:
                detections[frame_id] = []

            detections[frame_id].append((x1, y1, x2, y2, conf))
    
    #print(f"Loaded detections: {detections}")  # Debugging line
    return detections


def track_objects(detections, iou_threshold=0.3, confidence_threshold=0.5, grace_period=3):
    tracks = {}
    active_tracks = {}
    next_track_id = 242
    missed_frame_count = {}

    for frame_id in sorted(detections.keys()):
        new_tracks = []
        assigned_tracks = set()

        #print(f"Processing frame {frame_id} with {len(detections[frame_id])} detections")

        for det in detections[frame_id]:
            x1, y1, x2, y2, conf = det

            if conf < confidence_threshold:
                continue

            best_iou = 0
            best_track_id = None

            for track_id, last_bbox in active_tracks.items():
                iou_score = iou(last_bbox, (x1, y1, x2, y2))
                if iou_score > best_iou:
                    best_iou = iou_score
                    best_track_id = track_id

            if best_iou >= iou_threshold:
                tracks[best_track_id].append((frame_id, x1, y1, x2, y2, conf))
                active_tracks[best_track_id] = (x1, y1, x2, y2)
                assigned_tracks.add(best_track_id)
                missed_frame_count[best_track_id] = 0
            else:
                new_tracks.append((x1, y1, x2, y2, conf))

        for det in new_tracks:
            x1, y1, x2, y2, conf = det
            tracks[next_track_id] = [(frame_id, x1, y1, x2, y2, conf)]
            active_tracks[next_track_id] = (x1, y1, x2, y2)
            missed_frame_count[next_track_id] = 0
            next_track_id += 1

        for track_id in list(active_tracks.keys()):
            if track_id not in assigned_tracks:
                missed_frame_count[track_id] += 1
                if missed_frame_count[track_id] > grace_period:
                    del active_tracks[track_id]

    return tracks




def save_tracks(tracks, output_file):
    with open(output_file, "w") as f:
        for track_id, track_data in tracks.items():
            for frame_id, x1, y1, x2, y2, conf in track_data:
                # Ensure width and height are positive
                w = abs(x2 - x1)
                h = abs(y2 - y1)
                f.write(f"{frame_id}, {track_id}, {x1}, {y1}, {w}, {h}, {conf:.4f}, -1, -1, -1\n")

def search_best_params(detection_file, output_dir):
    # Define parameter search ranges
    iou_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    confidence_thresholds = [0.2, 0.4, 0.6, 0.8]
    grace_periods = [1, 3, 5, 8, 12, 15]
    
    # Load detections once
    detections = load_detections(detection_file)

    # Make output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Iterate over all parameter combinations
    for iou_threshold in iou_thresholds:
        for confidence_threshold in confidence_thresholds:
            for grace_period in grace_periods:
                # Print parameter combination being tested
                print(f"Testing with IoU threshold: {iou_threshold}, "
                      f"Confidence threshold: {confidence_threshold}, "
                      f"Grace period: {grace_period}")

                # Track objects with the current parameter combination
                tracked_objects = track_objects(
                    detections, 
                    iou_threshold=iou_threshold, 
                    confidence_threshold=confidence_threshold, 
                    grace_period=grace_period
                )
                
                # Generate output file name based on parameters
                output_file = os.path.join(
                    output_dir, 
                    f"tracked_iou_{iou_threshold}_conf_{confidence_threshold}_grace_{grace_period}.txt"
                )
                
                # Save the tracked objects to the output file
                save_tracks(tracked_objects, output_file)

                print(f"Results saved to {output_file}")

# file paths
detection_file = "Week2/det_yolo_v8n_fine_tuned.txt"
output_dir = "Week2/tracking_results/param_search"


# main
search_best_params(detection_file, output_dir)

# Visualize tracked objects
#visualize_tracks(detections, tracked_objects)
