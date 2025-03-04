import numpy as np

def iou(box1, box2):
    """
    Compute IOU between two bounding boxes.
    box = (x1, y1, x2, y2)
    """
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    # Compute intersection
    xi1, yi1 = max(x1, x1g), max(y1, y1g)
    xi2, yi2 = min(x2, x2g), min(y2, y2g)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    # Compute union
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)
    union_area = box1_area + box2_area - inter_area

    # Compute IOU
    return inter_area / union_area if union_area > 0 else 0

def load_annotations(annotation_file, delimiter=",", is_detection=False):
    annotations = []
    with open(annotation_file, "r") as f:
        for line in f:
            row = line.strip().split(delimiter) 
            frame_id = int(row[0])  # Frame number

            if is_detection:
                x, y, w, h = map(int, row[1:5])
                score = float(row[5])  
                annotations.append(
                    {"frame_id": frame_id, "bbox": (x, y, w, h), "score": score})
            else:
                x, y, w, h = map(int, row[2:6])
                annotations.append(
                    {"frame_id": frame_id, "bbox": (x, y, w, h)})

    return annotations

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
    return detections

def track_objects(detections, iou_threshold=0.5):
    """
    Tracks objects using IOU-based association.
    Returns a dictionary of tracks {track_id: [(frame_id, x1, y1, x2, y2, confidence)]}
    """
    tracks = {}  # {track_id: [(frame_id, x1, y1, x2, y2, confidence)]}
    active_tracks = {}  # {track_id: last_seen_bbox}
    next_track_id = 1

    for frame_id in sorted(detections.keys()):
        new_tracks = []  # Detections for this frame that haven't been assigned
        assigned_tracks = set()

        # Compare each detection to active tracks
        for det in detections[frame_id]:
            x1, y1, x2, y2, conf = det
            best_iou = 0
            best_track_id = None

            for track_id, last_bbox in active_tracks.items():
                iou_score = iou(last_bbox, (x1, y1, x2, y2))
                if iou_score > best_iou:
                    best_iou = iou_score
                    best_track_id = track_id

            # If a track is found with enough IOU, assign detection to it
            if best_iou >= iou_threshold:
                tracks[best_track_id].append((frame_id, x1, y1, x2, y2, conf))
                active_tracks[best_track_id] = (x1, y1, x2, y2)
                assigned_tracks.add(best_track_id)
            else:
                new_tracks.append((x1, y1, x2, y2, conf))

        # Create new tracks for unassigned detections
        for det in new_tracks:
            x1, y1, x2, y2, conf = det
            tracks[next_track_id] = [(frame_id, x1, y1, x2, y2, conf)]
            active_tracks[next_track_id] = (x1, y1, x2, y2)
            next_track_id += 1

        # Remove lost tracks (tracks that didn't get updated in this frame)
        active_tracks = {tid: bbox for tid, bbox in active_tracks.items() if tid in assigned_tracks}

    return tracks