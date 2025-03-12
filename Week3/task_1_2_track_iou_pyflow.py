import cv2
import numpy as np
import pyflow


class PyFlowEstimator:

    def __init__(self, alpha=0.012, ratio=0.75, minWidth=20, nOuterFPIterations=7,
                 nInnerFPIterations=1, nSORIterations=30, colType=0):
        self.alpha = alpha
        self.ratio = ratio
        self.minWidth = minWidth
        self.nOuterFPIterations = nOuterFPIterations
        self.nInnerFPIterations = nInnerFPIterations
        self.nSORIterations = nSORIterations
        self.colType = colType

    def preprocess_frame(self, frame):

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img = img[:, :, np.newaxis].astype(np.float64) / 255.0
        return img

    def compute_flow(self, prev_frame, next_frame):

        image_1 = self.preprocess_frame(prev_frame)
        image_2 = self.preprocess_frame(next_frame)

        u, v, _ = pyflow.coarse2fine_flow(
            image_1, image_2,
            self.alpha, self.ratio, self.minWidth,
            self.nOuterFPIterations, self.nInnerFPIterations,
            self.nSORIterations, self.colType
        )

        return np.dstack((u, v))


def apply_optical_flow(flow, bbox):

    x1, y1, x2, y2 = map(int, bbox)
    flow_region = flow[y1:y2, x1:x2]

    if flow_region.size == 0:
        return x1, y1, x2, y2

    dx = np.median(flow_region[..., 0])
    dy = np.median(flow_region[..., 1])

    return int(x1 + dx), int(y1 + dy), int(x2 + dx), int(y2 + dy)


def iou(box1, box2):

    x1, y1, x2, y2 = box1
    x1b, y1b, x2b, y2b = box2

    xi1, yi1 = max(x1, x1b), max(y1, y1b)
    xi2, yi2 = min(x2, x2b), min(y2, y2b)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2b - x1b) * (y2b - y1b)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0


def save_tracks(tracks, output_file, frame_id):
    with open(output_file, "a") as f:
        for track_id, track_data in tracks.items():
            for fid, x1, y1, x2, y2, conf in track_data:
                if fid == frame_id:
                    w = abs(x2 - x1)
                    h = abs(y2 - y1)
                    f.write(f"{frame_id}, {track_id}, {x1}, {y1}, {w}, {h}, {conf:.4f}, -1, -1, -1\n")


def load_detections(detection_file):

    detections = {}

    with open(detection_file, "r") as f:
        for line in f:
            frame_id, x, y, w, h, conf = map(float, line.strip().split())
            frame_id = int(frame_id)

            if frame_id not in detections:
                detections[frame_id] = []

            x2, y2 = x + w, y + h
            detections[frame_id].append((int(x), int(y), int(x2), int(y2), conf))

    return detections


def track_objects_with_iou_pyflow(detection_file, video_path, output_file, iou_threshold=0.1,
                                  confidence_threshold=0.6, grace_period=1):
    detections = load_detections(detection_file)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    flow_estimator = PyFlowEstimator()
    tracks = {}
    active_tracks = {}
    next_track_id = 242
    missed_frame_count = {}

    success, prev_frame = cap.read()
    if not success:
        print("Error: Could not read first frame.")
        return

    open(output_file, "w").close()

    for frame_id in sorted(detections.keys()):
        success, curr_frame = cap.read()
        if not success:
            break

        flow = flow_estimator.compute_flow(prev_frame, curr_frame)
        prev_frame = curr_frame.copy()

        new_tracks = []
        assigned_tracks = set()

        for det in detections[frame_id]:
            x1, y1, x2, y2, conf = det
            if conf < confidence_threshold:
                continue

            predicted_bbox = apply_optical_flow(flow, (x1, y1, x2, y2))
            best_iou = 0
            best_track_id = None

            for track_id, last_bbox in active_tracks.items():
                iou_score = iou(last_bbox, predicted_bbox)
                if iou_score > best_iou:
                    best_iou = iou_score
                    best_track_id = track_id

            if best_iou >= iou_threshold:
                tracks[best_track_id].append((frame_id, *predicted_bbox, conf))
                active_tracks[best_track_id] = predicted_bbox
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

        save_tracks(tracks, output_file, frame_id)

    cap.release()
    return tracks


# File paths
detection_file = "AICity_data/AICity_data/train/S03/c010/det/det_yolo_v8n_fine_tuned.txt"
video_path = "AICity_data/AICity_data/train/S03/c010/vdo.avi"
output_file = "tracking_results/track_iou_pyflow.txt"


# File path testing
# detection_file = "AICity_data/AICity_data/train/S03/c010/det/det_yolo_v8n_fine_tuned_test.txt"

# Run tracking
track_objects_with_iou_pyflow(detection_file, video_path, output_file)
