import cv2
import numpy as np
import pyflow
from sort.sort import Sort


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


def save_tracks(tracks, output_file, frame_id):

    with open(output_file, "a") as f:
        for track in tracks:
            x1, y1, x2, y2, track_id = map(int, track[:5])
            w = abs(x2 - x1)
            h = abs(y2 - y1)
            f.write(f"{frame_id}, {track_id}, {x1}, {y1}, {w}, {h}, -1, -1, -1\n")


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


def track_objects_with_kalman_pyflow(detection_file, video_path, output_file, confidence_threshold=0.6):

    detections = load_detections(detection_file)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    flow_estimator = PyFlowEstimator()
    tracker = Sort()  # Kalman Filtering

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

        new_detections = []
        if frame_id in detections:
            for x1, y1, x2, y2, conf in detections[frame_id]:
                if conf < confidence_threshold:
                    continue

                new_bbox = apply_optical_flow(flow, (x1, y1, x2, y2))
                new_detections.append([*new_bbox, conf])

        dets = np.array(new_detections) if new_detections else np.empty((0, 5))

        # Kalman Filter
        tracked_objects = tracker.update(dets)

        save_tracks(tracked_objects, output_file, frame_id)

    cap.release()
    return tracked_objects


# File paths
detection_file = "AICity_data/AICity_data/train/S03/c010/det/det_yolo_v8n_fine_tuned.txt"
video_path = "AICity_data/AICity_data/train/S03/c010/vdo.avi"
output_file = "tracking_results/track_kalman_pyflow.txt"


# File path testing
# detection_file = "AICity_data/AICity_data/train/S03/c010/det/det_yolo_v8n_fine_tuned_test.txt"

# Run tracking
track_objects_with_kalman_pyflow(detection_file, video_path, output_file)
