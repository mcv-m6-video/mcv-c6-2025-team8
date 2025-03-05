import os
import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.metrics import precision_recall_curve, average_precision_score

MODEL_PATH = "runs/detect/train14/weights/best.pt"
CONF_THRESHOLD = 0.5


def load_yolo():
    return YOLO(MODEL_PATH)


def detect_objects_yolo(frame, model):

    results = model(frame)[0]
    detections = []

    for det in results.boxes:
        x1, y1, x2, y2 = det.xyxy[0].tolist()
        confidence = det.conf[0].item()

        if confidence > CONF_THRESHOLD:
            w, h = int(x2 - x1), int(y2 - y1)
            detections.append((int(x1), int(y1), w, h, confidence))

    return detections


def save_detections_txt(detections, output_file):

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        for frame_id, objects in detections.items():
            for (x, y, w, h, score) in objects:
                f.write(f"{frame_id} {x} {y} {w} {h} {score:.4f}\n")


def process_test_frames(test_file, output_txt, parked_cars):

    model = load_yolo()
    detections = {}

    with open(test_file, "r") as f:
        test_images = f.read().splitlines()

    for i, img_path in enumerate(test_images):
        frame_id = int(os.path.basename(img_path).split(".")[0])
        img = cv2.imread(img_path)

        objects = detect_objects_yolo(img, model)
        detections[frame_id] = objects

        print(f"Processed {i+1}/{len(test_images)} test frames...")

    detections = filter_parked_cars(detections, parked_cars)
    save_detections_txt(detections, output_txt)


def filter_parked_cars(detections, parked_cars):

    filtered_detections = {}

    for frame_id, objects in detections.items():
        filtered_objects = []
        for (x, y, w, h, score) in objects:
            box_x2, box_y2 = x + w, y + h
            is_parked = any(
                x1 <= x <= x2 and y1 <= y <= y2 or
                x1 <= box_x2 <= x2 and y1 <= box_y2 <= y2
                for (x1, y1, x2, y2) in parked_cars
            )
            if not is_parked:
                filtered_objects.append((x, y, w, h, score))
        filtered_detections[frame_id] = filtered_objects

    return filtered_detections


def compute_iou(boxA, boxB):

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    return interArea / float(boxAArea + boxBArea - interArea)


def evaluate_detections(detections, ground_truths, iou_threshold=0.5):

    y_true = []
    y_scores = []
    assigned_gts = set()

    # Sort detections by confidence
    detections_sorted = []
    for det in detections:
        frame_id = det["frame_id"]
        bbox = det["bbox"]
        score = det["score"]

        detections_sorted.append((frame_id, bbox, score))

    detections_sorted.sort(key=lambda d: d[2], reverse=True)

    # Iterate through sorted detections and match with ground truths
    all_gt_boxes = {(gt["frame_id"], tuple(gt["bbox"])) for gt in ground_truths}  # Set of all GTs for FN tracking

    detected_gt_boxes = set()

    for frame_id, det_bbox, score in detections_sorted:
        best_iou = 0
        best_gt = None

        # Get ground truths for the current frame
        frame_ground_truths = [gt for gt in ground_truths if gt["frame_id"] == frame_id]

        for gt in frame_ground_truths:
            iou = compute_iou(det_bbox, gt["bbox"])
            if iou > best_iou and (frame_id, tuple(gt["bbox"])) not in assigned_gts:
                best_iou = iou
                best_gt = (frame_id, tuple(gt["bbox"]))

        # Assign TP or FP
        if best_iou >= iou_threshold and best_gt:
            y_true.append(1)
            assigned_gts.add(best_gt)
            detected_gt_boxes.add(best_gt)
        else:
            y_true.append(0)

        y_scores.append(score)

    # Assign FN for undetected ground truth objects
    undetected_gt_boxes = all_gt_boxes - detected_gt_boxes

    for _ in undetected_gt_boxes:
        y_true.append(1)
        y_scores.append(0)

    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)

    return precision, recall, ap


def evaluate_map(detection_file, ground_truth_file):

    detections = load_annotations(detection_file, delimiter=" ", is_detection=True)
    ground_truths = load_annotations(ground_truth_file, delimiter=",", is_detection=False)

    precision_05, recall_05, ap_05 = evaluate_detections(detections, ground_truths, iou_threshold=0.5)
    precision_07, recall_07, ap_07 = evaluate_detections(detections, ground_truths, iou_threshold=0.7)

    iou_scores = [compute_iou(d["bbox"], gt["bbox"]) for d in detections for gt in ground_truths]
    mean_iou = np.mean(iou_scores) if iou_scores else 0

    print(f"mAP@0.5: {ap_05:.4f}, mAP@0.7: {ap_07:.4f}, mIoU: {mean_iou:.4f}")

    return ap_05, ap_07, mean_iou


def load_annotations(annotation_file, delimiter=",", is_detection=False):

    annotations = []

    with open(annotation_file, "r") as f:
        for line in f:
            row = line.strip().split(delimiter)
            frame_id = int(row[0])
            x, y, w, h = map(int, row[1:5]) if is_detection else map(int, row[2:6])
            score = float(row[5]) if is_detection else 1.0
            annotations.append({"frame_id": frame_id, "bbox": (x, y, w, h), "score": score})

    return annotations


test_file = "datasets/test_all_frames.txt"
ground_truth_file = "AICity_data/AICity_data/train/S03/c010/gt/gt.txt"
detection_output = "AICity_data/AICity_data/train/S03/c010/det/det_yolo_v8n_fine_tuned_all_frames.txt"

# Counted and detected from the video
parked_cars = [(1274, 274, 1534, 554), (552, 70, 668, 170), (872, 82, 1014, 150)]

process_test_frames(test_file, detection_output, parked_cars)

evaluate_map(detection_output, ground_truth_file)
