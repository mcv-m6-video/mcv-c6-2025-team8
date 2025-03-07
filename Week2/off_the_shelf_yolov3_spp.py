import os
import cv2
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score

# YOLO-Konfiguration
WEIGHTS_PATH = "models/yolov3-spp.weights"
CONFIG_PATH = "models/yolov3-spp.cfg"
NAMES_PATH = "models/coco.names"
CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
RESIZE_FACTOR = 0.5
CAR_CLASS_ID = 2


def load_yolo():

    net = cv2.dnn.readNetFromDarknet(CONFIG_PATH, WEIGHTS_PATH)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    return net, output_layers


def detect_objects_yolo(frame, net, output_layers):

    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward(output_layers)
    height, width = frame.shape[:2]
    results = []

    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > CONF_THRESHOLD and class_id == CAR_CLASS_ID:
                box = detection[0:4] * np.array([width, height, width, height])
                (center_x, center_y, w, h) = box.astype("int")
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                results.append((x, y, int(w), int(h), confidence))

    return results


def save_detections_txt(detections, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        for frame_id, objects in detections.items():
            for (x, y, w, h, score) in objects:
                f.write(f"{frame_id} {x} {y} {w} {h} {score:.4f}\n")


def process_video(video_path, output_txt, parked_cars):

    cap = cv2.VideoCapture(video_path)
    net, output_layers = load_yolo()
    frame_id = 0
    detections = {}
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        objects = detect_objects_yolo(frame, net, output_layers)
        detections[frame_id] = objects

        percentage_done = (frame_id / total_frames) * 100 if total_frames > 0 else 0
        print(f"Processing frame {frame_id} - {percentage_done:.2f}% done")

        frame_id += 1

    cap.release()
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


def evaluate_detections_old(detections, ground_truths, iou_threshold=0.5):

    y_true = []
    y_scores = []
    # print(ground_truths)

    for det in detections:
        best_iou = 0
        for gt in ground_truths:
            iou = compute_iou(det["bbox"], gt["bbox"])
            if iou > best_iou:
                best_iou = iou
        y_true.append(1 if best_iou >= iou_threshold else 0)
        y_scores.append(det["score"])
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)

    return precision, recall, ap


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
    all_gt_boxes = {(gt["frame_id"], tuple(gt["bbox"])) for gt in ground_truths}

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


video_path = "AICity_data/AICity_data/train/S03/c010/vdo.avi"
ground_truth_file = "AICity_data/AICity_data/train/S03/c010/gt/gt.txt"
detection_output = "AICity_data/AICity_data/train/S03/c010/det/det_yolo_v3_spp.txt"

# Counted and detected from the video
parked_cars = [(1274, 274, 1534, 554), (552, 70, 668, 170), (872, 82, 1014, 150)]

process_video(video_path, detection_output, parked_cars)
evaluate_map(detection_output, ground_truth_file)
