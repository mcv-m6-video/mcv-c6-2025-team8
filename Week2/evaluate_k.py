import os
import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.metrics import precision_recall_curve, average_precision_score

MODEL_PATH = "runs/detect/train22/weights/best.pt"
TEST_FILE = "datasets/test_kC3_all_frames.txt"
CONF_THRESHOLD = 0.5
IMAGE_SIZE = (1920, 1080)


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
    for frame_id, dets in detections.items():
        for det in dets:
            detections_sorted.append((frame_id, det["bbox"], det["score"]))

    detections_sorted.sort(key=lambda d: d[2], reverse=True)

    # Iterate over all frames
    all_frame_ids = set(detections.keys()).union(set(ground_truths.keys()))

    for frame_id in all_frame_ids:
        frame_detections = detections.get(frame_id, [])
        frame_ground_truths = ground_truths.get(frame_id, [])

        # Match detections to ground truths
        for det_bbox, score in [(det["bbox"], det["score"]) for det in frame_detections]:
            best_iou = 0
            best_gt = None

            for gt in frame_ground_truths:
                iou = compute_iou(det_bbox, gt["bbox"])
                if iou > best_iou and (frame_id, tuple(gt["bbox"])) not in assigned_gts:
                    best_iou = iou
                    best_gt = (frame_id, tuple(gt["bbox"]))

            # Assign TP or FP
            if best_iou >= iou_threshold and best_gt:
                y_true.append(1)
                assigned_gts.add(best_gt)
            else:
                y_true.append(0)

            y_scores.append(score)

        # If there are ground truths but no detections, count as False Negatives (FN)
        unmatched_gts = [gt for gt in frame_ground_truths if (frame_id, tuple(gt["bbox"])) not in assigned_gts]
        for _ in unmatched_gts:
            y_true.append(1)
            y_scores.append(0)

    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)

    return precision, recall, ap


def load_ground_truth(test_file):
    annotations = {}

    with open(test_file, "r") as f:
        test_images = f.read().splitlines()

    for img_path in test_images:
        frame_id = int(os.path.basename(img_path).split(".")[0])
        label_path = img_path.replace("images", "labels").replace(".jpg", ".txt")

        if not os.path.exists(label_path):
            continue

        with open(label_path, "r") as f:
            objects = []
            for line in f.readlines():
                parts = line.strip().split()
                x_center, y_center, w, h = map(float, parts[1:5])

                img_w, img_h = IMAGE_SIZE
                x1 = int((x_center - w / 2) * img_w)
                y1 = int((y_center - h / 2) * img_h)
                x2 = int((x_center + w / 2) * img_w)
                y2 = int((y_center + h / 2) * img_h)

                objects.append({"bbox": (x1, y1, x2 - x1, y2 - y1), "score": 1.0})

            annotations[frame_id] = objects
    return annotations


def run_evaluation():
    print(f"Evaluating Fold {TEST_FILE}...")

    model = YOLO(MODEL_PATH)

    with open(TEST_FILE, "r") as f:
        test_images = f.read().splitlines()

    detections = {}

    for idx, img_path in enumerate(test_images):
        frame_id = int(os.path.basename(img_path).split(".")[0])
        img = cv2.imread(img_path)

        results = model(img)[0]

        objects = []
        for det in results.boxes:
            x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
            confidence = det.conf[0].item()

            if confidence > CONF_THRESHOLD:
                w, h = x2 - x1, y2 - y1
                objects.append({"bbox": (x1, y1, w, h), "score": confidence})

        detections[frame_id] = objects

    ground_truths = load_ground_truth(TEST_FILE)

    precision_05, recall_05, ap_05 = evaluate_detections(detections, ground_truths, iou_threshold=0.5)
    precision_07, recall_07, ap_07 = evaluate_detections(detections, ground_truths, iou_threshold=0.7)

    iou_scores = [
        compute_iou(d["bbox"], gt["bbox"])
        for det_list in detections.values() for d in det_list
        for gt_list in ground_truths.values() for gt in gt_list
    ]
    mean_iou = np.mean(iou_scores) if iou_scores else 0

    print(f"\n **Evaluation Results {TEST_FILE}**")
    print(f"mAP@0.5: {ap_05:.4f} mAP@0.7: {ap_07:.4f} mIoU: {mean_iou:.4f}")


run_evaluation()
