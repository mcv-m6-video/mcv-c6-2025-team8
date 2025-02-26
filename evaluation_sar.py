import sys
import csv
import cv2
import numpy as np
import json
from sklearn.metrics import precision_recall_curve, average_precision_score
import sys
import os

old_stdout = sys.stdout
log_file = open("result.log", "w", encoding='utf-8')
sys.stdout = log_file


def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def evaluate_detections(detections, ground_truths, iou_threshold=0.5):
    y_true = []
    y_scores = []

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


def load_annotations(annotation_file, delimiter=",", is_detection=False):
    annotations = []
    with open(annotation_file, "r") as f:
        for line in f:
            row = line.strip().split(delimiter)  # Use specified delimiter
            frame_id = int(row[0])  # Frame number

            if is_detection:
                # Detections: x, y, w, h at indices 1-4
                x, y, w, h = map(int, row[1:5])
                score = float(row[5])  # Score is at index 5
                annotations.append(
                    {"frame_id": frame_id, "bbox": (x, y, w, h), "score": score})
            else:
                # Ground truth: x, y, w, h at indices 2-5
                x, y, w, h = map(int, row[2:6])
                annotations.append(
                    {"frame_id": frame_id, "bbox": (x, y, w, h)})

    return annotations

def main(detection_file, ground_truth_file, alpha_threshold=0.5, alpha=-1, rho=-1):
    detections = load_annotations(detection_file, delimiter=" ", is_detection=True)

    ground_truths = load_annotations(ground_truth_file, delimiter=",", is_detection=False)

    detections = [d for d in detections if d["score"] >= alpha_threshold]
    precision, recall, ap = evaluate_detections(detections, ground_truths)
    print(f"α={alpha}, ρ={rho}: {ap:.4f}")
    return precision, recall, ap

# Example usage
# Replace with actual detection results
detection_file = "AICity_data/AICity_data/train/S03/c010/det/det_masks_gausian.txt"
detection_file = "AICity_data/AICity_data/train/S03/c010/det/det_masks.txt"

# Replace with actual annotations
ground_truth_file = "C:/Users/saran/OneDrive/Documenten/C6_videoanalysis/mcv-c6-2025-team8/AICity_data/train/S03/c010/gt/gt.txt"


alpha = 4.084443440871048
rho = 0.0958002209837511

detection_file = f"C:/Users/saran/OneDrive/Documenten/C6_videoanalysis/mcv-c6-2025-team8/AICity_data/AICity_data/train/S03/c010/det/det_masks_gausian_{alpha}_{rho}.txt"

main(detection_file, ground_truth_file, alpha=alpha, rho=rho)
# print(f"Alpha: {alpha}")

# print('_______________________________________________')

sys.stdout = old_stdout
log_file.close()
