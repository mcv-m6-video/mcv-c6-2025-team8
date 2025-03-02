import os
import cv2
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score

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
    first_frame = None

    with open(annotation_file, "r") as f:
        lines = f.readlines()
        if not lines:
            return annotations, None  # Return empty list if file is empty
        
        for line in lines:
            row = line.strip().split(delimiter)  # Use specified delimiter
            frame_id = int(row[0])  # Frame number

            if first_frame is None:
                first_frame = frame_id  # Store first frame number

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

    return annotations, first_frame

def align_frames(detections, gt_first_frame):
    """
    Adjust detection frame numbers so they align with the first GT frame.
    """
    if not detections:
        return []
    
    det_first_frame = min(d["frame_id"] for d in detections)
    frame_offset = gt_first_frame - det_first_frame

    for det in detections:
        det["frame_id"] += frame_offset

    return detections

def evaluate_all_detections(detection_folder, ground_truth_file, output_log="evaluation_results.txt", alpha_threshold=0.5):
    with open(output_log, "w") as log_file:
        log_file.write("Detection File | AP@0.5\n")
        log_file.write("---------------------------------\n")

        for detection_file in os.listdir(detection_folder):
            if detection_file.endswith(".txt"):
                detection_path = os.path.join(detection_folder, detection_file)

                # Load GT and detections
                detections, det_first_frame = load_annotations(detection_path, delimiter=" ", is_detection=True)
                if not detections:
                    print(f"Skipping {detection_file}: No detections found.")
                    continue

                ground_truths, gt_first_frame = load_annotations(ground_truth_file, delimiter=",", is_detection=False)

                # Align detection frames
                detections = align_frames(detections, gt_first_frame)

                # Filter low-confidence detections
                detections = [d for d in detections if d["score"] >= alpha_threshold]

                # Evaluate
                _, _, ap = evaluate_detections(detections, ground_truths)

                print(f"{detection_file} -> AP0.5: {ap:.4f}")

                # Log the result
                log_file.write(f"{detection_file} | {ap:.4f}\n")

# Define paths
detection_folder = "results_avi_2/morph"  # Folder where detection .txt files are stored
ground_truth_file = "AICity_data/AICity_data/train/S03/c010/gt/gt.txt"

# Run evaluation
evaluate_all_detections(detection_folder, ground_truth_file)
