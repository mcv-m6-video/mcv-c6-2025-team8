from skopt import gp_minimize
import optuna
from skopt.space import Real
import os
import numpy as np
import cv2
import sys
import time
from evaluation_sar import main as evaluate_detections

old_stdout = sys.stdout
log_file = open("Output.log", "w", encoding='utf-8')
sys.stdout = log_file

def load_video_frames(video_path, max_frames=None):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
        count += 1
        if max_frames and count >= max_frames:
            break
    cap.release()
    return np.array(frames)


def compute_background_model(frames):
    mean_bg = np.mean(frames, axis=0)
    var_bg = np.var(frames, axis=0)
    return mean_bg, var_bg


def segment_foreground(frame, mean_bg, var_bg, alpha=3):
    std_bg = np.sqrt(var_bg)
    fg_mask = np.abs(frame - mean_bg) >= alpha * (std_bg + 2)
    return fg_mask.astype(np.uint8) * 255  # Convert to binary mask


def apply_morphology(fg_mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    return fg_mask


def detect_objects(fg_mask, min_area=1000, max_aspect_ratio=3, min_width=25, min_height=25):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        fg_mask, connectivity=8)
    objects = []
    for i in range(1, num_labels):  # Skip background
        x, y, w, h, area = stats[i]
        aspect_ratio = w / float(h)  # To filter out elongated noise

        if (
            area >= min_area
            and aspect_ratio <= max_aspect_ratio
            and w >= min_width and h >= min_height
        ):
            objects.append((x, y, w, h))

        # if area > 500:  # Filter small objects
        #     objects.append((x, y, w, h))
    return objects


def detect_objects_yolo(frame, min_area=1000, max_aspect_ratio=3, min_width=25, min_height=25):
    results = model(frame)  # Detect objects using YOLO
    objects = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
            w = x2 - x1
            h = y2 - y1
            area = w * h  # Calculate bounding box area
            aspect_ratio = w / float(h)  # Calculate aspect ratio

            # Apply filtering conditions
            if (
                area >= min_area
                and aspect_ratio <= max_aspect_ratio
                and w >= min_width and h >= min_height
            ):
                # Store bounding box as (x, y, w, h)
                objects.append((x1, y1, w, h))

    return objects  # Return list of detected objects


def filter_parked_cars(objects, parked_regions):
    """Remove detected objects inside predefined parked car regions."""
    filtered_objects = []
    for (x, y, w, h) in objects:
        in_parked_area = any(px <= x <= px + pw and py <= y <=
                             py + ph for (px, py, pw, ph) in parked_regions)
        if not in_parked_area:
            filtered_objects.append((x, y, w, h))
    return filtered_objects


def save_detections_txt(detections, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        for frame_id, objects in detections.items():
            for (x, y, w, h, score) in objects:
                f.write(f"{frame_id} {x} {y} {w} {h} {score:.4f}\n")

# Define video path
video_path = "AICity_data/train/S03/c010/vdo.avi"
ground_truth_file = "C:/Users/saran/OneDrive/Documenten/C6_videoanalysis/mcv-c6-2025-team8/AICity_data/train/S03/c010/gt/gt.txt"

def process_video_adaptive(video_path, save_txt=True, alpha=3, rho=0.05):
    frames = load_video_frames(video_path)
    split_idx = len(frames) // 4
    mean_bg, var_bg = compute_background_model(frames[:split_idx])

    detections = {}
    output_txt_path = f"AICity_data/AICity_data/train/S03/c010/det/det_masks_gausian_{alpha}_{rho}.txt"

    for i in range(split_idx, len(frames)):
        frame = frames[i]
        fg_mask = segment_foreground(frame, mean_bg, var_bg, alpha=alpha)
        fg_mask = apply_morphology(fg_mask)
        objects = detect_objects(fg_mask)

        detections[i] = [(x, y, w, h, 1.0) for (x, y, w, h) in objects]

        mean_bg = rho * frame + (1 - rho) * mean_bg  
        var_bg = rho * (frame - mean_bg) ** 2 + (1 - rho) * var_bg  

    if save_txt:
        os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)
        save_detections_txt(detections, output_txt_path)

    return output_txt_path

def objective(trial):
    """Objective function for Optuna Bayesian Optimization."""
    alpha = trial.suggest_float("alpha", 1.5, 2)  # Alpha range
    rho = trial.suggest_float("rho", 0.001, 0.005)  # Rho range

    print(f"\n🔍 Testing α={alpha}, ρ={rho}")

    detection_file = process_video_adaptive(video_path, save_txt=True, alpha=alpha, rho=rho)
    _, _, ap = evaluate_detections(detection_file, ground_truth_file, alpha=alpha, rho=rho)

    print(f"📊 α={alpha}, ρ={rho} → AP: {ap:.4f}")
    return -ap  # Minimize negative AP

# Run Bayesian Optimization using Optuna
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=15)  # Number of evaluations

# Get best parameters
best_alpha = study.best_params["alpha"]
best_rho = study.best_params["rho"]
best_ap = -study.best_value

print("\n✅ Best Parameters Found:")
print(f"Optimal α: {best_alpha:.3f}, Optimal ρ: {best_rho:.3f}, Best AP: {best_ap:.4f}")

# Reset stdout
sys.stdout = old_stdout
log_file.close()