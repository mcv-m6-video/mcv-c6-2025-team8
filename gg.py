import os
import numpy as np
import cv2
import sys

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


def detect_objects(fg_mask, min_area=1500, max_aspect_ratio=3, min_width=25, min_height=25):
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

    return objects

def save_detections_txt(detections, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        for frame_id, objects in detections.items():
            for (x, y, w, h, score) in objects:
                f.write(f"{frame_id} {x} {y} {w} {h} {score:.4f}\n")


def process_video(video_path, save_output=False, save_txt=True, alpha=3):
    frames = load_video_frames(video_path)
    split_idx = len(frames) // 4  # 25% for background modeling
    mean_bg, var_bg = compute_background_model(frames[:split_idx])

    detections = {}
    output_txt_path = f"AICity_data/AICity_data/train/S03/c010/det/det_masks_gausian_{alpha}.txt"

    for i in range(split_idx, len(frames)):
        fg_mask = segment_foreground(frames[i], mean_bg, var_bg, alpha=alpha)
        fg_mask = apply_morphology(fg_mask)
        objects = detect_objects(fg_mask)

        frame_bgr = cv2.cvtColor(frames[i], cv2.COLOR_GRAY2BGR)
        detections[i] = []

        for (x, y, w, h) in objects:
            score = 1.0  # Assign confidence score (can be improved)
            detections[i].append((x, y, w, h, score))
            cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Foreground", fg_mask)
        cv2.imshow("Detections", frame_bgr)

        if save_output:
            os.makedirs("output", exist_ok=True)
            cv2.imwrite(f"output/frame_{i}.png", frame_bgr)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    if save_txt:
        save_detections_txt(detections, output_txt_path)

    cv2.destroyAllWindows()


video_path = "AICity_data/AICity_data/train/S03/c010/vdo.avi"
alpha_list = [1.5, 2, 2.5, 3, 3.5, 4.5, 5,
              5.5, 6, 6.5, 7, 8, 9, 11, 13]

for alpha in alpha_list:
    process_video(video_path, alpha=alpha)

sys.stdout = old_stdout
log_file.close()
