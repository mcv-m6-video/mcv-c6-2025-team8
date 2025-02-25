import cv2
import os


def load_annotations(annotation_file, delimiter=",", is_detection=False):
    annotations = {}
    with open(annotation_file, "r") as f:
        for line in f:
            row = line.strip().split(delimiter)
            frame_id = int(row[0])

            if is_detection:
                x, y, w, h = map(int, row[1:5])
                score = float(row[5])
                if frame_id not in annotations:
                    annotations[frame_id] = []
                annotations[frame_id].append((x, y, w, h, score))
            else:
                x, y, w, h = map(int, row[2:6])
                if frame_id not in annotations:
                    annotations[frame_id] = []
                annotations[frame_id].append((x, y, w, h))
    return annotations


def visualize(video_path, gt_file, det_file, output_path=None):
    cap = cv2.VideoCapture(video_path)
    gt_boxes = load_annotations(gt_file, delimiter=",", is_detection=False)
    det_boxes = load_annotations(det_file, delimiter=" ", is_detection=True)

    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        height = 700
        width = int(frame.shape[1] * (height / frame.shape[0]))
        frame = cv2.resize(frame, (width, height))  # Resize height only

        # Draw GT boxes (blue)
        if frame_id in gt_boxes:
            for (x, y, w, h) in gt_boxes[frame_id]:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, "GT", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Draw detected boxes (green)
        if frame_id in det_boxes:
            for (x, y, w, h, score) in det_boxes[frame_id]:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"Det: {score:.2f}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Visualization", frame)

        if output_path:
            os.makedirs(output_path, exist_ok=True)
            cv2.imwrite(f"{output_path}/frame_{frame_id}.png", frame)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Example usage
video_path = "AICity_data/AICity_data/train/S03/c010/vdo.avi"
gt_file = "AICity_data/AICity_data/train/S03/c010/gt/gt.txt"
det_file = "AICity_data/AICity_data/train/S03/c010/det/det_masks_gausian_6.txt"

visualize(video_path, gt_file, det_file, output_path="output_visualization")
