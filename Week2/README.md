# Video Surveillance for Road

This project focuses on object detection and tracking tasks. It involves implementing off-the-shelf and fine-tuned object detection models, performing K-Fold Cross-validation, and evaluating tracking performance using various techniques.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Task 1: Object Detection](#task-1-object-detection)
   - [Task 1.1: Off-the-shelf](#task-11-off-the-shelf)
   - [Task 1.2: Fine-tune to Your Data](#task-12-fine-tune-to-your-data)
   - [Task 1.3: K-Fold Cross-validation](#task-13-k-fold-cross-validation)
3. [Task 2: Object Tracking](#task-2-object-tracking)
   - [Task 2.1: Tracking by Overlap](#task-21-tracking-by-overlap)
   - [Task 2.2: Tracking with a Kalman Filter](#task-22-tracking-with-a-kalman-filter)
   - [Task 2.3: IDF1, HOTA Scores](#task-23-idf1-hota-scores)
4. [File Descriptions](#file-descriptions)
5. [Requirements](#requirements)

---

## Project Overview

This project addresses two main areas:

- **Object Detection:** Detecting objects in video frames using off-the-shelf models, fine-tuning models on custom data, and evaluating performance using K-Fold Cross-validation.
- **Object Tracking:** Implementing and evaluating object tracking using tracking by overlap and a Kalman filter, followed by performance evaluation using IDF1 and HOTA scores.

---

## Task 1: Object Detection

### Task 1.1: Off-the-shelf
- **Description:** Use pre-trained object detection models (YOLOv3, Faster R-CNN) to detect objects in the provided video dataset. Evaluate performance using mAP@0.5.

### Task 1.2: Fine-tune to Your Data
- **Description:** Fine-tune a pre-trained object detection model on a custom video dataset. Train the model using the first 25% of the frames (80% for training and 20% for validation), and evaluate performance on the remaining 75% of frames. Fine-tuning is aimed at improving detection accuracy on the custom dataset.

### Task 1.3: K-Fold Cross-validation
- **Description:** Implement K-Fold Cross-validation to assess the robustness and generalization of the object detection model. Split the dataset into K subsets and train and evaluate the model K times.

---

## Task 2: Object Tracking

### Task 2.1: Tracking by Overlap
- **Description:** Track detected objects by measuring the overlap of bounding boxes from frame to frame. The object is considered tracked if the overlap is above a defined threshold.

### Task 2.2: Tracking with a Kalman Filter
- **Description:** Implement a Kalman filter to predict and update the position of objects in the video.

### Task 2.3: IDF1, HOTA Scores
- **Description:** Use tracking evaluation metrics such as IDF1 and HOTA to assess the quality of the tracking algorithm.

---

## File Descriptions

- **`YOLO_V8_script.py`**: Script for tracking by overlap using YOLOv8. It processes video frames, detects objects using YOLOv8, and tracks them based on bounding box overlap.
- **`kalman_tracking.py`**: Implements tracking with a Kalman filter.
- **`task_2.3.py`**: Script for evaluating tracking performance using IDF1 and HOTA scores.

---

## Requirements

- Python 3.x
- TensorFlow/PyTorch (for object detection models)
- OpenCV
- NumPy
- Detectron (for Faster R-CNN)
- TrackEval (for tracking metrics)
