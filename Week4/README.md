# Multi-Camera Vehicle Tracking

## Overview
This project focuses on multi-camera vehicle tracking using object detection, tracking, and re-identification (ReID). The pipeline involves:

- Object Detection using YOLO.

- Tracking using SORT.

- Cross-camera Re-identification (ReID) to match objects across multiple camera views.

- Filtering of detections based on matched object pairs.

- Evaluation using HOTA and IDF1 metrics.

## Folder Structure
```
project_root/
│── output/
│   ├── detection/
│   │   ├── S01/
│   │   │   ├── detections_1.txt
│   │   │   ├── ...
│   │   ├── S03/
│   │   │   ├── detections_10.txt
│   │   │   ├── ...
│   │   ├── S04/
│   │   │   ├── detections_16.txt
│   │   │   ├── ...
│   ├── filtered_detection/
│   │   ├── S01/
│   │   │   ├── filtered_detections_1.txt
│   │   │   ├── ...
│   │   ├── S03/
│   │   │   ├── filtered_detections_10.txt
│   │   │   ├── ...
│   │   ├── S04/
│   │   │   ├── filtered_detections_16.txt
│   │   │   ├── ...
│── matched_objects_S01.txt
│── matched_objects_S03.txt
│── matched_objects_S04.txt
│── filtering.py
│── matching.py
│── HOTA-IDF1.py
│── README.md
```


## Filtering Detections
The `filtering.py` script filters detections based on object matching.

### Usage:
```bash
python filtering.py
```
This script:
1. Loads the matched object pairs from `matched_objects_{sequence}.txt`.
2. Filters detections from `output/detection/{sequence}/detections_{camera_id}.txt`.
3. Saves filtered detections to `output/filtered_detection/{sequence}/filtered_detections_{camera_id}.txt`.

## Evaluation with HOTA and IDF1
The `HOTA-IDF1.py` script evaluates the filtered detections.

### Usage:
```bash
python HOTA-IDF1.py
```
This script computes:
- **HOTA (Higher Order Tracking Accuracy)**: Evaluates both detection and association accuracy.
- **IDF1 (ID F-score)**: Measures identity preservation in tracking.

## Results
After filtering, the output filtered detections will be available in `output/filtered_detection/`. The evaluation metrics will be printed on the console.

## Requirements
- Python 3.x ![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat&logo=python)
- Ultralytics YOLO ![Ultralytics YOLO](https://img.shields.io/badge/Ultralytics-YOLOv8-blue?style=flat&logo=ultralytics)
- torchreid ![torchreid](https://img.shields.io/badge/Library-torchreid-orange)
- collections ![Collections](https://img.shields.io/badge/Library-collections-green)
- torch ![PyTorch](https://img.shields.io/badge/Framework-PyTorch-red?style=flat&logo=pytorch)
- scipy ![SciPy](https://img.shields.io/badge/Library-SciPy-blue?style=flat&logo=scipy)  
- torchvision ![Torchvision](https://img.shields.io/badge/Library-Torchvision-lightgrey) 

