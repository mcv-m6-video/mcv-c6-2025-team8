# Video Surveillance for Road

This project focuses on developing a video surveillance system for road and traffic monitoring. It involves background modeling, object detection, and performance evaluation.

## Table of Contents
- [Overview](#overview)
- [File Descriptions](#Fil_Descriptions)
- [Installation](#installation)
- [Usage](#usage)



## Overview
The project involves:

1. **Gaussian Background Modeling**
   - Model background using the first 25% of video frames.
   - Compute mean and variance for each pixel.
   - Use the remaining 75% to segment the foreground.

2. **Adaptive Background Modeling**
    - **Dynamic Background Update**: Uses an adaptive running average to update the background model.
    - **Foreground Segmentation**: Computes mean and variance per pixel and applies an adaptive threshold.
    - **Object Detection**: Filters connected components based on size and aspect ratio.

3. **mAP@0.5 vs Alpha Thresholding**
   - Evaluate the segmentation using AP@0.5.
   - Filter noise and refine detected objects.
   - Comparison between adaptive and non-adaptive models, analyzing their performance under different parameters.
     
## File Descriptions
- `non_adaptive.py`- Implements a static background modeling approach for object detection without adaptation.
- `evaluation.py` - Assesses detections from the non-adaptive model by computing IoU and mAP@0.5.
- `adaptive_modeling.py` - Implements adaptive background modeling and object detection.
- `evaluating_sar.py` - Evaluates object detections from the adaptive model using IoU and computes mAP@0.5.

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/mcv-m6-video/mcv-c6-2025-team8.git
   cd mcv-c6-2025-team8
   ```
2. Download the dataset from UAB Campus Virtual (2018/2019 AI City Challenge).

## Usage

### 1. Background Modeling & Foreground Segmentation
```sh
python non_adaptive.py --input vdo.avi --output output.mp4
```

### 2. Evaluation (mAP@0.5)
```sh
python evaluation.py --ground_truth gt.txt --detections det.txt
```

