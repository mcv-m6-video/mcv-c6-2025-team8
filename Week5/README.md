# mcv-c6-2025-team8 / Week 5

## Overview

This folder contains all scripts used to run the baseline model from the repository [CVMasterActionRecognitionSpotting](https://github.com/arturxe2/CVMasterActionRecognitionSpotting) and to explore several improvements in Week 5 of the MCV C6 course.

### Enhancements Summary

We extended the baseline model with improvements in two main areas:

1. **Backbone Feature Extractors**  
   Replaced the default backbone with state-of-the-art pretrained models:
   - [ResNet-50](https://arxiv.org/abs/1512.03385)  
   - [EfficientNet](https://arxiv.org/abs/1905.11946)  
   - [ConvNeXt](https://arxiv.org/abs/2201.03545)

2. **Object-Aware Feature Fusion**  
   Integrated object-level information using:
   - [YOLOv8s](https://github.com/ultralytics/ultralytics) for bounding box detection  
   - [ByteTrack](https://github.com/ifzhang/ByteTrack) for object ID tracking

This enabled us to add bounding box coordinates and object identities (e.g., "sports ball" and "person") as additional features to the model.

---

## Folder Structure

```
Week5/
├── checkpoints/                    # Saved model checkpoints
│   └── checkpoint_best.pt
├── config/                         # Config files for each model
│   ├── baseline.json
│   ├── baseline_with_yolo_tracking.json
│   ├── convnext.json
│   ├── efficientnet.json
│   └── resnet50.json
├── main_classification.py         # Baseline training script
├── main_classification_vince.py   # Extended script supporting backbones
├── main_yolo_tracking.py          # YOLO + ByteTrack execution script
├── model_eval.py                  # Inference & evaluation
├── preprocess_tracking.py         # Tracking data preprocessing
├── yolo_tracking.py               # YOLOv8 detection logic
├── zero_pad_file_names.py         # Frame renaming utility
└── README.md
```

---

## Installation

Follow the installation guide from the original repository:  
[github.com/arturxe2/CVMasterActionRecognitionSpotting](https://github.com/arturxe2/CVMasterActionRecognitionSpotting)

Then:

1. Copy all `.py` scripts from `Week5/` into your cloned repo.
2. Place all `*.json` config files into the `config/` folder of the repo.
3. Copy the `checkpoint_best.pt` model into your working directory for evaluation.

---

## Usage

### Training with Pretrained Backbones

You can use `main_classification_vince.py` just like the original `main_classification.py`.  
Specify the model backbone via the `--model` flag:

```bash
python main_classification_vince.py --model efficientnet
```

Valid options include: `resnet50`, `efficientnet`, `convnext`, etc.

---

## Adding YOLO Tracking Features

To include object-level tracking information in your input, follow these steps:

### 1. Zero-pad Frame Filenames
This ensures consistency for YOLO inference.

```bash
python zero_pad_file_names.py
```

### 2. Run YOLOv8s + ByteTrack on Your Frames

```bash
python yolo_tracking.py
```

This will detect and track objects (e.g., players, ball) across frames.

### 3. Preprocess Tracking Features

For better performance and loading speed:

```bash
python preprocess_tracking.py
```

### 4. Train with YOLO-Enhanced Inputs

Finally, use the augmented inputs for training:

```bash
python main_classification_vince.py --model baseline_with_yolo_tracking
```

Make sure the corresponding `baseline_with_yolo_tracking.json` is in your `config/` folder.

---

## References

- **YOLOv8**: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)  
- **ByteTrack**: [Zhang et al., 2021](https://github.com/ifzhang/ByteTrack)  
- **ResNet-50**: [He et al., 2015](https://arxiv.org/abs/1512.03385)  
- **EfficientNet**: [Tan & Le, 2019](https://arxiv.org/abs/1905.11946)  
- **ConvNeXt**: [Liu et al., 2022](https://arxiv.org/abs/2201.03545)
