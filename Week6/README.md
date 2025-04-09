# mcv-c6-2025-team8 / Week 6

## Overview

This folder contains all scripts used to run the baseline model from the repository [CVMasterActionSpotting](https://github.com/arturxe2/CVMasterActionSpotting.git) and to explore several improvements in Week 6 of the MCV C6 course.

### Our Experiments
- baseline + Bi-LSTM + fine-tune
- baseline + Bi-LSTM
- x3d + Bi-LSTM
- baseline
- baseline + transformer
- x3d
  
---

## Folder Structure

```
Week6/
├── checkpoints/                    # Saved model checkpoints
│   └── checkpoint_best.pt
├── config/                         # Config files for each model
│   ├── baseline.json
│   ├── baseline_with_yolo.json
│   ├── bi_lstm.json
│   ├── x3d.json
│   └── transformer.json
│   ├── x3d_and_bi_lstm.json
│   ├── README.md
├── main_spotting.py
├── main_spotting_bi_lstm.py
├── main_spotting_x3d.py
├── main_spotting_x3d_and_bi_lstm.py
└── README.md
```

---

## Installation

Follow the installation guide from the original repository:  
[github.com/arturxe2/CVMasterActionSpotting](https://github.com/arturxe2/CVMasterActionSpotting.git)

Then:

1. Copy all `.py` scripts from `Week5/` into your cloned repo.
2. Place all `*.json` config files into the `config/` folder of the repo.
3. Copy the `checkpoint_best.pt` model into your working directory for evaluation.

---

## Usage

### Training

You can use `main_spotting_bi_lstm.py` just like the original `main_spotting.py`.  
Specify the model via the `--model` flag:

```bash
python main_spotting_bi_lstm.py --model bi_lstm
```

---

## References

- **YOLOv8**: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)  
- **X3D**: [pytorchvideo](https://github.com/facebookresearch/pytorchvideo)  
- **Bi-LSTM**: [Bi-LSTM](https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html)  
