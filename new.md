# New Files Added for HW3

This repository is based on YOLOv5. For the homework, I added a custom WIDER FACE training and evaluation pipeline instead of only using the original YOLO-format dataloader and validation script.

## Added Files

### `datasets/widerface_dataset.py`

Custom PyTorch dataset for WIDER FACE.

Main functions:

- Parses official WIDER FACE annotation files:
  - `wider_face_train_bbx_gt.txt`
  - `wider_face_val_bbx_gt.txt`
- Loads images from:
  - `data/widerface/WIDER_train/images`
  - `data/widerface/WIDER_val/images`
- Handles the WIDER FACE special case where images with zero boxes still contain one dummy annotation line.
- Skips invalid face boxes where `invalid == 1`.
- Converts boxes from WIDER FACE format:

```text
x1 y1 w h blur expression illumination invalid occlusion pose
```

to YOLO normalized format:

```text
class cx cy w h
```

- Applies YOLO-style letterbox resizing.
- Returns YOLOv5-compatible training targets:

```text
[image_index, class_id, cx, cy, w, h]
```

### `datasets/__init__.py`

Marks `datasets/` as a Python package so the custom dataset can be imported by the training and evaluation scripts.

### `train_widerface.py`

Custom training entry point for this homework.

Main functions:

- Uses `WiderFaceDataset` instead of YOLOv5's default label-folder dataloader.
- Builds a YOLOv5 detection model from YAML, for example:

```text
models/yolov5n.yaml
models/yolov5s.yaml
```

- Sets the detection task to one class:

```text
0: face
```

- Reuses YOLOv5 model, loss, and optimizer utilities.
- Supports training from scratch with `--weights ''`.
- Supports CUDA training with `--device 0`.
- Saves training outputs to:

```text
runs/train/<run_name>/
  results.csv
  train_loss_curve.png
  weights/
    last.pt
    best.pt
```

The loss curve image is generated automatically from `results.csv`.

Example:

```bash
python train_widerface.py \
  --device 0 \
  --data-root data/widerface \
  --cfg models/yolov5n.yaml \
  --weights '' \
  --epochs 50 \
  --img 640 \
  --batch-size 32 \
  --workers 8 \
  --name widerface_yolov5n_5090
```

### `eval_widerface.py`

Custom evaluation script for this homework.

Main functions:

- Loads a trained checkpoint.
- Runs inference on WIDER FACE validation images.
- Applies YOLOv5 NMS.
- Implements custom detection evaluation:
  - confidence sorting
  - IoU matching
  - true positive / false positive accumulation
  - precision-recall curve generation
  - AP and mAP computation
- Reports required homework metrics:
  - `mAP@0.5`
  - `mAP@0.9`
  - `mAP@0.5:0.95`
  - `AP@0.7`
- Saves the required PR curve at IoU 0.7.
- Saves qualitative prediction examples.

Evaluation outputs are saved to:

```text
runs/eval/<run_name>/
  metrics_summary.json
  PR_curve_iou0.7.png
  sample_predictions/
```

Example:

```bash
python eval_widerface.py \
  --device 0 \
  --weights runs/train/widerface_yolov5n_5090/weights/best.pt \
  --data-root data/widerface \
  --img 640 \
  --batch-size 32 \
  --workers 8 \
  --name widerface_yolov5n_5090
```

### `homework.md`

The original homework requirement document.

It describes:

- building a YOLO detector
- training on a face detection dataset
- reporting training loss curves
- reporting mAP metrics
- reporting PR curves at IoU 0.7

### `plan.md`

Implementation plan for this homework.

It explains the intended scope:

- reference YOLOv5 for model/loss/training utilities
- implement a custom WIDER FACE dataloader
- implement a custom evaluation flow
- train on WIDER FACE
- report the required metrics and figures

## What Was Not Added to Git

The WIDER FACE dataset itself is not committed to git because it is large.

The expected local data layout is:

```text
data/widerface/
  WIDER_train/images/
  WIDER_val/images/
  wider_face_split/
```

This directory is ignored by `.gitignore`.

## Why These Additions Matter

The original YOLOv5 repository already contains a complete object detection framework. The homework, however, asks for a trainable YOLO pipeline and allows referencing existing implementations.

The main self-implemented parts in this submission are:

- WIDER FACE annotation parsing
- custom PyTorch `Dataset` and `DataLoader`
- custom AP/mAP evaluation
- PR curve generation at IoU 0.7
- training and evaluation scripts specialized for face detection

This makes the project more than a direct Ultralytics command-line run while still using YOLOv5 as a reasonable reference implementation.

