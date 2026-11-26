# HW3 Plan: YOLO Face Detection with Custom Data Loader and Evaluation

## Goal

Complete the assignment by building a reproducible YOLO face-detection workflow on WIDER FACE while keeping the self-implemented part clear:

- Use this YOLOv5 repository as the reference implementation for model definition, training loop, loss, and inference utilities.
- Implement my own WIDER FACE data loader instead of relying on the built-in YOLO-format loader.
- Implement my own evaluation flow for AP/mAP and PR curves required by the homework.
- Train and evaluate on WIDER FACE train/val, then write an English report with curves, metrics, and discussion.

This matches the TA's clarification: the purpose is to run through the training and evaluation process, and using existing YOLO code is acceptable as long as the data loading and evaluation pipeline are not just black-box calls.

## Dataset

Use the full WIDER FACE train/val data currently stored at:

```text
data/widerface/
  WIDER_train/images/
  WIDER_val/images/
  wider_face_split/
```

Checked dataset statistics:

- Train images: 12880
- Val images: 3226
- Train boxes: 159420
- Val boxes: 39708
- Missing referenced train/val images: 0

The official annotation format is:

```text
file_name
number_of_boxes
x1 y1 w h blur expression illumination invalid occlusion pose
```

I will skip boxes where `invalid == 1`, and keep all valid faces as class `0: face`.

## Implementation Scope

### 1. Custom WIDER FACE Data Loader

Planned file:

```text
datasets/widerface_dataset.py
```

Responsibilities:

- Parse `wider_face_train_bbx_gt.txt` and `wider_face_val_bbx_gt.txt`.
- Handle the WIDER FACE edge case where images with `0` boxes still have one dummy annotation line.
- Load images from `WIDER_train/images` and `WIDER_val/images`.
- Convert boxes from absolute `xywh` to normalized YOLO `cx cy w h`.
- Skip invalid boxes and boxes with non-positive width/height.
- Apply resizing/letterbox so images can be batched.
- Return targets in YOLO training format:

```text
[image_index, class_id, cx, cy, w, h]
```

- Provide a custom `collate_fn` for batches.

The loader should be compatible with the existing YOLOv5 training code, so the model/loss/training loop can be reused without treating the dataset as a pre-converted YOLO-label folder.

### 2. Training Entry Point

Planned file:

```text
train_widerface.py
```

Responsibilities:

- Build `WiderFaceDataset` for train and val.
- Create PyTorch `DataLoader`s with the custom `collate_fn`.
- Instantiate a YOLOv5 model from `models/yolov5n.yaml` or `models/yolov5s.yaml` with `nc=1`.
- Reuse YOLOv5 loss and optimizer logic where practical.
- Save checkpoints and `results.csv`.
- Save loss curves for the report.

Main experiment:

```bash
python train_widerface.py \
  --data-root data/widerface \
  --cfg models/yolov5n.yaml \
  --epochs 50 \
  --img 640 \
  --batch-size 16 \
  --weights ''
```

If compute is limited, I will run a smaller subset first to debug the full pipeline, then train on the full train split or a documented sampled subset.

### 3. Custom Evaluation Flow

Planned file:

```text
eval_widerface.py
```

Responsibilities:

- Load a trained checkpoint.
- Run inference on the WIDER FACE val split.
- Apply confidence filtering and NMS.
- Match predictions to ground-truth boxes by IoU and confidence order.
- Compute:
  - Precision
  - Recall
  - mAP@0.5
  - mAP@0.9
  - mAP@[0.5:0.95]
  - PR curve at IoU threshold 0.7
- Save evaluation outputs:

```text
runs/eval/widerface/
  metrics_summary.json
  PR_curve_iou0.7.png
  P_curve.png
  R_curve.png
  sample_predictions/
```

The AP implementation will follow the standard object-detection protocol:

1. Sort predictions by confidence.
2. For each IoU threshold, greedily match each prediction to one unmatched ground-truth box.
3. Accumulate true positives and false positives.
4. Build precision-recall curves.
5. Integrate AP using interpolated precision over recall.

### 4. Experiments

Required main run:

- YOLOv5n or YOLOv5s trained from scratch on WIDER FACE.

Optional comparisons, depending on time/compute:

- Scratch vs pretrained initialization.
- YOLOv5n vs YOLOv5s.
- Input size 416 vs 640.
- Simple augmentation on/off.

These optional experiments are useful because the repository is based on an existing YOLO implementation, so extra analysis helps show understanding beyond just running a framework.

## Report Outline

The final English report will contain:

1. Task description and dataset summary.
2. Explanation of the custom WIDER FACE data loader.
3. Explanation of the custom evaluation protocol.
4. Training setup:
   - model variant
   - epochs
   - image size
   - batch size
   - optimizer
   - hardware
5. Results:
   - training loss curve
   - mAP@0.5
   - mAP@0.9
   - mAP@[0.5:0.95]
   - PR curve at IoU 0.7
   - qualitative predictions
6. Discussion:
   - small-face difficulty in WIDER FACE
   - effect of image size and model size
   - limitations from compute or subset training
7. Conclusion.

## Acceptance Checklist

- [ ] Custom WIDER FACE annotation parser works for train and val.
- [ ] Custom `Dataset` and `DataLoader` return correctly shaped image/target batches.
- [ ] A small overfit/debug run completes without crashing.
- [ ] Full or sampled training run produces loss curves.
- [ ] Custom evaluation script outputs mAP@0.5, mAP@0.9, mAP@[0.5:0.95].
- [ ] PR curve at IoU 0.7 is saved.
- [ ] Example prediction visualizations are saved.
- [ ] English report includes results, curves, discussion, and conclusion.
