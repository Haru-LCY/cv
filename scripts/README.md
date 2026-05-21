# Running Scripts

These scripts collect the commands used for HW3 training and evaluation.

They assume the WIDER FACE dataset is available at:

```text
data/widerface/
  WIDER_train/images/
  WIDER_val/images/
  wider_face_split/
```

All scripts should be run from the repository root or directly by path, for example:

```bash
./scripts/00_smoke_train.sh
./scripts/01_train_main_yolov5n_640.sh
./scripts/02_eval_main_yolov5n_640.sh
```

## Common Environment Variables

You can override defaults without editing the scripts:

```bash
DEVICE=0 BATCH_SIZE=64 WORKERS=8 ./scripts/01_train_main_yolov5n_640.sh
```

Common variables:

- `DEVICE`: CUDA device, default `0`
- `DATA_ROOT`: WIDER FACE path, default `data/widerface`
- `WORKERS`: dataloader workers, default `8`
- `BATCH_SIZE`: script-specific batch size
- `EPOCHS`: default `50`

## Recommended Order

1. Smoke test:

```bash
./scripts/00_smoke_train.sh
```

2. Main training and evaluation:

```bash
./scripts/01_train_main_yolov5n_640.sh
./scripts/02_eval_main_yolov5n_640.sh
```

3. Optional experiments:

```bash
./scripts/03_train_optional_yolov5s_640.sh
./scripts/04_eval_optional_yolov5s_640.sh
./scripts/05_train_optional_yolov5n_800.sh
./scripts/06_eval_optional_yolov5n_800.sh
./scripts/07_train_optional_yolov5n_416.sh
./scripts/08_eval_optional_yolov5n_416.sh
```

4. Optional pretrained experiment, if `yolov5n.pt` is available:

```bash
PRETRAINED_WEIGHTS=yolov5n.pt ./scripts/09_train_optional_yolov5n_pretrained_640.sh
./scripts/10_eval_optional_yolov5n_pretrained_640.sh
```

## Output Locations

Training outputs:

```text
runs/train/<run_name>/
  results.csv
  train_loss_curve.png
  weights/best.pt
  weights/last.pt
```

Evaluation outputs:

```text
runs/eval/<run_name>/
  metrics_summary.json
  PR_curve_iou0.7.png
  sample_predictions/
```

