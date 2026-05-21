#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-0}"
DATA_ROOT="${DATA_ROOT:-data/widerface}"
BATCH_SIZE="${BATCH_SIZE:-32}"
WORKERS="${WORKERS:-8}"
WEIGHTS="${WEIGHTS:-runs/train/widerface_yolov5n_5090/weights/best.pt}"

python eval_widerface.py \
  --device "$DEVICE" \
  --weights "$WEIGHTS" \
  --data-root "$DATA_ROOT" \
  --img 640 \
  --batch-size "$BATCH_SIZE" \
  --workers "$WORKERS" \
  --name widerface_yolov5n_5090

