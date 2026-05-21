#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-0}"
DATA_ROOT="${DATA_ROOT:-data/widerface}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-24}"
WORKERS="${WORKERS:-8}"

python train_widerface.py \
  --device "$DEVICE" \
  --data-root "$DATA_ROOT" \
  --cfg models/yolov5n.yaml \
  --weights '' \
  --epochs "$EPOCHS" \
  --img 800 \
  --batch-size "$BATCH_SIZE" \
  --workers "$WORKERS" \
  --name widerface_yolov5n_scratch_800

