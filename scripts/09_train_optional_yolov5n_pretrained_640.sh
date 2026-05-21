#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-0}"
DATA_ROOT="${DATA_ROOT:-data/widerface}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-32}"
WORKERS="${WORKERS:-8}"
PRETRAINED_WEIGHTS="${PRETRAINED_WEIGHTS:-yolov5n.pt}"

if [[ ! -f "$PRETRAINED_WEIGHTS" ]]; then
  echo "Missing pretrained weights: $PRETRAINED_WEIGHTS" >&2
  echo "Set PRETRAINED_WEIGHTS=/path/to/yolov5n.pt or place yolov5n.pt in the repo root." >&2
  exit 1
fi

python train_widerface.py \
  --device "$DEVICE" \
  --data-root "$DATA_ROOT" \
  --cfg models/yolov5n.yaml \
  --weights "$PRETRAINED_WEIGHTS" \
  --epochs "$EPOCHS" \
  --img 640 \
  --batch-size "$BATCH_SIZE" \
  --workers "$WORKERS" \
  --name widerface_yolov5n_pretrained_640

