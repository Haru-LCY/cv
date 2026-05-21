#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-0}"
DATA_ROOT="${DATA_ROOT:-data/widerface}"
WORKERS="${WORKERS:-4}"

python train_widerface.py \
  --device "$DEVICE" \
  --data-root "$DATA_ROOT" \
  --cfg models/yolov5n.yaml \
  --weights '' \
  --epochs 1 \
  --img 640 \
  --batch-size 8 \
  --workers "$WORKERS" \
  --max-train-samples 64 \
  --max-val-samples 16 \
  --name smoke_5090

