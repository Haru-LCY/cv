#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "Running optional scratch experiments..."

./scripts/03_train_optional_yolov5s_640.sh
./scripts/04_eval_optional_yolov5s_640.sh

./scripts/05_train_optional_yolov5n_800.sh
./scripts/06_eval_optional_yolov5n_800.sh

./scripts/07_train_optional_yolov5n_416.sh
./scripts/08_eval_optional_yolov5n_416.sh

PRETRAINED_WEIGHTS="${PRETRAINED_WEIGHTS:-yolov5n.pt}"
if [[ -f "$PRETRAINED_WEIGHTS" ]]; then
  echo "Running optional pretrained experiment with $PRETRAINED_WEIGHTS..."
  PRETRAINED_WEIGHTS="$PRETRAINED_WEIGHTS" ./scripts/09_train_optional_yolov5n_pretrained_640.sh
  ./scripts/10_eval_optional_yolov5n_pretrained_640.sh
else
  echo "Skipping pretrained optional experiment: $PRETRAINED_WEIGHTS not found."
  echo "To run it, place yolov5n.pt in repo root or set PRETRAINED_WEIGHTS=/path/to/yolov5n.pt."
fi

echo "All available optional experiments finished."

