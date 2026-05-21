#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

./scripts/03_train_optional_yolov5s_640.sh
./scripts/04_eval_optional_yolov5s_640.sh
./scripts/05_train_optional_yolov5n_800.sh
./scripts/06_eval_optional_yolov5n_800.sh
./scripts/07_train_optional_yolov5n_416.sh
./scripts/08_eval_optional_yolov5n_416.sh

