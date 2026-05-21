"""Custom WIDER FACE evaluation for the HW3 YOLO detector."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from datasets.widerface_dataset import WiderFaceDataset
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device

TRAPEZOID = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def box_iou_np(boxes1: np.ndarray, boxes2: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Compute pairwise IoU for xyxy boxes."""
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clip(0) * (boxes1[:, 3] - boxes1[:, 1]).clip(0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clip(0) * (boxes2[:, 3] - boxes2[:, 1]).clip(0)

    inter_x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    inter_y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    inter_x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    inter_y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])
    inter = (inter_x2 - inter_x1).clip(0) * (inter_y2 - inter_y1).clip(0)
    return inter / (area1[:, None] + area2[None, :] - inter + eps)


def match_predictions(pred_boxes: np.ndarray, gt_boxes: np.ndarray, iou_thres: float) -> np.ndarray:
    """Greedily match confidence-sorted predictions to ground truth at one IoU threshold."""
    tp = np.zeros(len(pred_boxes), dtype=np.float32)
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return tp

    ious = box_iou_np(pred_boxes, gt_boxes)
    matched_gt: set[int] = set()
    for pred_i in range(len(pred_boxes)):
        gt_i = int(ious[pred_i].argmax())
        if ious[pred_i, gt_i] >= iou_thres and gt_i not in matched_gt:
            tp[pred_i] = 1.0
            matched_gt.add(gt_i)
    return tp


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """Compute interpolated AP with the COCO-style 101-point recall grid."""
    if len(recall) == 0:
        return 0.0
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    return float(TRAPEZOID(np.interp(x, mrec, mpre), x))


def precision_recall_ap(tp: np.ndarray, conf: np.ndarray, num_targets: int):
    """Build PR arrays and AP from per-prediction TP flags and confidence scores."""
    if num_targets == 0:
        return np.array([]), np.array([]), 0.0
    if len(conf) == 0:
        return np.array([0.0]), np.array([1.0]), 0.0

    order = np.argsort(-conf)
    tp = tp[order]
    fp = 1.0 - tp
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / (num_targets + 1e-16)
    precision = tp_cum / (tp_cum + fp_cum + 1e-16)
    return precision, recall, compute_ap(recall, precision)


def plot_pr_curve(recall: np.ndarray, precision: np.ndarray, ap: float, save_path: Path, iou_thres: float):
    """Save the PR curve used in the report."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5), tight_layout=True)
    ax.plot(recall, precision, linewidth=2, label=f"face AP={ap:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve at IoU {iou_thres:.2f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def draw_sample(path: str, gt_boxes: np.ndarray, pred: np.ndarray, save_path: Path):
    """Save one qualitative prediction image with GT in green and predictions in red."""
    im = cv2.imread(path)
    if im is None:
        return
    for box in gt_boxes.astype(int):
        cv2.rectangle(im, tuple(box[:2]), tuple(box[2:]), (0, 180, 0), 2)
    for *xyxy, conf, _cls in pred.tolist():
        xyxy = [int(x) for x in xyxy]
        cv2.rectangle(im, tuple(xyxy[:2]), tuple(xyxy[2:]), (0, 0, 220), 2)
        cv2.putText(im, f"{conf:.2f}", (xyxy[0], max(0, xyxy[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 220), 1)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), im)


def resolve_device(device_arg: str, batch_size: int) -> torch.device:
    """Prefer Apple MPS and avoid silent CPU fallback when MPS is requested."""
    requested = str(device_arg).strip().lower()
    if not requested and torch.backends.mps.is_available():
        requested = "mps"
    device = select_device(requested, batch_size=batch_size)
    if requested == "mps" and device.type != "mps":
        raise RuntimeError(
            "MPS was requested, but torch.backends.mps.is_available() is False in this Python environment. "
            "Fix the PyTorch/MPS environment or choose another device explicitly."
        )
    return device


@torch.no_grad()
def run(
    weights: str,
    data_root: str = "data/widerface",
    imgsz: int = 640,
    batch_size: int = 16,
    conf_thres: float = 0.001,
    nms_iou_thres: float = 0.6,
    device: str = "",
    workers: int = 4,
    project: str = "runs/eval",
    name: str = "widerface",
    max_samples: int | None = None,
    save_samples: int = 16,
):
    """Evaluate a trained detector on WIDER FACE val with custom AP/PR metrics."""
    device_obj = resolve_device(device, batch_size=batch_size)
    save_dir = Path(project) / name
    save_dir.mkdir(parents=True, exist_ok=True)

    model = attempt_load(weights, device=device_obj, fuse=True)
    stride = int(model.stride.max()) if hasattr(model, "stride") else 32
    dataset = WiderFaceDataset(
        root=data_root,
        split="val",
        img_size=imgsz,
        stride=stride,
        augment=False,
        scaleup=False,
        max_samples=max_samples,
    )
    path_to_index = {path: i for i, path in enumerate(dataset.im_files)}
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=WiderFaceDataset.collate_fn,
        pin_memory=device_obj.type != "cpu",
    )

    model.eval()
    iouv = np.linspace(0.5, 0.95, 10)
    all_tp = []
    all_conf = []
    num_targets = 0
    saved = 0

    for ims, _targets, paths, shapes in tqdm(dataloader, desc="eval"):
        ims = ims.to(device_obj, non_blocking=True).float() / 255.0
        preds = model(ims)[0]
        preds = non_max_suppression(preds, conf_thres=conf_thres, iou_thres=nms_iou_thres, max_det=1000)

        for si, pred in enumerate(preds):
            shape0, ratio_pad = shapes[si]
            gt_boxes = dataset.get_ground_truth(path_to_index[paths[si]])
            num_targets += len(gt_boxes)

            if len(pred):
                pred = pred.clone()
                scale_boxes(ims[si].shape[1:], pred[:, :4], shape0, ratio_pad)
                pred_np = pred.cpu().numpy()
                order = np.argsort(-pred_np[:, 4])
                pred_np = pred_np[order]
            else:
                pred_np = np.zeros((0, 6), dtype=np.float32)

            image_tp = np.zeros((len(pred_np), len(iouv)), dtype=np.float32)
            for ti, iou_thres in enumerate(iouv):
                image_tp[:, ti] = match_predictions(pred_np[:, :4], gt_boxes, float(iou_thres))
            all_tp.append(image_tp)
            all_conf.append(pred_np[:, 4] if len(pred_np) else np.zeros(0, dtype=np.float32))

            if saved < save_samples:
                draw_sample(paths[si], gt_boxes, pred_np, save_dir / "sample_predictions" / f"{saved:04d}.jpg")
                saved += 1

    tp = np.concatenate(all_tp, axis=0) if all_tp else np.zeros((0, len(iouv)), dtype=np.float32)
    conf = np.concatenate(all_conf, axis=0) if all_conf else np.zeros(0, dtype=np.float32)

    ap = []
    pr_at_07 = None
    for ti, iou_thres in enumerate(iouv):
        precision, recall, ap_i = precision_recall_ap(tp[:, ti], conf, num_targets)
        ap.append(ap_i)
        if abs(float(iou_thres) - 0.7) < 1e-6:
            pr_at_07 = (precision, recall, ap_i)

    ap = np.asarray(ap, dtype=np.float32)
    map50 = float(ap[0])
    map90 = float(ap[np.argmin(np.abs(iouv - 0.9))])
    map5095 = float(ap.mean())

    if pr_at_07 is not None:
        precision_07, recall_07, ap_07 = pr_at_07
        plot_pr_curve(recall_07, precision_07, ap_07, save_dir / "PR_curve_iou0.7.png", 0.7)
        p_final = float(precision_07[-1]) if len(precision_07) else 0.0
        r_final = float(recall_07[-1]) if len(recall_07) else 0.0
    else:
        ap_07, p_final, r_final = 0.0, 0.0, 0.0

    summary = {
        "weights": str(weights),
        "data_root": str(data_root),
        "val_images": len(dataset),
        "val_targets": int(num_targets),
        "confidence_threshold": conf_thres,
        "nms_iou_threshold": nms_iou_thres,
        "mAP@0.5": map50,
        "mAP@0.9": map90,
        "mAP@0.5:0.95": map5095,
        "AP@0.7": float(ap_07),
        "precision_at_iou0.7_final": p_final,
        "recall_at_iou0.7_final": r_final,
        "ap_per_iou": {f"{x:.2f}": float(y) for x, y in zip(iouv, ap)},
    }
    with (save_dir / "metrics_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    return summary


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True, help="trained checkpoint path")
    parser.add_argument("--data-root", type=str, default="data/widerface")
    parser.add_argument("--imgsz", "--img", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--conf-thres", type=float, default=0.001)
    parser.add_argument("--nms-iou-thres", type=float, default=0.6)
    parser.add_argument("--device", default="")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", type=str, default="runs/eval")
    parser.add_argument("--name", type=str, default="widerface")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--save-samples", type=int, default=16)
    return parser.parse_args()


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    main(parse_opt())
