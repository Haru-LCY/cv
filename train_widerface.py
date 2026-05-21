"""Train YOLOv5 on WIDER FACE with the custom homework DataLoader."""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from datasets.widerface_dataset import WiderFaceDataset
from models.yolo import Model
from utils.general import LOGGER, colorstr, increment_path, init_seeds, intersect_dicts, one_cycle
from utils.loss import ComputeLoss
from utils.torch_utils import de_parallel, select_device, smart_optimizer

try:
    from ultralytics.utils.patches import torch_load
except Exception:  # pragma: no cover
    torch_load = torch.load


def save_loss_curve(results_csv: Path, save_path: Path):
    """Plot train box/object/class/total loss from results.csv."""
    rows = []
    with results_csv.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return

    epochs = [int(r["epoch"]) for r in rows]
    box = [float(r["train/box_loss"]) for r in rows]
    obj = [float(r["train/obj_loss"]) for r in rows]
    cls = [float(r["train/cls_loss"]) for r in rows]
    total = [float(r["train/total_loss"]) for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5), tight_layout=True)
    ax.plot(epochs, box, label="box")
    ax.plot(epochs, obj, label="obj")
    ax.plot(epochs, cls, label="cls")
    ax.plot(epochs, total, label="total", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def resolve_device(device_arg: str, batch_size: int) -> torch.device:
    """Prefer Apple MPS for this homework and fail instead of silently using CPU when requested."""
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


def load_model(cfg: str, weights: str, device: torch.device, hyp: dict, nc: int = 1) -> Model:
    """Create a YOLOv5 model and optionally initialize matching layers from a checkpoint."""
    model = Model(cfg, ch=3, nc=nc, anchors=hyp.get("anchors")).to(device)
    if weights:
        ckpt = torch_load(weights, map_location="cpu")
        state_dict = (ckpt.get("ema") or ckpt["model"]).float().state_dict()
        state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=["anchor"])
        model.load_state_dict(state_dict, strict=False)
        LOGGER.info(f"Transferred {len(state_dict)} checkpoint tensors from {weights}")
    return model


def train(opt):
    save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)
    weights_dir = save_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    results_csv = save_dir / "results.csv"

    init_seeds(opt.seed, deterministic=True)
    device = resolve_device(opt.device, batch_size=opt.batch_size)
    with open(opt.hyp, errors="ignore") as f:
        hyp = yaml.safe_load(f)

    model = load_model(opt.cfg, opt.weights, device, hyp, nc=1)
    stride = int(model.stride.max())
    gs = max(stride, 32)
    imgsz = int(np.ceil(opt.imgsz / gs) * gs)

    train_dataset = WiderFaceDataset(
        root=opt.data_root,
        split="train",
        img_size=imgsz,
        stride=stride,
        augment=True,
        fliplr=hyp.get("fliplr", 0.5),
        scaleup=True,
        max_samples=opt.max_train_samples,
    )
    val_dataset = WiderFaceDataset(
        root=opt.data_root,
        split="val",
        img_size=imgsz,
        stride=stride,
        augment=False,
        scaleup=False,
        max_samples=opt.max_val_samples,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.workers,
        collate_fn=WiderFaceDataset.collate_fn,
        pin_memory=device.type != "cpu",
    )
    nl = de_parallel(model).model[-1].nl
    hyp["box"] *= 3 / nl
    hyp["cls"] *= 1 / 80 * 3 / nl
    hyp["obj"] *= (imgsz / 640) ** 2 * 3 / nl
    hyp["label_smoothing"] = opt.label_smoothing
    model.nc = 1
    model.hyp = hyp
    model.names = {0: "face"}
    model.class_weights = torch.ones(1, device=device)

    optimizer = smart_optimizer(model, opt.optimizer, hyp["lr0"], hyp["momentum"], hyp["weight_decay"])
    lf = one_cycle(1, hyp["lrf"], opt.epochs)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    compute_loss = ComputeLoss(model)
    amp_enabled = device.type == "cuda" and not opt.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    LOGGER.info(
        f"Training WIDER FACE: {len(train_dataset)} train images, {len(val_dataset)} val images\n"
        f"Image size {imgsz}, batch size {opt.batch_size}\n"
        f"Logging results to {colorstr('bold', save_dir)}"
    )

    with results_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train/box_loss", "train/obj_loss", "train/cls_loss", "train/total_loss", "lr"])

    best_loss = float("inf")
    t0 = time.time()
    for epoch in range(opt.epochs):
        model.train()
        mloss = torch.zeros(3, device=device)
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{opt.epochs - 1}")
        optimizer.zero_grad()

        for i, (imgs, targets, _paths, _shapes) in enumerate(pbar):
            imgs = imgs.to(device, non_blocking=True).float() / 255.0
            targets = targets.to(device)

            with torch.amp.autocast("cuda", enabled=amp_enabled):
                pred = model(imgs)
                loss, loss_items = compute_loss(pred, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            mloss = (mloss * i + loss_items) / (i + 1)
            pbar.set_postfix(box=f"{mloss[0]:.4f}", obj=f"{mloss[1]:.4f}", cls=f"{mloss[2]:.4f}")

        scheduler.step()
        total_loss = float(mloss.sum().item())
        lr = optimizer.param_groups[0]["lr"]
        with results_csv.open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, float(mloss[0]), float(mloss[1]), float(mloss[2]), total_loss, lr])
        save_loss_curve(results_csv, save_dir / "train_loss_curve.png")

        ckpt = {
            "epoch": epoch,
            "model": deepcopy(de_parallel(model)).half(),
            "optimizer": optimizer.state_dict(),
            "hyp": hyp,
            "opt": vars(opt),
            "names": model.names,
            "date": datetime.now().isoformat(),
        }
        torch.save(ckpt, weights_dir / "last.pt")
        if total_loss < best_loss:
            best_loss = total_loss
            torch.save(ckpt, weights_dir / "best.pt")
        del ckpt

        LOGGER.info(
            f"Epoch {epoch}: box={mloss[0]:.5f}, obj={mloss[1]:.5f}, cls={mloss[2]:.5f}, total={total_loss:.5f}"
        )

    LOGGER.info(f"Training completed in {(time.time() - t0) / 3600:.3f} hours. Results saved to {save_dir}")
    return save_dir


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data/widerface")
    parser.add_argument("--cfg", type=str, default="models/yolov5n.yaml")
    parser.add_argument("--weights", type=str, default="", help="optional initial weights")
    parser.add_argument("--hyp", type=str, default="data/hyps/hyp.scratch-low.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--imgsz", "--img", type=int, default=640)
    parser.add_argument("--device", default="")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--optimizer", type=str, choices=["SGD", "Adam", "AdamW"], default="SGD")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--project", type=str, default="runs/train")
    parser.add_argument("--name", type=str, default="widerface_custom")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=None, help="debug subset size")
    parser.add_argument("--max-val-samples", type=int, default=None, help="reserved for matching eval/debug configs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main(opt):
    train(opt)


if __name__ == "__main__":
    main(parse_opt())
