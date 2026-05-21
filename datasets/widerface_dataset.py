"""WIDER FACE dataset loader used by the homework training/evaluation scripts."""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.augmentations import letterbox

IMG_FORMATS = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class WiderFaceRecord:
    """One WIDER FACE image and its valid face boxes in original pixel coordinates."""

    image_path: Path
    boxes_xyxy: np.ndarray


def xyxy_to_xywhn(boxes: np.ndarray, width: int, height: int, eps: float = 1e-3) -> np.ndarray:
    """Convert absolute xyxy boxes to normalized YOLO cxcywh boxes."""
    out = boxes.copy().astype(np.float32)
    out[:, 0] = ((boxes[:, 0] + boxes[:, 2]) / 2) / width
    out[:, 1] = ((boxes[:, 1] + boxes[:, 3]) / 2) / height
    out[:, 2] = (boxes[:, 2] - boxes[:, 0]) / width
    out[:, 3] = (boxes[:, 3] - boxes[:, 1]) / height
    return np.clip(out, eps, 1.0 - eps)


def parse_widerface_annotations(root: str | Path, split: str, skip_invalid: bool = True) -> list[WiderFaceRecord]:
    """Parse the official WIDER FACE bbox txt file.

    WIDER FACE has a small quirk: images with zero boxes still have one dummy box line in the txt file. The parser
    consumes that line when `num_boxes == 0`.
    """
    root = Path(root)
    split = split.lower()
    if split not in {"train", "val"}:
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")

    annotation_file = root / "wider_face_split" / f"wider_face_{split}_bbx_gt.txt"
    image_root = root / f"WIDER_{split}" / "images"
    if not annotation_file.is_file():
        raise FileNotFoundError(annotation_file)
    if not image_root.is_dir():
        raise FileNotFoundError(image_root)

    records: list[WiderFaceRecord] = []
    with annotation_file.open() as f:
        while True:
            rel_path = f.readline()
            if not rel_path:
                break
            rel_path = rel_path.strip()
            if not rel_path:
                continue

            num_boxes = int(f.readline().strip())
            rows_to_read = num_boxes if num_boxes > 0 else 1
            rows = [f.readline().strip() for _ in range(rows_to_read)]
            boxes = []
            for row in rows[:num_boxes]:
                vals = row.split()
                if len(vals) < 10:
                    continue
                x, y, w, h = map(float, vals[:4])
                invalid = int(vals[7])
                if skip_invalid and invalid:
                    continue
                if w <= 0 or h <= 0:
                    continue
                boxes.append([x, y, x + w, y + h])

            image_path = image_root / rel_path
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            records.append(
                WiderFaceRecord(
                    image_path=image_path,
                    boxes_xyxy=np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
                )
            )
    return records


class WiderFaceDataset(Dataset):
    """Custom WIDER FACE dataset that returns YOLOv5-compatible image/target batches."""

    def __init__(
        self,
        root: str | Path = "data/widerface",
        split: str = "train",
        img_size: int = 640,
        stride: int = 32,
        augment: bool = False,
        fliplr: float = 0.0,
        scaleup: bool = True,
        max_samples: int | None = None,
        skip_invalid: bool = True,
    ):
        self.root = Path(root)
        self.split = split.lower()
        self.img_size = int(img_size)
        self.stride = int(stride)
        self.augment = bool(augment)
        self.fliplr = float(fliplr)
        self.scaleup = bool(scaleup)
        self.records = parse_widerface_annotations(self.root, self.split, skip_invalid=skip_invalid)
        if max_samples is not None:
            self.records = self.records[: int(max_samples)]
        self.im_files = [str(r.image_path) for r in self.records]
        self.n = len(self.records)
        self.labels = [self._labels_for_record(r) for r in self.records]

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int):
        record = self.records[index]
        im = cv2.imread(str(record.image_path))
        if im is None:
            raise FileNotFoundError(record.image_path)

        shape0 = im.shape[:2]  # h, w
        boxes = record.boxes_xyxy.copy()
        if boxes.size:
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape0[1])
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape0[0])

        im, ratio, pad = letterbox(
            im,
            new_shape=self.img_size,
            auto=False,
            scaleup=self.scaleup,
            stride=self.stride,
        )
        shape = im.shape[:2]  # h, w after letterbox

        labels = self._boxes_to_letterboxed_labels(boxes, ratio, pad, shape)

        if self.augment and self.fliplr > 0 and np.random.random() < self.fliplr:
            im = np.fliplr(im)
            if len(labels):
                labels[:, 1] = 1.0 - labels[:, 1]

        im = im[:, :, ::-1].transpose(2, 0, 1)  # BGR HWC to RGB CHW
        im = np.ascontiguousarray(im)

        targets = np.zeros((len(labels), 6), dtype=np.float32)
        if len(labels):
            targets[:, 1:] = labels

        shapes = (shape0, (ratio, pad))
        return torch.from_numpy(im), torch.from_numpy(targets), str(record.image_path), shapes

    def get_ground_truth(self, index: int) -> np.ndarray:
        """Return valid original-scale xyxy ground-truth boxes for evaluation."""
        return self.records[index].boxes_xyxy.copy()

    def _labels_for_record(self, record: WiderFaceRecord) -> np.ndarray:
        """Build lightweight labels metadata for class weighting.

        The training/evaluation code computes exact coordinates in `__getitem__`.
        For this single-class assignment, the metadata only needs class ids.
        """
        labels = np.zeros((len(record.boxes_xyxy), 5), dtype=np.float32)
        return labels

    @staticmethod
    def _boxes_to_letterboxed_labels(
        boxes: np.ndarray,
        ratio: tuple[float, float],
        pad: tuple[float, float],
        shape: tuple[int, int],
    ) -> np.ndarray:
        if not len(boxes):
            return np.zeros((0, 5), dtype=np.float32)

        boxes = boxes.astype(np.float32)
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * ratio[0] + pad[0]
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * ratio[1] + pad[1]
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])

        keep = (boxes[:, 2] - boxes[:, 0] > 1) & (boxes[:, 3] - boxes[:, 1] > 1)
        boxes = boxes[keep]
        labels = np.zeros((len(boxes), 5), dtype=np.float32)
        if len(boxes):
            labels[:, 1:] = xyxy_to_xywhn(boxes, shape[1], shape[0])
        return labels

    @staticmethod
    def collate_fn(batch: Iterable):
        ims, targets, paths, shapes = zip(*batch)
        for i, target in enumerate(targets):
            target[:, 0] = i
        return torch.stack(ims, 0), torch.cat(targets, 0), paths, shapes
