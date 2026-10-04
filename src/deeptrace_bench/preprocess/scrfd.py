"""SCRFD face detection with onnxruntime: boxes and five landmarks per face.

Ported from GenD's ``src/retinaface.py`` (MIT, Copyright (c) 2025 Andy, at commit 387a422),
which is itself insightface's SCRFD inference code (MIT). The model file (``det_10g.onnx``,
the ``scrfd_10g`` tool) is insightface's and licensed for non-commercial research only.

Changes from upstream, none of which alter the numbers:

- Frames come in as RGB (the harness's convention) and go to ``blobFromImage`` without
  ``swapRB``; upstream passes BGR with ``swapRB=True``. Both give the network the same RGB blob.
- Always ``CPUExecutionProvider`` with an explicit thread count, so detections don't depend on
  which accelerator a machine has. (CoreML and CUDA providers can shift scores slightly.)
- No ``wget`` from a mirror: the model path comes from the pinned weights folder.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np


def distance2bbox(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    """Decode (left, top, right, bottom) distances from anchor centres into boxes."""
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    """Decode (dx, dy) offsets from anchor centres into keypoints, flattened per anchor."""
    preds = []
    for i in range(0, distance.shape[1], 2):
        preds.append(points[:, i % 2] + distance[:, i])
        preds.append(points[:, i % 2 + 1] + distance[:, i + 1])
    return np.stack(preds, axis=-1)


def nms(dets: np.ndarray, threshold: float) -> list[int]:
    """Greedy non-maximum suppression over ``[x1, y1, x2, y2, score]`` rows."""
    x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1 + 1) * np.maximum(0.0, yy2 - yy1 + 1)
        overlap = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(overlap <= threshold)[0] + 1]
    return keep


class SCRFD:
    """An SCRFD detector on CPU.

    Args:
        model_path: the ``.onnx`` file.
        input_size: side of the square, letterboxed network input.
        det_threshold: minimum face score.
        nms_threshold: IoU above which overlapping boxes are merged.
        threads: onnxruntime intra-op threads (default: up to 4).
    """

    def __init__(
        self,
        model_path: Path,
        input_size: int = 640,
        det_threshold: float = 0.4,
        nms_threshold: float = 0.4,
        threads: int | None = None,
    ) -> None:
        import onnxruntime

        # onnxruntime's telemetry uploads usage events from a background thread, which races
        # interpreter shutdown on macOS and aborts the process after scoring has finished.
        # The detector has no reason to send anything anywhere.
        onnxruntime.disable_telemetry_events()
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = threads or min(4, os.cpu_count() or 1)
        options.inter_op_num_threads = 1
        self.session = onnxruntime.InferenceSession(
            str(model_path), options, providers=["CPUExecutionProvider"]
        )
        self.input_size = input_size
        self.det_threshold = det_threshold
        self.nms_threshold = nms_threshold
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        self._centers: dict[tuple[int, int, int], np.ndarray] = {}
        n_out = len(self.output_names)
        # 6 or 9 outputs: 3 strides with 2 anchors each; 10 or 15: 5 strides with 1 anchor.
        # 9 and 15 also carry keypoints.
        if n_out in (6, 9):
            self.fmc, self.strides, self.num_anchors = 3, [8, 16, 32], 2
        elif n_out in (10, 15):
            self.fmc, self.strides, self.num_anchors = 5, [8, 16, 32, 64, 128], 1
        else:
            raise ValueError(f"{model_path}: unexpected SCRFD output count {n_out}")
        self.use_kps = n_out in (9, 15)
        if not self.use_kps:
            raise ValueError(f"{model_path}: no landmark outputs, which alignment needs")

    def _anchor_centers(self, height: int, width: int, stride: int) -> np.ndarray:
        key = (height, width, stride)
        if key not in self._centers:
            centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
            centers = (centers * stride).reshape((-1, 2))
            if self.num_anchors > 1:
                centers = np.stack([centers] * self.num_anchors, axis=1).reshape((-1, 2))
            self._centers[key] = centers
        return self._centers[key]

    def detect(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Find faces in an RGB uint8 image.

        Returns:
            ``(dets, kps)``: ``[n, 5]`` boxes as ``x1, y1, x2, y2, score`` and ``[n, 5, 2]``
            landmarks (left eye, right eye, nose, left and right mouth corner), in the
            image's pixel coordinates, best first. Empty arrays when no face is found.
        """
        size = self.input_size
        im_ratio = rgb.shape[0] / rgb.shape[1]
        if im_ratio > 1.0:
            new_h, new_w = size, int(size / im_ratio)
        else:
            new_w, new_h = size, int(size * im_ratio)
        det_scale = new_h / rgb.shape[0]
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        canvas[:new_h, :new_w] = cv2.resize(rgb, (new_w, new_h))
        blob = cv2.dnn.blobFromImage(
            canvas, 1.0 / 128.0, (size, size), (127.5, 127.5, 127.5), swapRB=False
        )
        outs = self.session.run(self.output_names, {self.input_name: blob})

        scores_all, boxes_all, kps_all = [], [], []
        for idx, stride in enumerate(self.strides):
            scores = outs[idx]
            boxes = outs[idx + self.fmc] * stride
            kps = outs[idx + 2 * self.fmc] * stride
            centers = self._anchor_centers(size // stride, size // stride, stride)
            keep = np.where(scores >= self.det_threshold)[0]
            scores_all.append(scores[keep])
            boxes_all.append(distance2bbox(centers, boxes)[keep])
            kps_all.append(distance2kps(centers, kps).reshape((-1, 5, 2))[keep])

        scores = np.vstack(scores_all).ravel()
        if scores.size == 0:
            return np.zeros((0, 5), np.float32), np.zeros((0, 5, 2), np.float32)
        order = scores.argsort()[::-1]
        boxes = np.vstack(boxes_all) / det_scale
        kps = np.vstack(kps_all) / det_scale
        dets = np.hstack((boxes, scores[:, None])).astype(np.float32)[order]
        keep = nms(dets, self.nms_threshold)
        return dets[keep], kps[order][keep].astype(np.float32)
