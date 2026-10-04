"""Face detection and cropping for video and image models.

The shared pipeline, one item at a time:

1. **Frames.** A video gives ``frames_per_clip`` frames spread evenly over its length (read in
   order with ``grab``, so a wrong frame count in the header only shortens the list); an image
   is a one-frame clip. Frames are RGB from here on.
   Datasets of ready-made face crops (a face filling the whole image) defeat the detector,
   which needs some context around a face, so an image with no face is tried once more with
   a black border (``image_pad_retry`` of each side). Coordinates are always stored in the
   original image, so cropping is the same either way; the cache records the padding used.
2. **Detect once per dataset.** One detector (the ``scrfd_10g`` tool, the model GenD's own
   ``detector.py`` runs) finds faces in each frame, and the largest face's box and five
   landmarks are cached as parquet parts under ``faces/<dataset>/<detector>__<hash>/``. The
   hash covers the detector's weights and settings and the frame sampling, so a changed
   setting never reuses stale boxes. Detection is the slow step; cropping is cheap.
3. **Crop per model.** Models were trained on different crops (GenD aligns five landmarks at
   scale 1.3 and keeps the native size, SBI cuts 380 px boxes, DeepfakeBench uses 256 px), so
   each model's config carries a ``CropSpec`` and crops are cut from the cached detections.
4. **Failures are data.** An unreadable file raises ``PreprocessError("unreadable")``; too few
   frames with a face raises ``PreprocessError("no_face")``. Both become the item's status,
   counted per group, never dropped: face detectors miss darker faces more often, so dropping
   failures would bias the very gaps we measure.

Native mode (a model's own preprocessing, for reproducing its paper) is not built yet.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
import pandas as pd
from pydantic import BaseModel

from ..paths import dataset_dir, faces_dir, weights_dir
from . import PreprocessError

log = logging.getLogger(__name__)

Detect = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]

# GenD's alignment template: where the eyes, nose tip and mouth corners land, as fractions of
# the crop before the margin is added (detector.py at commit 387a422).
_TEMPLATE = np.array(
    [[0.34, 0.46], [0.66, 0.46], [0.5, 0.64], [0.37, 0.82], [0.63, 0.82]], dtype=np.float32
)


@dataclass(frozen=True)
class FaceDetection:
    """One face found in one frame."""

    frame_index: int
    box: tuple[float, float, float, float]
    landmarks: np.ndarray
    score: float


class CropSpec(BaseModel):
    """How a model wants its faces cut."""

    size: int | None = None  # None: native size, from the landmark spread times the margin
    margin: float = 1.3
    align: Literal["none", "five_point"] = "none"
    normalize: Literal["imagenet", "clip", "none"] = "imagenet"


def sample_frame_indices(n_frames: int, k: int) -> list[int]:
    """Pick ``k`` frame indices spread evenly over a video of ``n_frames`` frames."""
    if n_frames <= 0:
        return []
    if n_frames <= k:
        return list(range(n_frames))
    return [int(i) for i in np.linspace(0, n_frames - 1, k).round()]


# --- reading ---------------------------------------------------------------------------------


def read_image(path: Path) -> np.ndarray:
    """Read an image as RGB uint8.

    Raises:
        PreprocessError: ``unreadable`` if it can't be decoded.
    """
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise PreprocessError("unreadable", str(path))
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def read_video_frames(path: Path, k: int) -> tuple[list[tuple[int, np.ndarray]], int]:
    """Read ``k`` evenly spread frames of a video as RGB uint8.

    Frames are decoded in order with ``grab`` and only the wanted ones converted, which is
    robust to formats that seek badly. OpenCV applies the rotation stored in phone videos.
    This is the one place video is decoded, so the backend can be swapped at parity time.

    Returns:
        ``([(frame_index, rgb), ...], frames_in_header)``.

    Raises:
        PreprocessError: ``unreadable`` if the file can't be opened or yields no frame.
    """
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise PreprocessError("unreadable", f"{path}: cannot open")
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n_frames <= 0:
            n_frames = _count_frames(cap)
            cap.release()
            cap = cv2.VideoCapture(str(path))
        wanted = sample_frame_indices(n_frames, k)
        want = set(wanted)
        frames: list[tuple[int, np.ndarray]] = []
        index = 0
        while wanted and index <= wanted[-1] and cap.grab():
            if index in want:
                ok, bgr = cap.retrieve()
                if ok:
                    frames.append((index, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
            index += 1
    finally:
        cap.release()
    if not frames:
        raise PreprocessError("unreadable", f"{path}: no frames decoded")
    return frames, n_frames


def _count_frames(cap: cv2.VideoCapture) -> int:
    count = 0
    while cap.grab():
        count += 1
    return count


# --- choosing and cropping -------------------------------------------------------------------


def largest_face(dets: np.ndarray) -> int:
    """Index of the detection with the largest box (``dets`` rows: x1, y1, x2, y2, score)."""
    areas = (dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1])
    return int(np.argmax(areas))


def align_face(
    image: np.ndarray, landmarks: np.ndarray, size: int | None = None, margin: float = 1.3
) -> np.ndarray:
    """Warp a face so its five landmarks land on GenD's template, with ``margin`` context.

    A port of ``align_face`` from GenD's ``detector.py`` (MIT): a similarity transform fitted
    with LMEDS, bilinear warping. With ``size=None`` the crop keeps the face's native
    resolution: the mean ratio of landmark distances to template distances, times the margin.
    """
    if size is None:
        pairs = np.triu_indices(5, k=1)
        face = np.linalg.norm(landmarks[:, None] - landmarks[None, :], axis=-1)[pairs]
        template = np.linalg.norm(_TEMPLATE[:, None] - _TEMPLATE[None, :], axis=-1)[pairs]
        size = int(np.round(np.mean(face / template) * margin))
    size = max(size, 1)
    pad = size * (margin - 1) / 2.0
    dst = (_TEMPLATE * size + pad) * (size / (size + 2 * pad))
    matrix = cv2.estimateAffinePartial2D(
        landmarks.astype(np.float32), dst.astype(np.float32), method=cv2.LMEDS
    )[0]
    if matrix is None:
        raise PreprocessError("no_face", "landmarks are degenerate")
    return cv2.warpAffine(image, matrix, (size, size), flags=cv2.INTER_LINEAR)


def box_crop(image: np.ndarray, box: np.ndarray, size: int | None, margin: float) -> np.ndarray:
    """A square crop around a box, enlarged by ``margin`` and clipped to the image."""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = max(x2 - x1, y2 - y1) * margin / 2
    h, w = image.shape[:2]
    left, top = int(max(cx - half, 0)), int(max(cy - half, 0))
    right, bottom = int(min(cx + half, w)), int(min(cy + half, h))
    crop = image[top:bottom, left:right]
    if crop.size == 0:
        raise PreprocessError("no_face", "box lies outside the frame")
    return crop if size is None else cv2.resize(crop, (size, size))


def crop_face(image: np.ndarray, box: np.ndarray, kps: np.ndarray, spec: CropSpec) -> np.ndarray:
    """Cut one face from a frame per a model's crop spec."""
    if spec.align == "five_point":
        return align_face(image, kps, spec.size, spec.margin)
    return box_crop(image, box, spec.size, spec.margin)


# --- detection cache -------------------------------------------------------------------------


def cache_key(detector_id: str, settings: dict[str, Any]) -> str:
    """``<detector>__<8 hex>``, the hex over everything that changes the detections."""
    blob = json.dumps(settings, sort_keys=True, separators=(",", ":"))
    return f"{detector_id}__{hashlib.sha256(blob.encode()).hexdigest()[:8]}"


class FaceCache:
    """Per-frame detections of one dataset, as parquet parts in one folder.

    One row per sampled frame: ``item_id``, ``frame_index``, ``n_frames`` (frames in the
    file), ``pad`` (border added before detection, as a fraction of each side), ``det_score``,
    ``box`` (4 floats) and ``kps`` (10 floats), the last three null when the frame has no
    face. Parts are written like score parts, so a killed job loses at most
    one buffer and shards never overwrite each other.
    """

    def __init__(
        self,
        directory: Path,
        settings: dict[str, Any],
        part_prefix: str = "part",
        flush_every: int = 64,
    ) -> None:
        self.directory = directory
        self.part_prefix = part_prefix
        self.flush_every = flush_every
        directory.mkdir(parents=True, exist_ok=True)
        spec = directory / "spec.json"
        if not spec.exists():
            spec.write_text(json.dumps(settings, indent=2, sort_keys=True) + "\n")
        self._rows: dict[str, list[dict[str, Any]]] = {}
        for part in sorted(directory.glob("part-*.parquet")):
            for record in pd.read_parquet(part).to_dict("records"):
                self._rows.setdefault(record["item_id"], []).append(record)
        self._pending: list[dict[str, Any]] = []
        self._next = len(list(directory.glob(f"{part_prefix}-*.parquet")))

    def get(self, item_id: str) -> list[dict[str, Any]] | None:
        """The cached frames of an item, or None if it hasn't been detected yet."""
        return self._rows.get(item_id)

    def put(self, item_id: str, frames: list[dict[str, Any]]) -> None:
        """Cache an item's frames (each a dict with the columns above, minus ``item_id``)."""
        rows = [{"item_id": item_id, **f} for f in frames]
        self._rows[item_id] = rows
        self._pending.extend(rows)
        if len({r["item_id"] for r in self._pending}) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        """Write pending rows as a new part."""
        if not self._pending:
            return
        path = self.directory / f"{self.part_prefix}-{self._next:05d}.parquet"
        tmp = path.with_name(path.name + ".tmp")
        pd.DataFrame(self._pending).to_parquet(tmp, index=False)
        tmp.rename(path)
        self._next += 1
        self._pending = []


# --- the loader the scorer calls -------------------------------------------------------------


class FaceLoader:
    """Turns a manifest row (video or image) into the face crops a model scores.

    Args:
        crop: the model's crop spec.
        make_detector: builds the detector on first use (so a fully cached run never loads it).
        detector_id: the detector tool's id, for the cache key.
        settings: everything that changes detections (weights hash, thresholds, sampling).
        frames_per_clip: frames sampled per video.
        min_face_frames: a video needs at least this many frames with a face.
        image_min_face_frames: the same for an image (a one-frame clip).
        image_pad_retry: border to add when an image shows no face, as a fraction of each
            side; 0 disables the retry.
        save_crops_to: if set, write up to ``save_crops`` crops per item there as PNG, under
            the item's id, for people to look at.
        part_prefix: names this process's cache parts (one per shard).
    """

    def __init__(
        self,
        crop: CropSpec,
        make_detector: Callable[[], Detect],
        detector_id: str,
        settings: dict[str, Any],
        frames_per_clip: int = 32,
        min_face_frames: int = 8,
        image_min_face_frames: int = 1,
        image_pad_retry: float = 0.0,
        save_crops_to: Path | None = None,
        save_crops: int = 0,
        part_prefix: str = "part",
    ) -> None:
        self.crop = crop
        self.make_detector = make_detector
        self.detector_id = detector_id
        self.settings = settings
        self.key = cache_key(detector_id, settings)
        self.frames_per_clip = frames_per_clip
        self.min_face_frames = min_face_frames
        self.image_min_face_frames = image_min_face_frames
        self.image_pad_retry = image_pad_retry
        self.save_crops_to = save_crops_to
        self.save_crops = save_crops
        self.part_prefix = part_prefix
        self._detector: Detect | None = None
        self._caches: dict[str, FaceCache] = {}

    @classmethod
    def from_registry(cls, registry, model, **kwargs: Any) -> FaceLoader:  # noqa: ANN001
        """A loader set up from the eval config, the face detector tool and the model's crop."""
        from ..fetch import locked_hashes
        from .scrfd import SCRFD

        video = registry.eval.video
        tool = registry.face_detector()
        model_path = weights_dir(tool.id) / tool.weights[0].name
        settings = {
            "detector": tool.id,
            "weights": locked_hashes(tool.id),
            "input_size": video["detector_input_size"],
            "det_threshold": video["det_threshold"],
            "nms_threshold": video["nms_threshold"],
            "face_choice": video["face_choice"],
            "frames_per_clip": video["frames_per_clip"],
            "frame_sampling": video["frame_sampling"],
            "image_pad_retry": video["image_pad_retry"],
        }
        if settings["face_choice"] != "largest" or settings["frame_sampling"] != "uniform":
            raise NotImplementedError("only largest-face choice and uniform sampling exist")

        def make() -> Detect:
            if not model_path.exists():
                raise FileNotFoundError(
                    f"{model_path} is missing; run scripts/setup_models.py {model.id}"
                )
            return SCRFD(
                model_path,
                input_size=video["detector_input_size"],
                det_threshold=video["det_threshold"],
                nms_threshold=video["nms_threshold"],
            ).detect

        return cls(
            crop=CropSpec(**model.input.get("crop", {})),
            make_detector=make,
            detector_id=tool.id,
            settings=settings,
            frames_per_clip=video["frames_per_clip"],
            min_face_frames=video["min_face_frames"],
            image_min_face_frames=video["image_min_face_frames"],
            image_pad_retry=video["image_pad_retry"],
            **kwargs,
        )

    def _cache(self, dataset: str) -> FaceCache:
        if dataset not in self._caches:
            directory = faces_dir(dataset, self.key)
            self._caches[dataset] = FaceCache(directory, self.settings, self.part_prefix)
        return self._caches[dataset]

    def _find(self, rgb: np.ndarray, is_image: bool) -> tuple[np.ndarray, np.ndarray, float]:
        """Detections in ``rgb``'s coordinates, and the padding that found them."""
        assert self._detector is not None
        dets, kps = self._detector(rgb)
        if len(dets) or not is_image or self.image_pad_retry <= 0:
            return dets, kps, 0.0
        h, w = rgb.shape[:2]
        py, px = int(round(h * self.image_pad_retry)), int(round(w * self.image_pad_retry))
        padded = cv2.copyMakeBorder(rgb, py, py, px, px, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        dets, kps = self._detector(padded)
        if len(dets):
            dets = dets.copy()
            dets[:, [0, 2]] -= px
            dets[:, [1, 3]] -= py
            kps = kps - np.array([px, py], dtype=kps.dtype)
        return dets, kps, self.image_pad_retry

    def _detect(
        self, frames: list[tuple[int, np.ndarray]], n_frames: int, is_image: bool
    ) -> list[dict]:
        if self._detector is None:
            self._detector = self.make_detector()
        records = []
        for index, rgb in frames:
            dets, kps, pad = self._find(rgb, is_image)
            record: dict[str, Any] = {"frame_index": index, "n_frames": n_frames, "pad": pad}
            if len(dets):
                j = largest_face(dets)
                record.update(
                    det_score=float(dets[j, 4]),
                    box=[float(v) for v in dets[j, :4]],
                    kps=[float(v) for v in kps[j].ravel()],
                )
            else:
                record.update(det_score=None, box=None, kps=None)
            records.append(record)
        return records

    def __call__(self, row: pd.Series) -> list[np.ndarray]:
        """The crops for one item, in frame order.

        Raises:
            PreprocessError: ``unreadable`` or ``no_face``.
        """
        path = dataset_dir(row["dataset"]) / row["rel_path"]
        is_image = row["modality"] == "image"
        if is_image:
            frames, n_frames = [(0, read_image(path))], 1
        else:
            frames, n_frames = read_video_frames(path, self.frames_per_clip)

        cache = self._cache(row["dataset"])
        records = cache.get(row["item_id"])
        if records is None:
            records = self._detect(frames, n_frames, is_image)
            cache.put(row["item_id"], records)

        images = dict(frames)
        crops: list[tuple[int, np.ndarray]] = []
        for record in records:
            image = images.get(record["frame_index"])
            if image is None or record["kps"] is None:
                continue
            box = np.asarray(record["box"], dtype=np.float32)
            kps = np.asarray(record["kps"], dtype=np.float32).reshape(5, 2)
            crops.append((record["frame_index"], crop_face(image, box, kps, self.crop)))

        need = self.image_min_face_frames if is_image else self.min_face_frames
        if len(crops) < need:
            raise PreprocessError("no_face", f"{len(crops)} of {len(records)} frames have a face")
        if self.save_crops_to is not None and self.save_crops > 0:
            self._save(row["item_id"], crops[: self.save_crops])
        return [crop for _, crop in crops]

    def _save(self, item_id: str, crops: list[tuple[int, np.ndarray]]) -> None:
        from ..fetch import ensure_inside

        assert self.save_crops_to is not None
        folder = ensure_inside(self.save_crops_to, self.save_crops_to / item_id)
        folder.mkdir(parents=True, exist_ok=True)
        for index, crop in crops:
            bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(folder / f"frame_{index:04d}.png"), bgr)

    def close(self) -> None:
        """Write any cached detections not yet on disk."""
        for cache in self._caches.values():
            cache.flush()
