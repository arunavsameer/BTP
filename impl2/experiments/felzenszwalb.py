"""Felzenszwalb and Huttenlocher graph segmentation (IJCV 2004).

Each region becomes one box. OpenCV's ximgproc graph segmentation is used
when contrib is installed. Otherwise a small 4-neighbor version of the same
algorithm runs here. It does not fall back to color flood-fill.

The merge test is the paper's: an edge joins two regions only when its
color distance is at most min(Int(a) + k/|a|, Int(b) + k/|b|). Int is the
heaviest edge already inside the region.
"""

from __future__ import annotations

import cv2
import numpy as np

from config import FelzenszwalbConfig
from regions import Region


def ximgproc_available() -> bool:
    module = getattr(cv2, "ximgproc", None)
    if module is None or not hasattr(module, "segmentation"):
        return False
    return hasattr(module.segmentation, "createGraphSegmentation")


def segmentation_backend() -> str:
    return "ximgproc" if ximgproc_available() else "python"


def working_image(frame: np.ndarray, process_width: int) -> tuple[np.ndarray, float, float]:
    """Downscale. Returns the small BGR image and the scale back to full size."""
    height, width = frame.shape[:2]
    if process_width <= 0 or process_width >= width:
        return np.ascontiguousarray(frame), 1.0, 1.0
    small_w = int(process_width)
    small_h = max(1, int(round(height * small_w / width)))
    small = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(small), width / float(small_w), height / float(small_h)


def segment_labels(
    image: np.ndarray,
    k: float,
    min_size: int,
    sigma: float,
    backend: str = "auto",
) -> np.ndarray:
    """Integer label map, one id per region. `backend` is auto, ximgproc, or python."""
    image = np.ascontiguousarray(image)
    if backend == "auto":
        backend = segmentation_backend()
    if backend == "ximgproc":
        if not ximgproc_available():
            raise RuntimeError("cv2.ximgproc.segmentation is not installed")
        segmenter = cv2.ximgproc.segmentation.createGraphSegmentation(
            float(sigma), float(k), int(min_size)
        )
        labels = segmenter.processImage(image)
        if labels.ndim == 3:
            labels = labels[:, :, 0]
        return np.ascontiguousarray(labels)

    if backend != "python":
        raise ValueError("backend must be auto, ximgproc, or python")
    blurred = image
    if sigma > 0:
        blurred = cv2.GaussianBlur(image, (0, 0), float(sigma))
    return _segment_python(blurred, float(k), int(min_size))


def _grid_edges(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """4-neighbor edges. Weight is Euclidean distance in RGB."""
    height, width = image.shape[:2]
    pixels = image.astype(np.float32)
    index = np.arange(height * width, dtype=np.int32).reshape(height, width)

    horizontal = pixels[:, 1:, :] - pixels[:, :-1, :]
    vertical = pixels[1:, :, :] - pixels[:-1, :, :]
    horizontal_w = np.sqrt((horizontal * horizontal).sum(axis=2)).ravel()
    vertical_w = np.sqrt((vertical * vertical).sum(axis=2)).ravel()

    weights = np.concatenate([horizontal_w, vertical_w])
    starts = np.concatenate([index[:, :-1].ravel(), index[:-1, :].ravel()])
    ends = np.concatenate([index[:, 1:].ravel(), index[1:, :].ravel()])
    return weights, starts, ends


def _segment_python(image: np.ndarray, k: float, min_size: int) -> np.ndarray:
    height, width = image.shape[:2]
    count = height * width
    weights, starts, ends = _grid_edges(image)
    order = np.argsort(weights, kind="mergesort")
    weights = weights[order]
    starts = starts[order]
    ends = ends[order]

    parent = list(range(count))
    size = [1] * count
    # Int starts at 0 and each region is one pixel, so the threshold is k/1.
    threshold = [float(k)] * count

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def join(root_a: int, root_b: int, weight: float) -> None:
        if size[root_a] < size[root_b]:
            root_a, root_b = root_b, root_a
        parent[root_b] = root_a
        size[root_a] += size[root_b]
        # The joining edge is the heaviest in the new region, because edges
        # are visited lightest first. Threshold becomes Int + k/|C|.
        threshold[root_a] = float(weight) + float(k) / size[root_a]

    edge_count = len(weights)
    for index in range(edge_count):
        weight = float(weights[index])
        root_a = find(int(starts[index]))
        root_b = find(int(ends[index]))
        if root_a == root_b:
            continue
        if weight <= threshold[root_a] and weight <= threshold[root_b]:
            join(root_a, root_b, weight)

    # Enforce the minimum size. A tiny region joins its nearest neighbor.
    if min_size > 1:
        for index in range(edge_count):
            root_a = find(int(starts[index]))
            root_b = find(int(ends[index]))
            if root_a == root_b:
                continue
            if size[root_a] < min_size or size[root_b] < min_size:
                join(root_a, root_b, float(weights[index]))

    flat = np.empty(count, dtype=np.int32)
    root_label: dict[int, int] = {}
    next_label = 0
    for node in range(count):
        root = find(node)
        label = root_label.get(root)
        if label is None:
            label = next_label
            root_label[root] = label
            next_label += 1
        flat[node] = label
    return flat.reshape(height, width)


def boxes_from_labels(
    labels: np.ndarray,
    scale_x: float,
    scale_y: float,
    full_w: int,
    full_h: int,
    cfg: FelzenszwalbConfig,
    reject_mask: np.ndarray | None = None,
    reject_fraction: float = 1.0,
    name: str = "seg",
) -> list[Region]:
    """Turn a label map into boxes. Optionally drop regions that match a mask."""
    height, width = labels.shape[:2]
    frame_area = float(full_w * full_h)
    flat = np.ascontiguousarray(labels).reshape(-1)
    order = np.argsort(flat, kind="mergesort")
    sorted_labels = flat[order]
    if sorted_labels.size == 0:
        return []

    cuts = np.flatnonzero(sorted_labels[1:] != sorted_labels[:-1]) + 1
    starts = np.concatenate(([0], cuts))
    ends = np.concatenate((cuts, [sorted_labels.size]))
    xs = order % width
    ys = order // width
    mask_flat = None
    if reject_mask is not None:
        mask_flat = np.ascontiguousarray(reject_mask).reshape(-1).astype(bool)

    regions: list[Region] = []
    for start, end in zip(starts.tolist(), ends.tolist()):
        pixels = end - start
        area = pixels * scale_x * scale_y
        if area < cfg.min_area or area > cfg.max_area_frac * frame_area:
            continue
        if mask_flat is not None and float(mask_flat[order[start:end]].mean()) >= reject_fraction:
            continue

        x0 = int(xs[start:end].min())
        x1 = int(xs[start:end].max()) + 1
        y0 = int(ys[start:end].min())
        y1 = int(ys[start:end].max()) + 1
        fill = pixels / float((x1 - x0) * (y1 - y0))
        if fill < cfg.min_fill:
            continue
        regions.append(
            Region(
                x1=x0 * scale_x,
                y1=y0 * scale_y,
                x2=min(float(full_w), x1 * scale_x),
                y2=min(float(full_h), y1 * scale_y),
                area=area,
                name=name,
            )
        )

    regions.sort(key=lambda region: region.area, reverse=True)
    return regions[: cfg.max_boxes]


def find_regions(frame: np.ndarray, cfg: FelzenszwalbConfig) -> list[Region]:
    small, scale_x, scale_y = working_image(frame, cfg.process_width)
    labels = segment_labels(small, cfg.k, cfg.min_size, cfg.sigma)
    if labels.shape[:2] != small.shape[:2]:
        raise RuntimeError(f"segmentation returned {labels.shape}, expected {small.shape[:2]}")
    return boxes_from_labels(labels, scale_x, scale_y, frame.shape[1], frame.shape[0], cfg, name="seg")
