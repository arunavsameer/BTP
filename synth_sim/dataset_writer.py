"""Per-episode dataset sink: frames, MP4, or both."""

from __future__ import annotations

import json
import shutil
import subprocess
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from config import SimulationConfig


def _f(x: Any) -> float:
    return float(x)


def _write_one(
    rgb_path: Optional[Path],
    depth_path: Optional[Path],
    seg_path: Optional[Path],
    json_path: Optional[Path],
    rgb: Optional[np.ndarray],
    depth: Optional[np.ndarray],
    seg: Optional[np.ndarray],
    annotation: Optional[Dict[str, Any]],
) -> None:
    if rgb_path is not None and rgb is not None:
        rgb_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb, mode="RGB").save(rgb_path, optimize=False)
    if depth_path is not None and depth is not None:
        depth_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(depth_path, np.ascontiguousarray(depth.astype(np.float32)))
    if seg_path is not None and seg is not None:
        seg_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(seg.astype(np.uint16), mode="I;16").save(seg_path)
    if json_path is not None and annotation is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with json_path.open("w", encoding="utf-8") as fh:
            json.dump(annotation, fh, indent=2)


class VideoSink:
    """Incremental RGB → MP4 writer. Prefers OpenCV, then ffmpeg pipe."""

    def __init__(self, path: Path, fps: int, width: int, height: int) -> None:
        self.path = Path(path)
        self.fps = int(fps)
        self.w = int(width)
        self.h = int(height)
        self._cv = None
        self._proc = None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self._try_ffmpeg():
            return
        if self._try_cv2():
            return
        raise RuntimeError(
            "Could not open an MP4 writer. Install opencv-python or ffmpeg."
        )

    def _try_cv2(self) -> bool:
        try:
            import cv2
        except ImportError:
            return False
        fourccs = ("mp4v", "avc1", "XVID", "H264")
        for code in fourccs:
            vw = cv2.VideoWriter(
                str(self.path),
                cv2.VideoWriter_fourcc(*code),
                float(self.fps),
                (self.w, self.h),
            )
            if vw.isOpened():
                self._cv = vw
                self._bgr = True
                return True
            vw.release()
        return False

    def _try_ffmpeg(self) -> bool:
        exe = shutil.which("ffmpeg")
        if exe is None:
            return False
        cmd = [
            exe, "-y",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{self.w}x{self.h}",
            "-r", str(self.fps),
            "-i", "-",
            "-an",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            str(self.path),
        ]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self._bgr = False
        return True

    def write(self, rgb: np.ndarray) -> None:
        frame = np.ascontiguousarray(rgb)
        if frame.shape[0] != self.h or frame.shape[1] != self.w:
            raise ValueError(f"frame {frame.shape} != {(self.h, self.w, 3)}")
        if self._cv is not None:
            import cv2
            self._cv.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            return
        if self._proc is not None and self._proc.stdin is not None:
            self._proc.stdin.write(frame.tobytes())

    def close(self) -> None:
        if self._cv is not None:
            self._cv.release()
            self._cv = None
        if self._proc is not None:
            if self._proc.stdin:
                self._proc.stdin.close()
            self._proc.wait(timeout=30)
            self._proc = None


class EpisodeSink:
    def __init__(
        self,
        root: Path,
        episode_index: int,
        cfg: SimulationConfig,
        output_mode: str,
        pool: ThreadPoolExecutor,
        max_inflight: int,
    ) -> None:
        self.cfg = cfg
        self.episode_index = int(episode_index)
        self.mode = output_mode
        self.write_frames = output_mode in ("frames", "both")
        self.write_video = output_mode in ("video", "both")
        self.dir = Path(root) / f"episode_{self.episode_index:05d}"
        self.rgb_dir = self.dir / "rgb"
        self.depth_dir = self.dir / "depth"
        self.seg_dir = self.dir / "segmentation"
        self.ann_dir = self.dir / "annotations"
        self.dir.mkdir(parents=True, exist_ok=True)
        if self.write_frames:
            for d in (self.rgb_dir, self.depth_dir, self.seg_dir, self.ann_dir):
                d.mkdir(parents=True, exist_ok=True)
        else:
            self.ann_dir.mkdir(parents=True, exist_ok=True)
        self._pool = pool
        self._futs: List[Future] = []
        self._max_inflight = max_inflight
        self._video: Optional[VideoSink] = None
        if self.write_video:
            self._video = VideoSink(
                self.dir / "video.mp4",
                fps=cfg.fps,
                width=cfg.camera.width,
                height=cfg.camera.height,
            )

    def _harvest(self, block: bool = False) -> None:
        alive: List[Future] = []
        for fut in self._futs:
            if block or fut.done():
                fut.result()
            else:
                alive.append(fut)
        self._futs = [] if block else alive

    def submit(
        self,
        frame_index: int,
        rgb: np.ndarray,
        depth: np.ndarray,
        seg: np.ndarray,
        annotation: Dict[str, Any],
    ) -> None:
        if self._video is not None:
            self._video.write(rgb)

        if len(self._futs) >= self._max_inflight:
            self._harvest(block=False)
            if len(self._futs) >= self._max_inflight:
                self._futs[0].result()
                self._futs = self._futs[1:]

        stem = f"frame_{frame_index:04d}"
        rgb_c = np.ascontiguousarray(rgb.copy()) if self.write_frames else None
        depth_c = np.ascontiguousarray(depth.copy()) if self.write_frames else None
        seg_c = np.ascontiguousarray(seg.copy()) if self.write_frames else None
        fut = self._pool.submit(
            _write_one,
            (self.rgb_dir / f"{stem}.png") if self.write_frames else None,
            (self.depth_dir / f"{stem}.npy") if self.write_frames else None,
            (self.seg_dir / f"{stem}.png") if self.write_frames else None,
            self.ann_dir / f"{stem}.json",
            rgb_c,
            depth_c,
            seg_c,
            annotation,
        )
        self._futs.append(fut)

    def close(self) -> None:
        self._harvest(block=True)
        if self._video is not None:
            self._video.close()
            self._video = None


class DatasetWriter:
    def __init__(
        self,
        root: Path | str,
        cfg: SimulationConfig,
        output_mode: str = "both",
        max_workers: Optional[int] = None,
    ) -> None:
        self.root = Path(root)
        self.cfg = cfg
        mode = str(output_mode).lower().strip()
        if mode not in ("frames", "video", "both"):
            raise ValueError("output_mode must be 'frames', 'video', or 'both'")
        self.output_mode = mode
        self.root.mkdir(parents=True, exist_ok=True)
        workers = int(max_workers if max_workers is not None else cfg.writer_workers)
        self._pool = ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="ds-write")
        self._max_inflight = max(8, workers * 8)
        self._active: Optional[EpisodeSink] = None

    def begin_episode(self, episode_index: int) -> EpisodeSink:
        if self._active is not None:
            self._active.close()
        self._active = EpisodeSink(
            self.root, episode_index, self.cfg, self.output_mode, self._pool, self._max_inflight
        )
        return self._active

    def close(self) -> None:
        if self._active is not None:
            self._active.close()
            self._active = None
        self._pool.shutdown(wait=True)

    def __enter__(self) -> "DatasetWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def build_annotation(
    cfg: SimulationConfig,
    episode_index: int,
    frame_index: int,
    timestamp: float,
    cam_pos: List[float],
    cam_vel: List[float],
    euler_deg: Dict[str, float],
    environment: Dict[str, Any],
    objects: List[Dict[str, Any]],
) -> Dict[str, Any]:
    cam = cfg.camera
    return {
        "dataset_metadata": {
            "generator": cfg.generator_name,
            "version": cfg.generator_version,
            "coordinate_system": "Right-Handed (+X Right, +Y Up, +Z Forward)",
        },
        "frame_id": f"ep{episode_index:05d}_f{frame_index:04d}",
        "timestamp": _f(timestamp),
        "camera_rig": {
            "world_position": [ _f(x) for x in cam_pos ],
            "linear_velocity": [ _f(x) for x in cam_vel ],
            "rotation_euler_deg": {
                "pitch": _f(euler_deg["pitch"]),
                "yaw": _f(euler_deg["yaw"]),
                "roll": _f(euler_deg["roll"]),
            },
            "intrinsics": {
                "fx": _f(cam.fx),
                "fy": _f(cam.fy),
                "cx": _f(cam.cx),
                "cy": _f(cam.cy),
                "resolution": [int(cam.width), int(cam.height)],
                "fov_deg": _f(cam.fov_x_deg),
            },
        },
        "environment": environment,
        "objects": objects,
    }
