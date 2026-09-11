import json
import subprocess
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLOE

IMPL_DIR = Path(__file__).resolve().parent
INPUT_DIR = IMPL_DIR / "input"
CROPPED_DIR = IMPL_DIR / "cropped"
OUTPUT_DIR = IMPL_DIR / "output"
WEIGHTS_DIR = IMPL_DIR / "weights"

TARGET_CLASSES = [
    "person",
    "car",
    "bicycle",
    "motorcycle",
    "scooter",
    "dog",
    "fire hydrant",
    "pole",
    "pillar",
    "trash can",
    "traffic cone",
    "box",
    "pothole",
]
MODEL_WEIGHTS = "weights/yoloe-26s-seg.pt"


def weights_path() -> Path:
    local = IMPL_DIR / MODEL_WEIGHTS
    return local if local.exists() else Path(MODEL_WEIGHTS)


class PrefixedCapture:
    """Replay a frame already read while probing, then continue from the original capture."""

    def __init__(self, cap, first_frame):
        self.cap = cap
        self._first = first_frame

    def read(self):
        if self._first is not None:
            frame, self._first = self._first, None
            return True, frame
        return self.cap.read()

    def release(self):
        self.cap.release()


class FFmpegCapture:
    """Software-decode codecs OpenCV cannot handle (e.g. AV1 without hardware support)."""

    def __init__(self, path, width, height):
        self.width = width
        self.height = height
        self.frame_size = width * height * 3
        self.proc = subprocess.Popen(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                path,
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-an",
                "-",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

    def read(self):
        raw = self.proc.stdout.read(self.frame_size)
        if not raw or len(raw) < self.frame_size:
            return False, None
        frame = np.frombuffer(raw, dtype=np.uint8).reshape((self.height, self.width, 3)).copy()
        return True, frame

    def release(self):
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait()


def open_frame_source(input_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return None, 0, 0, 0.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    ok, first = cap.read()
    if ok:
        return PrefixedCapture(cap, first), width, height, fps

    cap.release()
    if width <= 0 or height <= 0:
        return None, 0, 0, fps

    print("OpenCV could not decode this video (likely AV1 without hardware support). Falling back to ffmpeg...")
    return FFmpegCapture(input_path, width, height), width, height, fps


def process_and_extract_trajectories(input_path, output_path, labels_path=None, conf=0.25):
    input_path = str(input_path)
    output_path = str(Path(output_path).resolve())
    labels_path = str(Path(labels_path).resolve()) if labels_path else str(Path(output_path).with_suffix(".json"))

    print(f"Loading {weights_path().name}...")
    model = YOLOE(str(weights_path()))
    model.eval()
    model.set_classes(TARGET_CLASSES)

    cap, width, height, fps = open_frame_source(input_path)
    if cap is None:
        print(f"Error: Could not open {input_path}")
        return None

    if fps is None or fps <= 1e-3:
        fps = 30.0
    writer_fps = int(round(fps))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), writer_fps, (width, height))

    detections = []
    trajectory_data = defaultdict(list)
    frame_count = 0
    print("Processing video and extracting data...")

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        results = model.track(frame, persist=True, tracker="botsort.yaml", conf=conf, verbose=False)
        result = results[0]

        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().tolist()
            class_indices = result.boxes.cls.int().cpu().tolist()
            confidences = result.boxes.conf.cpu().tolist()
            if result.boxes.id is not None:
                track_ids = result.boxes.id.int().cpu().tolist()
            else:
                track_ids = [None] * len(boxes)

            names = result.names
            for box, track_id, cls_idx, confidence in zip(boxes, track_ids, class_indices, confidences):
                class_name = names[int(cls_idx)]
                record = {
                    "frame": frame_count,
                    "bbox": [round(v, 2) for v in box],
                    "class": class_name,
                    "confidence": round(float(confidence), 4),
                    "track_id": track_id,
                }
                detections.append(record)
                if track_id is not None:
                    trajectory_data[track_id].append(record)

        annotated_frame = result.plot()
        out.write(annotated_frame)

    cap.release()
    out.release()

    if frame_count == 0:
        print(f"Error: opened {input_path} but decoded 0 frames.")
        return None

    payload = {
        "video": {
            "input": input_path,
            "output": output_path,
            "width": width,
            "height": height,
            "fps": writer_fps,
            "frames": frame_count,
            "model": MODEL_WEIGHTS,
            "classes": TARGET_CLASSES,
        },
        "detections": detections,
        "trajectories": {str(track_id): records for track_id, records in trajectory_data.items()},
    }
    Path(labels_path).parent.mkdir(parents=True, exist_ok=True)
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    print(f"Done! Saved annotated video to {output_path}")
    print(f"Saved labels to {labels_path}")
    print(f"{len(detections)} detections across {frame_count} frames; {len(trajectory_data)} tracked objects.")
    return payload


if __name__ == "__main__":
    data = process_and_extract_trajectories(
        CROPPED_DIR / "untitledwalking.mp4",
        OUTPUT_DIR / "labels.mp4",
    )

    if not data:
        raise SystemExit(1)
    if data["trajectories"]:
        first_id, records = next(iter(data["trajectories"].items()))
        print(f"\nHistory for Object ID {first_id} ({records[0]['class']}):")
        print(f"Appeared in {len(records)} frames.")
        print(f"First seen at coordinates: {records[0]['bbox']}")
        print(f"Last seen at coordinates: {records[-1]['bbox']}")
    else:
        print("\nNo track IDs were assigned; per-frame detections were still saved.")
