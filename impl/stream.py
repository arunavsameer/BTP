import argparse
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Event, Lock, Thread

import cv2

from tracker import CROPPED_DIR, MODEL_WEIGHTS, TARGET_CLASSES, open_frame_source, weights_path
from ultralytics import YOLOE

DEFAULT_SOURCE = CROPPED_DIR / "untitledwalking.mp4"
HOST = "127.0.0.1"
PORT = 8080

PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Obstacle stream</title>
  <style>
    body { margin: 0; background: #111; color: #eee; font-family: sans-serif; text-align: center; }
    img { max-width: 100%; height: auto; background: #000; }
    p { color: #aaa; }
  </style>
</head>
<body>
  <h1>YOLOE live labels</h1>
  <p>Playback is paced at 30 FPS. Inference runs on the latest frame and may be slower on CPU.</p>
  <img src="/stream" alt="annotated stream">
</body>
</html>
"""


class StreamState:
    def __init__(self):
        self.lock = Lock()
        self.frame = None
        self.dets = []
        self.infer_fps = 0.0
        self.stream_fps = 0.0
        self.running = Event()
        self.running.set()


def extract_dets(result):
    dets = []
    if result.boxes is None or len(result.boxes) == 0:
        return dets
    boxes = result.boxes.xyxy.cpu().tolist()
    class_indices = result.boxes.cls.int().cpu().tolist()
    confidences = result.boxes.conf.cpu().tolist()
    if result.boxes.id is not None:
        track_ids = result.boxes.id.int().cpu().tolist()
    else:
        track_ids = [None] * len(boxes)
    names = result.names
    for box, track_id, cls_idx, confidence in zip(boxes, track_ids, class_indices, confidences):
        dets.append(
            {
                "bbox": box,
                "class": names[int(cls_idx)],
                "confidence": float(confidence),
                "track_id": track_id,
            }
        )
    return dets


def draw_dets(frame, dets, infer_fps, stream_fps, target_fps):
    vis = frame.copy()
    for det in dets:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        label = det["class"]
        if det["track_id"] is not None:
            label += f" #{det['track_id']}"
        label += f" {det['confidence']:.2f}"
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 255), 2)
        cv2.putText(vis, label, (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)

    line1 = f"Video you see: {stream_fps:.1f} fps  (goal {target_fps:.0f})"
    line2 = f"Box updates:   {infer_fps:.1f} fps"
    color = (80, 220, 80) if infer_fps >= 25 else (40, 180, 255) if infer_fps >= 10 else (40, 40, 255)
    cv2.rectangle(vis, (8, 8), (430, 52), (0, 0, 0), -1)
    cv2.putText(vis, line1, (16, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
    cv2.putText(vis, line2, (16, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return vis


def capture_loop(state, source, target_fps):
    interval = 1.0 / target_fps
    while state.running.is_set():
        cap, _, _, _ = open_frame_source(source)
        if cap is None:
            print(f"Error: could not open {source}")
            state.running.clear()
            return
        next_t = time.perf_counter()
        while state.running.is_set():
            ok, frame = cap.read()
            if not ok:
                break
            with state.lock:
                state.frame = frame
            now = time.perf_counter()
            sleep_s = next_t - now
            if sleep_s > 0:
                time.sleep(sleep_s)
            next_t += interval
            if now - next_t > 1:
                next_t = now + interval
        cap.release()


def infer_loop(state, model, conf, imgsz):
    while state.running.is_set():
        with state.lock:
            frame = None if state.frame is None else state.frame.copy()
        if frame is None:
            time.sleep(0.01)
            continue
        t0 = time.perf_counter()
        result = model.track(
            frame,
            persist=True,
            tracker="botsort.yaml",
            conf=conf,
            imgsz=imgsz,
            verbose=False,
        )[0]
        dt = time.perf_counter() - t0
        dets = extract_dets(result)
        with state.lock:
            state.dets = dets
            state.infer_fps = 1.0 / dt if dt > 0 else 0.0


def make_handler(state, target_fps):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            if self.path != "/stream":
                super().log_message(fmt, *args)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                body = PAGE.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path != "/stream":
                self.send_error(404)
                return

            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()

            interval = 1.0 / target_fps
            ema = target_fps
            while state.running.is_set():
                t0 = time.perf_counter()
                with state.lock:
                    frame = None if state.frame is None else state.frame.copy()
                    dets = list(state.dets)
                    infer_fps = state.infer_fps
                if frame is None:
                    time.sleep(0.01)
                    continue
                vis = draw_dets(frame, dets, infer_fps, ema, target_fps)
                ok, buf = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if not ok:
                    continue
                payload = buf.tobytes()
                try:
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(payload)}\r\n\r\n".encode("ascii"))
                    self.wfile.write(payload)
                    self.wfile.write(b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    return
                dt = time.perf_counter() - t0
                ema = 0.9 * ema + 0.1 * (1.0 / dt if dt > 0 else target_fps)
                with state.lock:
                    state.stream_fps = ema
                time.sleep(max(0.0, interval - dt))

    return Handler


def main():
    parser = argparse.ArgumentParser(description="Stream annotated video on localhost.")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=480)
    args = parser.parse_args()

    weights = weights_path()
    print(f"Loading {weights.name}...", flush=True)
    model = YOLOE(str(weights) if weights.exists() else MODEL_WEIGHTS)
    model.eval()
    model.set_classes(TARGET_CLASSES)

    state = StreamState()
    Thread(target=capture_loop, args=(state, args.source, args.fps), daemon=True).start()
    Thread(target=infer_loop, args=(state, model, args.conf, args.imgsz), daemon=True).start()

    server = ThreadingHTTPServer((args.host, args.port), make_handler(state, args.fps))
    url = f"http://{args.host}:{args.port}"
    print(f"Streaming {args.source} at {args.fps:.0f} FPS", flush=True)
    print(f"Open {url} to see boxes", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        state.running.clear()
        server.shutdown()


if __name__ == "__main__":
    main()
