"""
VisiAttend Pro — Student Registration
======================================
Captures N face samples from a live webcam, generates ArcFace embeddings
for each sample, and inserts them into the `register` table on Neon.tech.

Usage:
    python register_student.py                    # interactive prompts
    python register_student.py --name "Alice" --samples 5

Requirements (same env as attendance_scanner.py):
    pip install psycopg2-binary pgvector deepface opencv-python
    export DATABASE_URL="postgresql://user:pass@host/db?sslmode=require"
"""

import os
import sys
import time
import argparse
from pathlib import Path

# ── Silence TF / CUDA noise ───────────────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"]  = "0"

import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np

from deepface import DeepFace
from database import bootstrap_schema, get_connection, register_vector

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG  (must match attendance_scanner.py exactly)
# ──────────────────────────────────────────────────────────────────────────────
MODEL_NAME      = "ArcFace"
ALIGN_BACKEND   = "opencv"
FACE_RESIZE     = (160, 160)
FACE_DNN_CONF   = 0.55
FRAME_SCALE     = 0.6
MIN_FACE_PX     = 60
DEFAULT_SAMPLES = 20         # embeddings to capture per student
CAPTURE_WINDOW_SECONDS = 10

# DNN model files (same as scanner)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_FILE   = str(PROJECT_ROOT / "res10_300x300_ssd_iter_140000.caffemodel")
CONFIG_FILE  = str(PROJECT_ROOT / "deploy.prototxt")

# UI colours (BGR)
CLR_GREEN   = (0,  220,  80)
CLR_AMBER   = (0,  200, 255)
CLR_RED     = (0,   60, 200)
CLR_BLUE    = (200, 120,  0)
CLR_WHITE   = (255, 255, 255)
CLR_BLACK   = (0,    0,   0)
CLR_DARK    = (18,  18,  18)


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def download_dnn_files():
    import urllib.request
    if not os.path.exists(CONFIG_FILE):
        print("[INFO] Downloading deploy.prototxt …")
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/opencv/opencv/master/"
            "samples/dnn/face_detector/deploy.prototxt", CONFIG_FILE)
    if not os.path.exists(MODEL_FILE):
        print("[INFO] Downloading caffemodel …")
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/opencv/opencv_3rdparty/"
            "dnn_samples_face_detector_20170830/"
            "res10_300x300_ssd_iter_140000.caffemodel", MODEL_FILE)


def normalize_student_name(name: str) -> str:
    """Trim and collapse whitespace to avoid duplicate variants of same name."""
    return " ".join(name.strip().split())


def l2_norm(v: np.ndarray) -> np.ndarray | None:
    n = np.linalg.norm(v)
    return (v / n).astype(np.float32) if n > 1e-9 else None


def preprocess_face(img: np.ndarray) -> np.ndarray:
    return cv2.resize(img, FACE_RESIZE, interpolation=cv2.INTER_LANCZOS4)


def embed_face(roi: np.ndarray) -> np.ndarray | None:
    """ArcFace embed of one face crop — identical pipeline to scanner."""
    try:
        img = preprocess_face(roi)
        result = DeepFace.represent(
            img_path          = img,
            model_name        = MODEL_NAME,
            enforce_detection = False,
            detector_backend  = ALIGN_BACKEND,
            align             = True,
        )
        if result:
            emb = np.array(result[0]["embedding"], dtype=np.float32)
            return l2_norm(emb)
    except Exception:
        pass
    return None


def detect_best_face(frame: np.ndarray, net: cv2.dnn.Net):
    """
    Run DNN detector; return (x,y,x2,y2) of the highest-confidence face,
    or None if nothing qualifies.
    """
    h, w  = frame.shape[:2]
    small = cv2.resize(frame, (int(w * FRAME_SCALE), int(h * FRAME_SCALE)))
    sh, sw = small.shape[:2]

    blob = cv2.dnn.blobFromImage(small, 1.0, (300, 300), (104., 177., 123.))
    net.setInput(blob)
    dets = net.forward()

    best_conf, best_box = 0.0, None
    for i in range(dets.shape[2]):
        conf = float(dets[0, 0, i, 2])
        if conf < FACE_DNN_CONF:
            continue
        b = dets[0, 0, i, 3:7] * np.array([sw, sh, sw, sh])
        x, y, x2, y2 = (b / FRAME_SCALE).astype(int)
        x,  y  = max(0, x),  max(0, y)
        x2, y2 = min(w, x2), min(h, y2)
        if min(x2 - x, y2 - y) < MIN_FACE_PX:
            continue
        if conf > best_conf:
            best_conf, best_box = conf, (x, y, x2, y2)

    return best_box


# ──────────────────────────────────────────────────────────────────────────────
# DATABASE
# ──────────────────────────────────────────────────────────────────────────────
def student_exists(roll_no: str) -> bool:
    """Return True if a roll number already has rows in the register table."""
    conn = get_connection()
    try:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM register WHERE lower(trim(roll_no)) = lower(%s) LIMIT 1",
                (roll_no,)
            )
            return cur.fetchone() is not None
    finally:
        conn.close()


def aggregate_embeddings(embeddings: list[np.ndarray]) -> np.ndarray:
    """Average all sample embeddings and L2-normalize to one final vector."""
    stacked = np.stack(embeddings, axis=0)
    mean_emb = np.mean(stacked, axis=0).astype(np.float32)
    normalized = l2_norm(mean_emb)
    if normalized is None:
        raise ValueError("Failed to build a valid final embedding")
    return normalized


def upsert_student_embedding(roll_no: str, name: str, embedding: np.ndarray) -> None:
    """Ensure only one DB row per roll number by replacing previous records."""
    conn = get_connection()
    try:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute("DELETE FROM register WHERE lower(trim(roll_no)) = lower(%s)", (roll_no,))
            cur.execute(
                "INSERT INTO register (roll_no, name, embedding) VALUES (%s, %s, %s::vector)",
                (roll_no, name, embedding.tolist())
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ──────────────────────────────────────────────────────────────────────────────
# UI HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def draw_corner_bracket(frame, x, y, x2, y2, color, thickness=2, length=20):
    for px, py, dx, dy in [(x, y, 1, 1), (x2, y, -1, 1),
                            (x, y2, 1, -1), (x2, y2, -1, -1)]:
        cv2.line(frame, (px, py), (px + dx * length, py), color, thickness)
        cv2.line(frame, (px, py), (px, py + dy * length), color, thickness)


def draw_hud(frame, name: str, captured: int, total: int,
             state: str, cooldown_pct: float = 0.0):
    """
    Overlay a HUD strip at the bottom of the frame.
    state: 'waiting' | 'face_found' | 'capturing' | 'done' | 'no_face'
    """
    H, W = frame.shape[:2]
    panel_h = 80

    # Dark panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, H - panel_h), (W, H), CLR_DARK, -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

    # Progress bar
    bar_w = int((captured / total) * (W - 40))
    cv2.rectangle(frame, (20, H - 18), (W - 20, H - 8), (50, 50, 50), -1)
    if bar_w > 0:
        cv2.rectangle(frame, (20, H - 18), (20 + bar_w, H - 8), CLR_GREEN, -1)

    # State colour + label
    state_map = {
        "waiting":    (CLR_AMBER,  "ALIGN FACE TO FRAME"),
        "face_found": (CLR_GREEN,  "FACE DETECTED — AUTO CAPTURING"),
        "capturing":  (CLR_BLUE,   "CAPTURING …"),
        "done":       (CLR_GREEN,  "ALL SAMPLES CAPTURED  —  SAVING TO DB"),
        "no_face":    (CLR_RED,    "NO FACE DETECTED"),
    }
    colour, label = state_map.get(state, (CLR_WHITE, state))

    # Name
    cv2.putText(frame, f"Student: {name}",
                (20, H - panel_h + 22),
                cv2.FONT_HERSHEY_DUPLEX, 0.6, CLR_WHITE, 1, cv2.LINE_AA)

    # Sample counter
    counter_txt = f"Samples: {captured} / {total}"
    (cw, _), _ = cv2.getTextSize(counter_txt, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
    cv2.putText(frame, counter_txt,
                (W - cw - 20, H - panel_h + 22),
                cv2.FONT_HERSHEY_DUPLEX, 0.6, CLR_WHITE, 1, cv2.LINE_AA)

    # State label (centred)
    (lw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)
    cv2.putText(frame, label,
                ((W - lw) // 2, H - panel_h + 52),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, colour, 1, cv2.LINE_AA)

    # Controls hint
    hint = "AUTO MODE: 10s window, target samples auto-captured    [Q] Quit"
    (hw, _), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
    cv2.putText(frame, hint,
                ((W - hw) // 2, H - 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (130, 130, 130), 1, cv2.LINE_AA)


def flash_overlay(frame, color, alpha=0.35):
    """Flash a coloured overlay on capture."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_thumbnail_strip(frame, thumbnails: list[np.ndarray], thumb_size=56):
    """Draw captured face thumbnails as a strip in the top-right corner."""
    H, W = frame.shape[:2]
    pad = 8
    x_start = W - (thumb_size + pad) * len(thumbnails) - pad
    y_start = pad

    for i, thumb in enumerate(thumbnails):
        t = cv2.resize(thumb, (thumb_size, thumb_size))
        x = x_start + i * (thumb_size + pad)
        frame[y_start:y_start + thumb_size, x:x + thumb_size] = t
        cv2.rectangle(frame, (x, y_start), (x + thumb_size, y_start + thumb_size),
                      CLR_GREEN, 1)
        cv2.putText(frame, str(i + 1), (x + 3, y_start + 13),
                    cv2.FONT_HERSHEY_DUPLEX, 0.38, CLR_WHITE, 1)


# ──────────────────────────────────────────────────────────────────────────────
# CORE REGISTRATION FLOW
# ──────────────────────────────────────────────────────────────────────────────
def register_student(roll_no: str, name: str, num_samples: int, capture_seconds: int):
    print(f"\n{'='*58}")
    print(f"  Registering: {roll_no} - {name}  ({num_samples} samples)")
    print(f"{'='*58}")

    # Existing entry will be replaced to keep one row per student.
    if student_exists(roll_no):
        print(f"\n[INFO] Roll No '{roll_no}' already exists. Existing embedding will be replaced.")

    # Open camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        sys.exit(1)

    # DNN net
    net = cv2.dnn.readNetFromCaffe(CONFIG_FILE, MODEL_FILE)

    collected_embeddings: list[np.ndarray] = []
    collected_thumbs:     list[np.ndarray] = []

    state         = "waiting"
    flash_frames  = 0
    flash_color   = CLR_GREEN
    last_box      = None
    fail_count    = 0

    print("\n  Instructions:")
    print("  • Look directly at the camera, vary your angle slightly between samples")
    print(f"  • Auto capture starts immediately (minimum {capture_seconds} seconds)")
    print(f"  • Target samples: {num_samples}")
    print("  • Keep your face visible and vary angles naturally")
    print("  • Press [Q] to quit without saving\n")

    start_ts = time.time()
    capture_interval = max(capture_seconds / max(1, num_samples), 0.08)
    next_capture_ts = start_ts
    extended_mode_notified = False

    while True:
        now = time.time()
        elapsed = now - start_ts
        if elapsed >= capture_seconds and len(collected_embeddings) < num_samples and not extended_mode_notified:
            print(
                f"[INFO] Collected {len(collected_embeddings)}/{num_samples} in {capture_seconds}s. "
                "Continuing until 20 samples are completed..."
            )
            extended_mode_notified = True

        if len(collected_embeddings) >= num_samples:
            state = "done"
            time.sleep(0.4)
            break

        ret, frame = cap.read()
        if not ret:
            fail_count += 1
            if fail_count > 30:
                print("[ERROR] Camera stopped."); break
            continue
        fail_count = 0
        frame = cv2.flip(frame, 1)

        box = detect_best_face(frame, net)
        captured = len(collected_embeddings)

        if box is not None:
            state = "face_found"
            last_box = box
        else:
            state = "waiting" if last_box is None else "no_face"

        # Draw face bracket
        if box is not None:
            x, y, x2, y2 = box
            bcolour = CLR_GREEN if state == "face_found" else CLR_AMBER
            draw_corner_bracket(frame, x, y, x2, y2, bcolour, thickness=2, length=22)

        # Flash effect after capture
        if flash_frames > 0:
            flash_overlay(frame, flash_color, alpha=0.30)
            flash_frames -= 1

        # Thumbnail strip
        if collected_thumbs:
            draw_thumbnail_strip(frame, collected_thumbs)

        # HUD
        draw_hud(frame, f"{roll_no} - {name}", captured, num_samples, state)

        cv2.imshow("VisiAttend — Student Registration", frame)
        key = cv2.waitKey(1) & 0xFF

        # ── QUIT ──
        if key == ord('q'):
            print("\n[QUIT] Registration cancelled — nothing saved.")
            cap.release(); cv2.destroyAllWindows(); return

        # ── AUTO CAPTURE ──
        if state == "face_found" and now >= next_capture_ts and captured < num_samples:
            x, y, x2, y2 = box
            roi = frame[y:y2, x:x2].copy()

            print(f"  [EMBEDDING] Sample {captured + 1}/{num_samples} … ", end="", flush=True)
            emb = embed_face(roi)

            if emb is not None:
                collected_embeddings.append(emb)
                collected_thumbs.append(cv2.resize(roi, (56, 56)))
                flash_color  = CLR_GREEN
                flash_frames = 6

                # Try to stay on pace to hit target samples by capture_seconds.
                if elapsed < capture_seconds and captured + 1 < num_samples:
                    remaining_time = max(0.2, capture_seconds - elapsed)
                    remaining_samples = max(1, num_samples - (captured + 1))
                    capture_interval = max(remaining_time / remaining_samples, 0.05)
                else:
                    capture_interval = 0.05

                next_capture_ts = now + capture_interval
                print(f"OK  (L2-norm: {np.linalg.norm(emb):.4f})")
            else:
                flash_color  = CLR_RED
                flash_frames = 6
                next_capture_ts = now + 0.15
                print("FAILED — try again")

    cap.release()
    cv2.destroyAllWindows()

    if not collected_embeddings:
        print("\n[ABORT] No embeddings captured — nothing saved.")
        return

    if len(collected_embeddings) < num_samples:
        print(
            f"\n[ABORT] Only {len(collected_embeddings)}/{num_samples} samples captured in "
            f"{capture_seconds}s. Please retry with better face visibility."
        )
        return

    # ── SAVE TO DB ────────────────────────────────────────────────────────────
    print(f"\n  Creating one final embedding from {len(collected_embeddings)} samples …")
    try:
        final_embedding = aggregate_embeddings(collected_embeddings)
        upsert_student_embedding(roll_no, name, final_embedding)
        print(f"\n{'='*58}")
        print(f"  ✓  {roll_no} - {name} registered successfully  (1 DB row saved)")
        print(f"{'='*58}\n")
    except Exception as e:
        print(f"\n[DB-ERROR] Failed to insert: {e}")
        sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="VisiAttend Pro — register a new student into Neon.tech DB")
    p.add_argument("--roll-no", "-r", type=str,
                   help="Roll number of the student (e.g. '22CS001')")
    p.add_argument("--name",    "-n", type=str,
                   help="Full name of the student (e.g. 'Alice Smith')")
    p.add_argument("--samples", "-s", type=int, default=DEFAULT_SAMPLES,
                   help=f"Number of face samples to capture (default: {DEFAULT_SAMPLES})")
    p.add_argument("--duration", "-d", type=int, default=CAPTURE_WINDOW_SECONDS,
                   help=f"Capture duration in seconds (default: {CAPTURE_WINDOW_SECONDS})")
    return p.parse_args()


def main():
    # Ensure required tables/extensions exist before registration starts.
    bootstrap_schema()
    download_dnn_files()
    args = parse_args()

    roll_no = normalize_student_name(args.roll_no or "")
    if not roll_no:
        roll_no = normalize_student_name(input("  Enter student roll number: "))
        if not roll_no:
            print("[ERROR] Roll number cannot be empty."); sys.exit(1)

    name = args.name
    if not name:
        print("\n  VisiAttend Pro — Student Registration")
        print("  " + "─"*38)
        name = normalize_student_name(input("  Enter student full name: "))
        if not name:
            print("[ERROR] Name cannot be empty."); sys.exit(1)
    else:
        name = normalize_student_name(name)
        if not name:
            print("[ERROR] Name cannot be empty."); sys.exit(1)

    samples = args.samples
    if samples < 1 or samples > 20:
        print("[ERROR] Samples must be between 1 and 20."); sys.exit(1)

    duration = args.duration
    if duration < 1 or duration > 60:
        print("[ERROR] Duration must be between 1 and 60 seconds."); sys.exit(1)

    register_student(roll_no, name, samples, duration)


if __name__ == "__main__":
    main()