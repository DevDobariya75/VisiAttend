"""
VisiAttend Pro — Slot-based Attendance Scanner
==============================================
Flow implemented by this script:
    1. Teacher creates an attendance slot with subject, room, faculty, start/end time.
    2. Camera starts and recognised students are marked present for that slot.
    3. On completion, all unmarked students remain absent and slot status is closed.

DB tables used:
    register                (roll_no, name, embedding)
    attendance_slots        (slot metadata and status)
    attendance_slot_records (slot-wise present/absent records)
"""

import os
import sys
import argparse
import uuid

# Silence TF noise before any import
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"]  = "0"   # use GPU if available

import warnings
warnings.filterwarnings("ignore")

import cv2
import threading
import time
import numpy as np
from datetime import datetime
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor

from deepface import DeepFace

# ── NEW: database helpers ──────────────────────────────
from database import bootstrap_schema, get_connection, register_vector

# ──────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────
MODEL_NAME         = "ArcFace"     # 512-dim embeddings
EMBED_DIM          = 512

ALIGN_BACKEND      = "opencv"      # detector used to find + align faces
SIMILARITY_THRESH  = 0.38          # strict (but slightly relaxed): cosine dist threshold—lower=better match
MIN_MATCH_GAP      = 0.25          # strict (but slightly relaxed): best must dominate runner-up

FACE_DNN_CONF      = 0.55          # DNN detector min confidence for live detection
FRAME_SCALE        = 0.5           # downscale factor for live DNN detection
MIN_FACE_PX        = 50            # skip faces smaller than this (px)
CONFIRM_FRAMES     = 5             # frames for confirmation (reduced: allows faster detection with movement)
RECOGNITION_FPS    = 12            # max recognition cycles / sec
EMBED_WORKERS      = 2             # parallel embedding workers
FACE_RESIZE        = (160, 160)    # ArcFace native input size

# Quality + tracker constraints: LENIENT on pose/movement, STRICT on identity.
MIN_FACE_SHARPNESS = 70.0          # allow slight blur; registered faces may move
TRACK_CENTER_MAX_FACTOR = 1.4      # allow head movement and slight shifts
MAX_FACE_ANGLE_YAW = 40            # allow faces turned up to ±40° left/right
MAX_FACE_ANGLE_PITCH = 30          # allow faces tilted up to ±30° up/down
FACE_SIZE_VARIANCE_MAX = 1.25      # allow size changes up to ±25% between consecutive frames


def estimate_head_pose(roi: np.ndarray) -> tuple[float, float]:
    """
    Estimate yaw and pitch using simple heuristics on landmark positions.
    Returns (yaw_angle, pitch_angle) in degrees. Conservative estimates.
    """
    try:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) > 0:
            x, y, w, h = faces[0]
            aspect = float(w) / max(1, h)
            yaw = 0.0 if 0.8 <= aspect <= 1.2 else (20.0 if aspect > 1.2 else -20.0)
            pitch = 0.0
            return yaw, pitch
    except:
        pass
    return 0.0, 0.0


# Quality + tracker constraints for higher precision in live classrooms.
MIN_FACE_SHARPNESS = 75.0          # Laplacian variance; lower values are blurrier
TRACK_CENTER_MAX_FACTOR = 1.25     # max center movement factor for track association

model_file  = "res10_300x300_ssd_iter_140000.caffemodel"
config_file = "deploy.prototxt"

# ──────────────────────────────────────────────────────
# DOWNLOAD DNN FILES
# ──────────────────────────────────────────────────────
def download_dnn_files():
    import urllib.request
    if not os.path.exists(config_file):
        print("[INFO] Downloading deploy.prototxt ...")
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
            config_file)
    if not os.path.exists(model_file):
        print("[INFO] Downloading caffemodel ...")
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
            model_file)

download_dnn_files()
face_net = cv2.dnn.readNetFromCaffe(config_file, model_file)

# ──────────────────────────────────────────────────────
# UTILITY
# ──────────────────────────────────────────────────────
def l2_norm(v: np.ndarray) -> np.ndarray | None:
    n = np.linalg.norm(v)
    return (v / n).astype(np.float32) if n > 1e-9 else None


def preprocess_face(img: np.ndarray) -> np.ndarray:
    """Resize to ArcFace native input size."""
    return cv2.resize(img, FACE_RESIZE, interpolation=cv2.INTER_LANCZOS4)


def warmup_models() -> None:
    """Warm up DeepFace model once so first real recognition is faster."""
    try:
        dummy = np.zeros((FACE_RESIZE[1], FACE_RESIZE[0], 3), dtype=np.uint8)
        DeepFace.represent(
            img_path=dummy,
            model_name=MODEL_NAME,
            enforce_detection=False,
            detector_backend=ALIGN_BACKEND,
            align=True,
        )
        print("[INFO] DeepFace model warm-up complete.")
    except Exception as e:
        print(f"[WARN] DeepFace warm-up skipped: {e}")


# ──────────────────────────────────────────────────────
# LIVE FACE EMBEDDING  (called every recognition cycle)
# ──────────────────────────────────────────────────────
def embed_live_face(roi: np.ndarray) -> np.ndarray | None:
    """Embed one live face crop. Must match dataset embedding pipeline."""
    try:
        img = preprocess_face(roi)
        result = DeepFace.represent(
            img_path          = img,
            model_name        = MODEL_NAME,
            enforce_detection = False,
            detector_backend  = ALIGN_BACKEND,
            align             = True,
        )
        if result and len(result) > 0:
            emb = np.array(result[0]["embedding"], dtype=np.float32)
            return l2_norm(emb)
    except Exception:
        pass
    return None


# ──────────────────────────────────────────────────────
# DB RECOGNITION  — cosine vector search via pgvector
# ──────────────────────────────────────────────────────
known_matrix: np.ndarray | None = None
known_rolls: list[str] = []
known_names: list[str] = []


def load_registered_embeddings() -> int:
    """Load latest embedding per roll number into memory for fast matching."""
    global known_matrix, known_rolls, known_names
    conn = get_connection()
    register_vector(conn)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT roll_no, name, embedding
                FROM (
                    SELECT DISTINCT ON (roll_no) roll_no, name, embedding, id
                    FROM register
                    ORDER BY roll_no, id DESC
                ) latest
                ORDER BY roll_no
                """
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        known_matrix = None
        known_rolls = []
        known_names = []
        return 0

    vectors: list[np.ndarray] = []
    rolls: list[str] = []
    names: list[str] = []
    for roll_no, name, emb in rows:
        vec = np.array(emb, dtype=np.float32)
        normed = l2_norm(vec)
        if normed is None:
            continue
        vectors.append(normed)
        rolls.append(roll_no)
        names.append(name)

    if not vectors:
        known_matrix = None
        known_rolls = []
        known_names = []
        return 0

    known_matrix = np.stack(vectors, axis=0)
    known_rolls = rolls
    known_names = names
    return len(known_rolls)


def recognize_from_cache(embedding: np.ndarray, occ: str, blur_var: float) -> dict | None:
    """
    Match live embedding against in-memory registered embeddings.
    Returns the matched student identity dict or None if no match.

    SQL distance range: 0.0 (identical) -> 2.0 (opposite).
    Final acceptance threshold is quality-aware and derived from SIMILARITY_THRESH.
    """
    try:
        if known_matrix is None or known_matrix.shape[0] == 0:
            return None

        # Cosine distance for normalized vectors: distance = 1 - dot(a, b)
        sims = known_matrix @ embedding
        order = np.argsort(-sims)
        best_idx = int(order[0])
        distance = float(1.0 - sims[best_idx])
        second_distance = float(1.0 - sims[int(order[1])]) if sims.shape[0] > 1 else 2.0
        match_gap = second_distance - distance
        roll_no = known_rolls[best_idx]
        name = known_names[best_idx]

        # STRICT identity matching with FLEXIBLE pose acceptance.
        # Heavy occlusion always rejected; others allowed if match is strong enough.
        if occ == "heavy":
            return None
        
        # Relax quality requirements for slight blur/mask if match is excellent
        if blur_var < MIN_FACE_SHARPNESS:
            # Allow slightly blurry if distance is very tight
            if distance > (SIMILARITY_THRESH - 0.05) or match_gap < (MIN_MATCH_GAP - 0.02):
                return None
        
        # For occluded faces (mask/hand), require tighter match
        if occ in ("mask", "hand"):
            if distance > (SIMILARITY_THRESH - 0.03) or match_gap < (MIN_MATCH_GAP + 0.03):
                return None
        else:
            # Clear faces: normal strict thresholds
            if distance > SIMILARITY_THRESH or match_gap < MIN_MATCH_GAP:
                return None
        
        return {
            "roll_no": roll_no,
            "name": name,
            "distance": distance,
            "match_gap": match_gap,
        }

    except Exception as e:
        print(f"  [MATCH-WARN] recognize_from_cache: {e}")
        return None


# ──────────────────────────────────────────────────────
# BATCH RECOGNITION  — in-memory nearest-neighbor matching
# ──────────────────────────────────────────────────────
def db_batch_recognize(face_rois: list[np.ndarray], occs: list[str], blurs: list[float], 
                       poses: list[tuple[float, float]]) -> list[dict | None]:
    """
    Embed all live face crops in parallel, then fire one DB vector search
    per face (also in parallel). Returns identity dict (or None) per ROI.
    """
    if not face_rois:
        return []

    # Step 1 — parallel ArcFace embedding
    with ThreadPoolExecutor(max_workers=min(EMBED_WORKERS, len(face_rois))) as pool:
        embeddings = list(pool.map(embed_live_face, face_rois))

    # Step 2 — in-memory vector search for valid embeddings
    results: list[dict | None] = [None] * len(face_rois)

    valid_idx = [(i, emb) for i, emb in enumerate(embeddings) if emb is not None]
    if not valid_idx:
        return results

    def _search(args):
        idx, emb = args
        # Allow varied poses; matching is strict enough to reject unknowns alone
        return idx, recognize_from_cache(emb, occs[idx], blurs[idx])

    with ThreadPoolExecutor(max_workers=min(EMBED_WORKERS, len(valid_idx))) as pool:
        for idx, name in pool.map(_search, valid_idx):
            results[idx] = name

    return results


# ──────────────────────────────────────────────────────
# ATTENDANCE SLOT OPERATIONS
# ──────────────────────────────────────────────────────
def parse_slot_datetime(value: str) -> datetime:
    """Parse input date/time in either YYYY-MM-DD HH:MM or HH:MM format."""
    value = value.strip()
    if not value:
        raise ValueError("Empty date/time value")

    for fmt in ("%Y-%m-%d %H:%M", "%H:%M"):
        try:
            parsed = datetime.strptime(value, fmt)
            if fmt == "%H:%M":
                now = datetime.now()
                parsed = parsed.replace(year=now.year, month=now.month, day=now.day)
            return parsed
        except ValueError:
            continue

    raise ValueError("Use 'YYYY-MM-DD HH:MM' or 'HH:MM' format")


def generate_class_id() -> str:
    """Generate a unique class identifier for each new attendance class."""
    stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    token = uuid.uuid4().hex[:8].upper()
    return f"CLASS-{stamp}-{token}"


def create_attendance_slot(subject_name: str, lecture_room: str,
                           faculty_name: str, start_time: datetime,
                           end_time: datetime) -> tuple[int, str]:
    """Create an attendance slot and pre-populate absent records for all students."""
    try:
        conn = get_connection()
        register_vector(conn)
        class_id = generate_class_id()

        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO attendance_slots (
                    class_id, subject_name, lecture_room, faculty_name, start_time, end_time, status
                ) VALUES (%s, %s, %s, %s, %s, %s, 'open')
                RETURNING id
                """,
                (class_id, subject_name, lecture_room, faculty_name, start_time, end_time)
            )
            slot_id = cur.fetchone()[0]

            # Initialize records for all registered students as absent.
            cur.execute(
                """
                INSERT INTO attendance_slot_records (slot_id, class_id, roll_no, name, status)
                SELECT %s, %s, src.roll_no, src.name, 'absent'
                FROM (
                    SELECT DISTINCT ON (roll_no) roll_no, name
                    FROM register
                    ORDER BY roll_no, id DESC
                ) AS src
                ON CONFLICT (slot_id, roll_no) DO NOTHING
                """,
                (slot_id, class_id)
            )

        conn.commit()
        conn.close()
        return slot_id, class_id

    except Exception as e:
        print(f"  [DB-ERR] create_attendance_slot: {e}")
        raise


def mark_present_for_slot(slot_id: int, class_id: str, roll_no: str, name: str) -> bool:
    """Mark one student as present in the active slot."""
    try:
        conn = get_connection()
        register_vector(conn)

        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO attendance_slot_records (slot_id, class_id, roll_no, name, status, marked_at)
                VALUES (%s, %s, %s, %s, 'present', %s)
                ON CONFLICT (slot_id, roll_no)
                DO UPDATE SET
                    class_id = EXCLUDED.class_id,
                    name = EXCLUDED.name,
                    status = 'present',
                    marked_at = EXCLUDED.marked_at
                """,
                (slot_id, class_id, roll_no, name, datetime.utcnow())
            )

        conn.commit()
        conn.close()
        return True

    except Exception as e:
        print(f"  [DB-ERR] mark_present_for_slot: {e}")
        return False


def complete_attendance_slot(slot_id: int) -> tuple[int, int]:
    """Close slot and return (present_count, absent_count)."""
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE attendance_slots SET status = 'completed', completed_at = now() WHERE id = %s",
                (slot_id,)
            )
            cur.execute(
                "SELECT COUNT(*) FROM attendance_slot_records WHERE slot_id = %s AND status = 'present'",
                (slot_id,)
            )
            present_count = int(cur.fetchone()[0])
            cur.execute(
                "SELECT COUNT(*) FROM attendance_slot_records WHERE slot_id = %s AND status = 'absent'",
                (slot_id,)
            )
            absent_count = int(cur.fetchone()[0])

        conn.commit()
        conn.close()
        return present_count, absent_count
    except Exception as e:
        print(f"  [DB-ERR] complete_attendance_slot: {e}")
        return 0, 0


def collect_slot_details(args) -> tuple[str, str, str, datetime, datetime]:
    """Collect attendance slot details from args or interactive prompts."""
    subject_name = (args.subject or input("Enter subject name: ")).strip()
    lecture_room = (args.room or input("Enter lecture room: ")).strip()
    faculty_name = (args.faculty or input("Enter faculty name: ")).strip()

    if not subject_name or not lecture_room or not faculty_name:
        raise ValueError("Subject, room, and faculty are required")

    start_raw = (args.start_time or input("Enter start time (YYYY-MM-DD HH:MM or HH:MM): ")).strip()
    end_raw = (args.end_time or input("Enter end time   (YYYY-MM-DD HH:MM or HH:MM): ")).strip()
    start_time = parse_slot_datetime(start_raw)
    end_time = parse_slot_datetime(end_raw)
    if end_time <= start_time:
        raise ValueError("End time must be after start time")

    return subject_name, lecture_room, faculty_name, start_time, end_time


def parse_args():
    parser = argparse.ArgumentParser(
        description="Start slot-wise attendance scanner and mark present/absent"
    )
    parser.add_argument("--subject", type=str, help="Subject name")
    parser.add_argument("--room", type=str, help="Lecture room")
    parser.add_argument("--faculty", type=str, help="Faculty name")
    parser.add_argument("--start-time", type=str, help="Start time: YYYY-MM-DD HH:MM or HH:MM")
    parser.add_argument("--end-time", type=str, help="End time: YYYY-MM-DD HH:MM or HH:MM")
    parser.add_argument(
        "--threshold",
        type=float,
        default=SIMILARITY_THRESH,
        help=f"Cosine distance match threshold (default: {SIMILARITY_THRESH})"
    )
    return parser.parse_args()


# ──────────────────────────────────────────────────────
# OCCLUSION ANALYSIS
# ──────────────────────────────────────────────────────
def analyze_occlusion(roi: np.ndarray) -> str:
    if roi is None or roi.size == 0:
        return "heavy"
    h, w  = roi.shape[:2]
    gray  = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    lv    = float(np.var(gray[h//2:,  w//6:w*5//6]))
    uv    = float(np.var(gray[:h//2,  w//6:w*5//6]))
    T     = 180
    if lv < T and uv >= T: return "mask"
    if uv < T and lv >= T: return "hand"
    if lv < T and uv < T:  return "heavy"
    return "clear"


def smart_crop(roi: np.ndarray, occ: str) -> np.ndarray:
    h = roi.shape[0]
    if occ == "mask": return roi[:int(h * 0.65), :]
    if occ == "hand": return roi[int(h * 0.45):, :]
    return roi


# ──────────────────────────────────────────────────────
# SIMPLE IoU TRACKER  (stabilises labels across frames)
# ──────────────────────────────────────────────────────
track_state    = {}
next_track_id  = 1
TRACK_IOU_MIN  = 0.25
HISTORY_LEN    = 5

def iou(a, b):
    ix1, iy1 = max(a[0],b[0]), max(a[1],b[1])
    ix2, iy2 = min(a[2],b[2]), min(a[3],b[3])
    iw, ih   = max(0, ix2-ix1), max(0, iy2-iy1)
    inter    = iw * ih
    if inter == 0: return 0.0
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union > 0 else 0.0


def center_distance_ok(a, b) -> bool:
    acx, acy = (a[0] + a[2]) * 0.5, (a[1] + a[3]) * 0.5
    bcx, bcy = (b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5
    dist = ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5
    aw, ah = max(1, a[2] - a[0]), max(1, a[3] - a[1])
    bw, bh = max(1, b[2] - b[0]), max(1, b[3] - b[1])
    ref = max(aw, ah, bw, bh)
    # Allow movement and size variation; identity matching is strict enough
    return dist <= (TRACK_CENTER_MAX_FACTOR * ref) and abs(aw - bw) <= max(bw * 0.25, 8) and abs(ah - bh) <= max(bh * 0.25, 8)


def update_tracks(detections):
    """
    detections: list of {box, roll_no, name, occ, distance, gap}
    Returns assigned list with track ids and current detection labels.
    """
    global next_track_id
    assigned   = []
    unmatched  = set(track_state.keys())

    for det in detections:
        best_id, best_iou_score = None, 0.0
        for tid in list(unmatched):
            s = iou(det["box"], track_state[tid]["box"])
            if s > best_iou_score and center_distance_ok(det["box"], track_state[tid]["box"]):
                best_iou_score, best_id = s, tid

        if best_id is not None and best_iou_score >= TRACK_IOU_MIN:
            track = track_state[best_id]
            unmatched.discard(best_id)
        else:
            best_id = next_track_id
            next_track_id += 1
            track = {
                "box": det["box"],
                "miss": 0,
            }
            track_state[best_id] = track

        track["box"] = det["box"]
        track["miss"] = 0

        assigned.append({
            "track_id": best_id,
            "roll_no": det["roll_no"],
            "name": det["name"],
            "box": det["box"],
            "occ": det["occ"],
            "distance": det["distance"],
            "gap": det["gap"],
        })

    # Expire tracks that disappeared
    for tid in list(track_state.keys()):
        if tid in unmatched:
            track_state[tid]["miss"] += 1
            if track_state[tid]["miss"] >= HISTORY_LEN:
                del track_state[tid]

    return assigned


# ──────────────────────────────────────────────────────
# SHARED STATE
# ──────────────────────────────────────────────────────
latest_frame        = None
recognition_results = []
is_running          = True
frame_lock          = threading.Lock()
already_marked      = set()
track_confirm_state = {}
active_slot_id      = None
active_class_id     = None

OCC_COLOR = {"clear":(0,200,80), "mask":(0,200,255), "hand":(0,150,255), "heavy":(0,0,200)}
OCC_LABEL = {"clear":"CLEAR", "mask":"MASK", "hand":"HAND", "heavy":"BLOCKED"}


# ──────────────────────────────────────────────────────
# RECOGNITION WORKER  (background thread)
# All DB I/O lives here — video loop stays at 30 FPS.
# ──────────────────────────────────────────────────────
def recognition_worker():
    global latest_frame, recognition_results, is_running, active_slot_id, active_class_id

    while is_running:
        t0 = time.time()

        with frame_lock:
            if latest_frame is None:
                time.sleep(0.02); continue
            frame = latest_frame.copy()

        h, w   = frame.shape[:2]
        small  = cv2.resize(frame, (int(w*FRAME_SCALE), int(h*FRAME_SCALE)))
        sh, sw = small.shape[:2]

        # DNN face detection on downscaled frame
        blob = cv2.dnn.blobFromImage(small, 1.0, (300,300), (104.,177.,123.))
        face_net.setInput(blob)
        dets = face_net.forward()

        boxes, rois, occs, blurs = [], [], [], []
        poses = []

        for i in range(dets.shape[2]):
            conf = float(dets[0,0,i,2])
            if conf < FACE_DNN_CONF: continue

            b           = dets[0,0,i,3:7] * np.array([sw,sh,sw,sh])
            x,y,x2,y2   = (b / FRAME_SCALE).astype(int)
            x,y         = max(0,x), max(0,y)
            x2,y2       = min(w,x2), min(h,y2)

            roi = frame[y:y2, x:x2]
            if roi.size == 0 or min(roi.shape[:2]) < MIN_FACE_PX: continue

            occ = analyze_occlusion(roi)
            blur_var = float(cv2.Laplacian(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())

            # AGGRESSIVE quality gates: skip anything questionable.
            if occ == "heavy" or blur_var < MIN_FACE_SHARPNESS:
                continue

            yaw, pitch = estimate_head_pose(roi)
            # Allow flexibility: only reject extreme angles or if heavily occluded+angled
            if abs(yaw) > MAX_FACE_ANGLE_YAW or abs(pitch) > MAX_FACE_ANGLE_PITCH:
                if occ == "heavy":
                    continue  # reject only if both heavily occluded AND extremely angled

            boxes.append((x,y,x2,y2))
            # Use full face ROI for embedding; partial cropping can destabilize ArcFace matching.
            rois.append(roi)
            occs.append(occ)
            blurs.append(blur_var)
            poses.append((yaw, pitch))

        # ── DB-based recognition (replaces matrix multiplication) ──────────
        identities = db_batch_recognize(rois, occs, blurs, poses) if rois else []
        # ───────────────────────────────────────────────────────────────────

        raw_dets = []
        for b, identity, o in zip(boxes, identities, occs):
            roll_no = None
            name = "Unknown"
            distance = None
            gap = None
            if identity is not None:
                roll_no = identity["roll_no"]
                name = identity["name"]
                distance = identity.get("distance")
                gap = identity.get("match_gap")
            raw_dets.append({
                "box": b,
                "roll_no": roll_no,
                "name": name,
                "occ": o,
                "distance": distance,
                "gap": gap,
            })
        tracked  = update_tracks(raw_dets)

        temp = []
        for det in tracked:
            tid = det["track_id"]
            box = det["box"]
            occ = det["occ"]
            roll_no = det["roll_no"]
            name = det["name"]

            state = track_confirm_state.get(tid)
            if state is None:
                state = {"roll_no": None, "frames": 0, "last_box": box}
                track_confirm_state[tid] = state

            if roll_no:
                if state["roll_no"] == roll_no:
                    state["frames"] += 1
                else:
                    state["roll_no"] = roll_no
                    state["frames"] = 1
                state["last_box"] = box

                if state["frames"] >= CONFIRM_FRAMES and roll_no not in already_marked:
                    # ── DB attendance insert (replaces CSV write) ──────────
                    success = mark_present_for_slot(active_slot_id, active_class_id, roll_no, name)
                    if success:
                        already_marked.add(roll_no)
                        print(f"  [MARKED]  {roll_no} - {name}  ({occ})  → present")
                    else:
                        print(f"  [WARN]    {roll_no} - {name}  ({occ})  → DB insert failed, will retry")
                    # ───────────────────────────────────────────────────────
            else:
                name = "Unknown"
                roll_no = None
                state["roll_no"] = None
                state["frames"] = 0

            confirmed = bool(roll_no and roll_no in already_marked)
            frames = state["frames"] if (roll_no and state["roll_no"] == roll_no) else 0

            temp.append({
                "roll_no":   roll_no,
                "name":      name,
                "box":       box,
                "occ":       occ,
                "confirmed": confirmed,
                "frames":    frames,
            })

        recognition_results = temp
        time.sleep(max(0, (1.0 / RECOGNITION_FPS) - (time.time()-t0)))


def main():
    global latest_frame, is_running, active_slot_id, active_class_id, SIMILARITY_THRESH

    bootstrap_schema()
    print("[INFO] Preparing models and registered embeddings...")
    warmup_models()
    total_registered = load_registered_embeddings()
    print(f"[INFO] Loaded {total_registered} registered student embedding(s) into memory.")
    args = parse_args()

    if not 0.0 < args.threshold <= 1.5:
        print("[ERROR] Threshold must be > 0.0 and <= 1.5")
        sys.exit(1)
    SIMILARITY_THRESH = float(args.threshold)

    try:
        subject_name, lecture_room, faculty_name, start_time, end_time = collect_slot_details(args)
    except ValueError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    active_slot_id, active_class_id = create_attendance_slot(
        subject_name=subject_name,
        lecture_room=lecture_room,
        faculty_name=faculty_name,
        start_time=start_time,
        end_time=end_time,
    )

    print("\n" + "="*64)
    print(f"[SLOT CREATED] ID: {active_slot_id}")
    print(f"[CLASS ID] {active_class_id}")
    print(f"Subject: {subject_name}")
    print(f"Room: {lecture_room} | Faculty: {faculty_name}")
    print(f"Start: {start_time} | End: {end_time}")
    print(f"Match Threshold: {SIMILARITY_THRESH}")
    print(f"Registered Cache: {total_registered} student(s)")
    print("Press Q to complete attendance and close slot.")
    print("="*64 + "\n")

    threading.Thread(target=recognition_worker, daemon=True).start()

    # CAMERA
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("[ERROR] Cannot open webcam. Check camera index.")
        sys.exit(1)

    print("[INFO] Camera opened. Taking attendance now...\n")
    fps_q      = deque(maxlen=30)
    fail_count = 0

    while True:
        t0 = time.time()
        ret, frame = cap.read()

        if not ret:
            fail_count += 1
            if fail_count > 30:
                print("[ERROR] Camera stopped.")
                break
            time.sleep(0.05)
            continue
        fail_count = 0
        frame = cv2.flip(frame, 1)

        with frame_lock:
            latest_frame = frame.copy()

        for r in recognition_results:
            x,y,x2,y2 = r["box"]
            name       = r["name"]
            roll_no    = r.get("roll_no")
            confirmed  = r["confirmed"]
            occ        = r["occ"]
            frames     = r["frames"]

            bc   = (0,220,80) if confirmed else (0,200,255) if name!="Unknown" else (0,0,200)
            oc   = OCC_COLOR.get(occ, (150,150,150))
            olbl = OCC_LABEL.get(occ, "?")

            # Corner bracket
            cl = 24
            for px,py,dx,dy in [(x,y,1,1),(x2,y,-1,1),(x,y2,1,-1),(x2,y2,-1,-1)]:
                cv2.line(frame,(px,py),(px+dx*cl,py),bc,3)
                cv2.line(frame,(px,py),(px,py+dy*cl),bc,3)

            # Name tag
            if name == "Unknown":
                lbl = "Unknown"
            else:
                display = f"{roll_no} - {name}" if roll_no else name
                if confirmed:
                    lbl = f"{display}  MARKED"
                else:
                    bar = "█"*frames + "░"*max(0, CONFIRM_FRAMES-frames)
                    lbl = f"{display}  {bar}"

            (tw,th),_ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)
            cv2.rectangle(frame,(x,y-th-14),(x+tw+10,y),bc,-1)
            cv2.putText(frame,lbl,(x+5,y-5),cv2.FONT_HERSHEY_DUPLEX,0.55,(255,255,255),1,cv2.LINE_AA)

            # Occlusion badge
            (bw,bh),_ = cv2.getTextSize(olbl, cv2.FONT_HERSHEY_DUPLEX, 0.42, 1)
            cv2.rectangle(frame,(x,y2),(x+bw+8,y2+bh+8),oc,-1)
            cv2.putText(frame,olbl,(x+4,y2+bh+3),cv2.FONT_HERSHEY_DUPLEX,0.42,(0,0,0),1,cv2.LINE_AA)

        fps_q.append(time.time()-t0)
        fps = int(1/(sum(fps_q)/len(fps_q))) if fps_q else 0
        cv2.rectangle(frame,(0,0),(760,32),(15,15,15),-1)
        cv2.putText(frame,
            f"Class:{active_class_id}  Slot:{active_slot_id}  FPS:{fps}  Faces:{len(recognition_results)}  Marked:{len(already_marked)}",
            (8,22), cv2.FONT_HERSHEY_DUPLEX, 0.55, (0,230,120), 1, cv2.LINE_AA)

        cv2.imshow("VisiAttend Pro", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            is_running = False
            break

    cap.release()
    cv2.destroyAllWindows()

    present_count, absent_count = complete_attendance_slot(active_slot_id)
    print(
        f"\n[DONE] Class {active_class_id} (slot {active_slot_id}) completed. "
        f"Present: {present_count}, Absent: {absent_count}"
    )


if __name__ == "__main__":
    main()