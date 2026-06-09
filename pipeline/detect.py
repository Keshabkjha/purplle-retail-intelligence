import json
import os
import sys
import uuid
from datetime import datetime, timedelta

import cv2
import numpy as np
import requests

# The above code is importing the `torch` module in Python, which is commonly used for numerical
# computations with support for GPU acceleration.
# The above code is importing the `torch` library in Python, which is commonly used for machine
# learning and deep learning tasks.
import torch
from ultralytics import YOLO

# PyTorch 2.6+ weights_only unpickling workaround for Ultralytics/YOLO models
original_load = torch.load


def patched_load(*args, **kwargs):
    kwargs["weights_only"] = False
    try:
        return original_load(*args, **kwargs)
    except TypeError:
        kwargs.pop("weights_only", None)
        return original_load(*args, **kwargs)


torch.load = patched_load

# Add project root to sys.path to allow direct execution via `python3 pipeline/detect.py`
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from pipeline.adaptive_models import (  # noqa: E402
    AdaptiveModelRegistry,
    build_identity_feature_vector,
    build_staff_feature_vector,
)

# ---------------------------------------------------------------------------
# Layout + Calibration loading (multi-store)
# ---------------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAYOUT_PATH = os.path.join(ROOT_DIR, "config", "store_layout.json")
CALIBRATION_PATH = os.path.join(ROOT_DIR, "config", "calibration.json")
INGEST_URL = os.getenv("INGEST_URL", "http://localhost:8000/events/ingest")

# Will be resolved per video
store_id = os.getenv("STORE_ID", "ST1008")
store_code = os.getenv("STORE_CODE", "store_1008")
zones: list = []
cameras_mapping: dict = {}
camera_roles: dict = {}

# Multi-store layout map: store_id -> {zones, cameras, camera_roles, ...}
STORE_LAYOUT_MAP: dict = {}
# Camera ID -> store_id map (for quick lookups)
CAM_TO_STORE: dict = {}

if os.path.exists(LAYOUT_PATH):
    try:
        with open(LAYOUT_PATH, "r") as f:
            layout_data = json.load(f)

        # Support both old (single store) and new (multi-store) formats
        if "stores" in layout_data:
            for store_entry in layout_data["stores"]:
                sid = store_entry["store_id"]
                STORE_LAYOUT_MAP[sid] = store_entry
                for cam_id in store_entry.get("cameras", {}).keys():
                    CAM_TO_STORE[cam_id] = sid
            # Default store
            default_sid = layout_data.get("default_store_id", list(STORE_LAYOUT_MAP.keys())[0])
            if default_sid in STORE_LAYOUT_MAP:
                store_id = default_sid
                store_code = STORE_LAYOUT_MAP[default_sid].get("store_code", store_id.lower())
                zones = STORE_LAYOUT_MAP[default_sid].get("zones", [])
                cameras_mapping = STORE_LAYOUT_MAP[default_sid].get("cameras", {})
                camera_roles = STORE_LAYOUT_MAP[default_sid].get("camera_roles", {})
        else:
            # Legacy single-store format
            store_id = layout_data.get("store_id", "ST1008")
            store_code = layout_data.get("store_code", store_id.lower())
            zones = layout_data.get("zones", [])
            cameras_mapping = layout_data.get("cameras", {})
            camera_roles = layout_data.get("camera_roles", {})
    except Exception as e:
        print(f"Error loading layout: {e}")

# Load calibration
calibration = {}
if os.path.exists(CALIBRATION_PATH):
    try:
        with open(CALIBRATION_PATH, "r") as f:
            calibration = json.load(f)
    except Exception as e:
        print(f"Error loading calibration: {e}")

MODEL_REGISTRY = AdaptiveModelRegistry()


# ---------------------------------------------------------------------------
# Camera-ID resolution from filename
# ---------------------------------------------------------------------------

def resolve_camera_id_and_store(video_path: str) -> tuple[str, str, str, list, dict, str]:
    """
    Given a video path, resolve:
    - camera_id  (matches keys in store_layout.json cameras dict)
    - store_id
    - store_code
    - zones list
    - camera_roles dict
    - camera_role ('entry' | 'zone' | 'billing')

    Handles actual filenames:
      Store 1: CAM 1 - zone.mp4, CAM 2 - zone.mp4, CAM 3 - entry.mp4, CAM 5 - billing.mp4
      Store 2: entry 1.mp4, entry 2.mp4, zone.mp4, billing_area.mp4
    """
    base_name = os.path.basename(video_path).lower()
    parent_dir = os.path.basename(os.path.dirname(video_path))

    # Determine which store based on parent folder
    if "store 1" in parent_dir.lower() or "store1" in parent_dir.lower():
        sid = "ST1008"
    elif "store 2" in parent_dir.lower() or "store2" in parent_dir.lower():
        sid = "ST1076"
    else:
        # Fall back to env or default
        sid = os.getenv("STORE_ID", "ST1008")

    store_entry = STORE_LAYOUT_MAP.get(sid, {})
    s_code = store_entry.get("store_code", sid.lower())
    s_zones = store_entry.get("zones", [])
    s_camera_roles = store_entry.get("camera_roles", {})
    cameras = store_entry.get("cameras", {})

    # Map filename -> camera_id by matching against cameras dict values
    cam_id = None
    for cid, filename in cameras.items():
        if filename.lower() == os.path.basename(video_path).lower():
            cam_id = cid
            break

    # Fallback: infer camera_id from filename patterns
    if cam_id is None:
        if "cam 3" in base_name or "entry" in base_name and "1" in base_name:
            # Store 1 entry or Store 2 entry 1
            if sid == "ST1008":
                cam_id = "cam1"
            else:
                cam_id = "cam1" if "1" in base_name else "cam2"
        elif "cam 1" in base_name:
            cam_id = "CAM1"
        elif "cam 2" in base_name or ("zone" in base_name and "cam" not in base_name):
            cam_id = "CAM2" if sid == "ST1008" else "CAM2"
        elif "cam 5" in base_name or "billing" in base_name:
            cam_id = "CAM5" if sid == "ST1008" else "PURPLLE_MUM_1076_CAM6"
        elif "entry 2" in base_name or "entry2" in base_name:
            cam_id = "cam2"
        else:
            cam_id = "CAM_UNKNOWN"

    role = s_camera_roles.get(cam_id, "zone")
    return cam_id, sid, s_code, s_zones, s_camera_roles, role


# ---------------------------------------------------------------------------
# Age bucket helper
# ---------------------------------------------------------------------------

def age_to_bucket(age: int) -> str:
    if age < 18:
        return "18-24"
    elif age < 25:
        return "18-24"
    elif age < 35:
        return "25-34"
    elif age < 45:
        return "35-44"
    elif age < 55:
        return "45-54"
    else:
        return "55+"


# ---------------------------------------------------------------------------
# Spatial helpers
# ---------------------------------------------------------------------------

def point_in_polygon(x, y, polygon):
    """Ray casting algorithm for point-in-polygon test."""
    num_vertices = len(polygon)
    inside = False
    p1 = polygon[0]
    for i in range(1, num_vertices + 1):
        p2 = polygon[i % num_vertices]
        if y > min(p1[1], p2[1]):
            if y <= max(p1[1], p2[1]):
                if x <= max(p1[0], p2[0]):
                    if p1[1] != p2[1]:
                        xinters = (y - p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1]) + p1[0]
                        if x <= xinters:
                            inside = not inside
        p1 = p2
    return inside


def map_camera_to_floor(px, py, camera_id, frame_w=1920, frame_h=1080):
    """Maps coordinates from camera frame to floor plan using Homography or simple scaling."""
    if camera_id in calibration:
        try:
            pts_src = np.array(calibration[camera_id]["src"], dtype=np.float32)
            pts_dst = np.array(calibration[camera_id]["dst"], dtype=np.float32)
            H, _ = cv2.findHomography(pts_src, pts_dst)
            point = np.array([[[px, py]]], dtype=np.float32)
            warped = cv2.perspectiveTransform(point, H)[0][0]
            return float(warped[0]), float(warped[1])
        except Exception as e:
            print(f"Homography failed for {camera_id}: {e}")

    # Fallback linear scaling: map to 940x470 floor plan coords
    scale_x = 940.0 / frame_w
    scale_y = 470.0 / frame_h

    if camera_id in ("cam1",):
        # Entry camera — left side of store
        wx = px * (130.0 / frame_w)
        wy = 150.0 + py * (280.0 / frame_h)
    elif camera_id == "CAM1":
        # Zone camera 1 — left half of store
        wx = px * (470.0 / frame_w)
        wy = py * (470.0 / frame_h)
    elif camera_id == "CAM2":
        # Zone camera 2 — right half of store
        wx = 470.0 + px * (470.0 / frame_w)
        wy = py * (470.0 / frame_h)
    elif camera_id in ("CAM5", "PURPLLE_MUM_1076_CAM6"):
        # Billing camera — far right
        wx = 820.0 + px * (120.0 / frame_w)
        wy = 100.0 + py * (360.0 / frame_h)
    elif camera_id == "cam2":
        # Store 2 entry 2
        wx = px * (200.0 / frame_w)
        wy = 500.0 + py * (120.0 / frame_h)
    else:
        wx = px * scale_x
        wy = py * scale_y

    return wx, wy


def determine_zone(wx, wy, camera_id, active_zones):
    """Finds which zone contains the warped floor coordinates."""
    # Camera-specific zones first
    for zone in active_zones:
        if zone.get("camera_id") == camera_id:
            coords = zone["polygon_coords"]
            if point_in_polygon(wx, wy, coords):
                return zone["zone_id"], zone.get("zone_name", zone["zone_id"]), zone.get("zone_type", "SHELF"), zone.get("is_revenue_zone", "Yes")

    # Global fallback
    for zone in active_zones:
        coords = zone["polygon_coords"]
        if point_in_polygon(wx, wy, coords):
            return zone["zone_id"], zone.get("zone_name", zone["zone_id"]), zone.get("zone_type", "SHELF"), zone.get("is_revenue_zone", "Yes")

    return None, None, None, None


# ---------------------------------------------------------------------------
# Staff detection
# ---------------------------------------------------------------------------

def is_staff_heuristic(track_id, bbox, frame):
    """Detect staff by analyzing uniform color (black/dark grey) in the upper torso."""
    try:
        x1, y1, x2, y2 = map(int, bbox)
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        torso_y2 = y1 + max(1, (y2 - y1) // 2)
        torso = frame[y1:torso_y2, x1:x2]
        if torso.size == 0:
            return False

        hsv_torso = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 50])
        mask = cv2.inRange(hsv_torso, lower_black, upper_black)
        match_ratio = cv2.countNonZero(mask) / (torso.shape[0] * torso.shape[1] + 1e-6)
        return match_ratio > 0.30
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Appearance embedding for Re-ID
# ---------------------------------------------------------------------------

def compute_appearance_embedding(crop):
    """Extract a lightweight normalized HSV color histogram as an appearance embedding."""
    try:
        if crop is None or crop.size == 0:
            return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten().tolist()
    except Exception:
        return None


def compare_appearance(hist1, hist2):
    """Compare two appearance embeddings using correlation."""
    if not hist1 or not hist2:
        return 0.0
    try:
        h1 = np.array(hist1, dtype=np.float32)
        h2 = np.array(hist2, dtype=np.float32)
        h1_norm = h1 - np.mean(h1)
        h2_norm = h2 - np.mean(h2)
        denom = np.sqrt(np.sum(h1_norm**2) * np.sum(h2_norm**2))
        if denom < 1e-6:
            return 0.0
        return float(np.sum(h1_norm * h2_norm) / denom)
    except Exception:
        return 0.0


def camera_transition_prior(previous_camera_id, current_camera_id):
    """Return a soft prior for whether a camera transition is operationally plausible."""
    if not previous_camera_id or not current_camera_id:
        return 0.0
    if previous_camera_id == current_camera_id:
        return 1.0

    transition_map = {
        "cam1": {"CAM1": 0.95, "CAM2": 0.90, "CAM5": 0.75, "PURPLLE_MUM_1076_CAM6": 0.75},
        "cam2": {"CAM1": 0.90, "CAM2": 0.90, "CAM5": 0.75, "PURPLLE_MUM_1076_CAM6": 0.75},
        "CAM1": {"cam1": 0.90, "CAM2": 0.88, "CAM5": 0.85},
        "CAM2": {"cam1": 0.90, "CAM1": 0.88, "CAM5": 0.85, "PURPLLE_MUM_1076_CAM6": 0.85},
        "CAM5": {"cam1": 0.80, "CAM1": 0.90, "CAM2": 0.90},
        "PURPLLE_MUM_1076_CAM6": {"cam1": 0.80, "cam2": 0.80, "CAM2": 0.90},
    }
    return transition_map.get(previous_camera_id, {}).get(current_camera_id, 0.60)


def zone_transition_prior(previous_zone_id, current_zone_id):
    """Return a soft prior for whether a zone transition is plausible."""
    if not previous_zone_id or not current_zone_id:
        return 0.50
    if previous_zone_id == current_zone_id:
        return 1.0
    if "ENTRY" in (previous_zone_id or "").upper() or "ENTRY" in (current_zone_id or "").upper():
        return 0.85
    if "BILLING" in (previous_zone_id or "").upper() or "BILLING" in (current_zone_id or "").upper():
        return 0.90
    return 0.80


# ---------------------------------------------------------------------------
# Cross-camera session tracker
# ---------------------------------------------------------------------------

class CrossCameraSessionTracker:
    def __init__(self, state_file="pipeline/session_state.json"):
        self.state_file = state_file
        self.sessions = {}
        self.local_to_unified = {}
        self.load_state()

    def load_state(self):
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r") as f:
                    data = json.load(f)
                    self.sessions = data.get("visitor_sessions", {})
            except Exception as e:
                print(f"Error loading cross-camera state: {e}")
                self.sessions = {}
        else:
            self.sessions = {}

    def save_state(self):
        try:
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
            with open(self.state_file, "w") as f:
                json.dump({"visitor_sessions": self.sessions}, f, indent=2)
        except Exception as e:
            print(f"Error saving cross-camera state: {e}")

    def is_reentry(self, unified_id, current_camera_id):
        if unified_id in self.sessions:
            sess = self.sessions[unified_id]
            for cam in sess.get("camera_track_ids", {}):
                if cam != current_camera_id:
                    return True
        return False

    def get_unified_id(
        self, track_id, camera_id, wx, wy, current_time, is_staff_initial,
        box=None, frame=None, zone_id=None, current_track_ids=None,
    ):
        """Maps a local camera track_id to a globally consistent unified visitor_id."""
        track_id = int(track_id)
        wx = float(wx)
        wy = float(wy)
        track_key = (camera_id, track_id)

        emb = None
        if box is not None and frame is not None:
            try:
                x1, y1, x2, y2 = map(int, box)
                h, w = frame.shape[:2]
                crop = frame[max(0, y1): min(h, y2), max(0, x1): min(w, x2)]
                if crop is not None and crop.size > 0:
                    torso_h = max(1, int(crop.shape[0] * 0.65))
                    crop = crop[:torso_h, :]
                emb = compute_appearance_embedding(crop)
            except Exception:
                pass

        if track_key in self.local_to_unified:
            unified_id = self.local_to_unified[track_key]
            if unified_id in self.sessions:
                sess = self.sessions[unified_id]
                sess["last_seen_camera"] = camera_id
                sess["last_seen_time"] = current_time.strftime("%Y-%m-%dT%H:%M:%SZ")
                sess["last_seen_x"] = wx
                sess["last_seen_y"] = wy
                sess["camera_track_ids"][camera_id] = track_id
                if zone_id is not None:
                    sess["last_zone_id"] = zone_id
                if emb is not None:
                    old_emb = sess.get("appearance_embedding")
                    if old_emb is not None:
                        updated = (0.7 * np.array(old_emb) + 0.3 * np.array(emb)).tolist()
                        sess["appearance_embedding"] = updated
                    else:
                        sess["appearance_embedding"] = emb
                    self.save_state()
            return unified_id

        matched_id = None
        max_match_score = -1.0
        best_dist = float("inf")
        best_diff = 0.0
        candidate_observations = []

        for uid, sess in self.sessions.items():
            if sess["last_seen_camera"] == camera_id:
                prev_track_id = sess.get("camera_track_ids", {}).get(camera_id)
                if prev_track_id is not None and current_track_ids is not None:
                    if int(prev_track_id) in current_track_ids:
                        continue
                else:
                    continue

            last_time_str = sess["last_seen_time"]
            try:
                last_time = datetime.strptime(last_time_str, "%Y-%m-%dT%H:%M:%SZ")
            except ValueError:
                continue

            diff = abs((current_time - last_time).total_seconds())
            if diff <= 30.0:
                lx, ly = sess["last_seen_x"], sess["last_seen_y"]
                dist = np.sqrt((wx - lx) ** 2 + (wy - ly) ** 2)

                if dist <= 150.0:
                    spatial_score = 1.0 - (dist / 150.0)
                    temporal_score = 1.0 - (diff / 30.0)
                    app_score = 0.5
                    if emb is not None and sess.get("appearance_embedding") is not None:
                        app_score = compare_appearance(emb, sess["appearance_embedding"])
                    cam_score = camera_transition_prior(sess["last_seen_camera"], camera_id)
                    zone_score = zone_transition_prior(sess.get("last_zone_id"), zone_id)

                    feature_vec = build_identity_feature_vector(
                        spatial_score=spatial_score,
                        temporal_score=temporal_score,
                        visual_score=app_score,
                        camera_score=cam_score,
                        zone_score=zone_score,
                        dist_norm=dist / 150.0,
                        time_norm=diff / 30.0,
                    )

                    heuristic_score = (
                        spatial_score * 0.30
                        + temporal_score * 0.22
                        + app_score * 0.26
                        + cam_score * 0.12
                        + zone_score * 0.10
                    )
                    learned_prob = MODEL_REGISTRY.predict_identity_probability(
                        feature_vec, fallback=heuristic_score
                    )
                    match_score = 0.58 * heuristic_score + 0.42 * learned_prob
                    candidate_observations.append(
                        (uid, feature_vec, match_score, heuristic_score, learned_prob)
                    )

                    if match_score >= 0.65 and match_score > max_match_score:
                        max_match_score = match_score
                        matched_id = uid

        if matched_id:
            print(f"🔗 [Re-ID] Track {track_id} in {camera_id} → {matched_id} (score: {max_match_score:.2f})")
            for (uid, feature_vec, match_score, _, _) in candidate_observations:
                if uid == matched_id and match_score >= 0.65:
                    MODEL_REGISTRY.identity_model.update(feature_vec, 1)
                elif match_score >= 0.60 and uid != matched_id:
                    MODEL_REGISTRY.identity_model.update(feature_vec, 0)
            self.local_to_unified[track_key] = matched_id
            sess = self.sessions[matched_id]
            sess["last_seen_camera"] = camera_id
            sess["last_seen_time"] = current_time.strftime("%Y-%m-%dT%H:%M:%SZ")
            sess["last_seen_x"] = wx
            sess["last_seen_y"] = wy
            sess["camera_track_ids"][camera_id] = track_id
            if zone_id is not None:
                sess["last_zone_id"] = zone_id
            if emb is not None:
                old_emb = sess.get("appearance_embedding")
                sess["appearance_embedding"] = (
                    (0.7 * np.array(old_emb) + 0.3 * np.array(emb)).tolist()
                    if old_emb else emb
                )
            self.save_state()
            return matched_id
        else:
            if candidate_observations:
                top_candidate = max(candidate_observations, key=lambda item: item[2])
                if top_candidate[2] >= 0.60:
                    MODEL_REGISTRY.identity_model.update(top_candidate[1], 0)

            # Build a Purplle-style ID token: ID_XXXXX
            new_id = f"ID_{60000 + track_id}"
            base_new_id = new_id
            counter = 1
            while new_id in self.sessions:
                new_id = f"{base_new_id}_{counter}"
                counter += 1

            self.local_to_unified[track_key] = new_id
            self.sessions[new_id] = {
                "unified_id": new_id,
                "last_seen_camera": camera_id,
                "last_seen_time": current_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "last_seen_x": wx,
                "last_seen_y": wy,
                "is_staff": is_staff_initial,
                "last_zone_id": zone_id,
                "appearance_embedding": emb,
                "camera_track_ids": {camera_id: track_id},
            }
            self.save_state()
            return new_id


# ---------------------------------------------------------------------------
# Staff status hybrid classifier
# ---------------------------------------------------------------------------

def update_staff_status(vid, sess, current_time, zone_id, wx, wy, is_clothing_staff, tracker):
    """Hybrid visual-behavioral staff classifier with a lightweight learned model."""
    unified_is_staff_init = False
    camera_count = 1
    if tracker and vid in tracker.sessions:
        unified_is_staff_init = tracker.sessions[vid].get("is_staff", False)
        camera_count = len(tracker.sessions[vid].get("camera_track_ids", {}))

    staff_score = 0.6 if is_clothing_staff or unified_is_staff_init else 0.0

    is_billing_zone = zone_id and "BILLING" in (zone_id or "").upper()
    if is_billing_zone and wx > 820:
        staff_score += 0.4
    if is_billing_zone:
        billing_duration = (current_time - sess["enter_time"]).total_seconds()
        if billing_duration > 90:
            staff_score += 0.3

    torso_match_ratio = 1.0 if is_clothing_staff else 0.0
    total_duration = (current_time - sess.get("first_seen_time", sess["enter_time"])).total_seconds()
    if total_duration > 180:
        staff_score += 0.3
    if camera_count >= 2:
        staff_score += 0.2

    heuristic_prob = min(max(staff_score / 1.8, 0.0), 1.0)
    feature_vec = build_staff_feature_vector(
        torso_match_ratio=torso_match_ratio,
        is_clothing_staff=is_clothing_staff,
        zone_id=zone_id or "",
        wx=wx,
        billing_duration_sec=(current_time - sess["enter_time"]).total_seconds() if is_billing_zone else 0.0,
        total_duration_sec=total_duration,
        camera_count=camera_count,
    )
    learned_prob = MODEL_REGISTRY.predict_staff_probability(feature_vec, fallback=heuristic_prob)
    combined_prob = (0.55 * learned_prob) + (0.45 * heuristic_prob)

    if MODEL_REGISTRY.staff_model.update_count < 5:
        if staff_score >= 0.6:
            is_staff = True
        elif staff_score < 0.4:
            is_staff = False
        else:
            is_staff = total_duration > 120 or camera_count >= 2
    else:
        if combined_prob >= 0.62:
            is_staff = True
        elif combined_prob < 0.38:
            is_staff = False
        else:
            is_staff = total_duration > 120 or camera_count >= 2

    if combined_prob >= 0.80 or staff_score >= 1.2:
        MODEL_REGISTRY.staff_model.update(feature_vec, 1)
    elif combined_prob <= 0.20 or staff_score <= 0.15:
        MODEL_REGISTRY.staff_model.update(feature_vec, 0)

    if tracker and vid in tracker.sessions:
        tracker.sessions[vid]["is_staff"] = is_staff
        tracker.save_state()

    return is_staff


# ---------------------------------------------------------------------------
# Event emission helpers
# ---------------------------------------------------------------------------

def post_event(event):
    """Posts a structured event to the FastAPI ingest endpoint."""
    try:
        response = requests.post(INGEST_URL, json=[event], timeout=2)
        if response.status_code == 207:
            body = response.json()
            if body.get("ingested", 0) > 0:
                raw_vid = event.get('visitor_id', event.get('id_token', '?'))
                # Mask PII: show only first 4 chars + *** to prevent clear-text logging of Re-ID tokens
                masked_vid = (raw_vid[:4] + "***") if isinstance(raw_vid, str) and len(raw_vid) > 4 else "****"
                print(f"✅ {event['event_type']} for {masked_vid} ingested.")
        else:
            print(f"❌ Failed ingestion ({response.status_code}): {response.text[:120]}")
    except Exception as e:
        print(f"Connection error to ingest API: {e}")


def make_entry_exit_event(event_type, vid, s_code, sid, camera_id, ts_str,
                          is_staff, conf, group_id=None, group_size=None,
                          gender="unknown", age=None, is_face_hidden=False, sess_seq=1):
    """Emit a Purplle-format entry/exit event matching sample_events.jsonl."""
    age_bucket = age_to_bucket(age) if age is not None else "unknown"
    return {
        "event_type": event_type,           # 'entry' or 'exit'
        "id_token": vid,                    # e.g. ID_60001
        "store_code": s_code,              # e.g. store_1008
        "camera_id": camera_id,
        "event_timestamp": ts_str,
        "is_staff": is_staff,
        "gender_pred": gender,
        "age_pred": age,
        "age_bucket": age_bucket,
        "is_face_hidden": is_face_hidden,
        "group_id": group_id,
        "group_size": group_size,
        # Also include scoring-harness compatible fields for /events/ingest
        "event_id": str(uuid.uuid4()),
        "store_id": sid,
        "visitor_id": vid,
        "timestamp": ts_str,
        "zone_id": None,
        "dwell_ms": 0,
        "confidence": float(conf),
        "metadata": {"queue_depth": None, "sku_zone": None, "session_seq": int(sess_seq)},
    }


def make_zone_event(event_type, track_id_int, vid, sid, camera_id, ts_str,
                    zone_id, zone_name, zone_type, is_revenue_zone,
                    hotspot_x, hotspot_y, is_staff, conf, dwell_ms=0,
                    gender="unknown", age=None, sess_seq=1):
    """Emit a Purplle-format zone_entered/zone_exited event."""
    age_bucket = age_to_bucket(age) if age is not None else "unknown"
    return {
        "event_type": event_type,           # 'zone_entered' or 'zone_exited'
        "track_id": track_id_int,
        "store_id": sid,
        "camera_id": camera_id,
        "zone_id": zone_id,
        "zone_name": zone_name or zone_id,
        "zone_type": zone_type or "SHELF",
        "is_revenue_zone": is_revenue_zone or "Yes",
        "event_time": ts_str,
        "zone_hotspot_x": round(hotspot_x, 1),
        "zone_hotspot_y": round(hotspot_y, 1),
        "gender": gender,
        "age": age,
        "age_bucket": age_bucket,
        "is_staff": is_staff,
        # Scoring-harness compat fields
        "event_id": str(uuid.uuid4()),
        "visitor_id": vid,
        "timestamp": ts_str,
        "dwell_ms": dwell_ms,
        "confidence": float(conf),
        "metadata": {"queue_depth": None, "sku_zone": zone_id, "session_seq": sess_seq},
    }


def make_queue_event(event_type, track_id_int, vid, sid, camera_id,
                     zone_id, zone_name, queue_join_ts, queue_served_ts,
                     queue_exit_ts, wait_seconds, queue_position, abandoned,
                     hotspot_x, hotspot_y, gender="unknown", age=None, sess_seq=1):
    """Emit a single Purplle queue lifecycle event (queue_completed / queue_abandoned)."""
    age_bucket = age_to_bucket(age) if age is not None else "unknown"
    return {
        "queue_event_id": str(uuid.uuid4()),
        "event_type": event_type,           # 'queue_completed' or 'queue_abandoned'
        "track_id": track_id_int,
        "store_id": sid,
        "camera_id": camera_id,
        "zone_id": zone_id,
        "zone_name": zone_name or "Billing Counter Queue",
        "zone_type": "BILLING",
        "is_revenue_zone": "Yes",
        "queue_join_ts": queue_join_ts,
        "queue_served_ts": queue_served_ts,
        "queue_exit_ts": queue_exit_ts,
        "wait_seconds": int(wait_seconds),
        "queue_position_at_join": int(queue_position),
        "abandoned": abandoned,
        "zone_hotspot_x": round(hotspot_x, 1),
        "zone_hotspot_y": round(hotspot_y, 1),
        "gender": gender,
        "age": age,
        "age_bucket": age_bucket,
        "is_staff": False,
        # Scoring-harness compat
        "event_id": str(uuid.uuid4()),
        "visitor_id": vid,
        "timestamp": queue_exit_ts,
        "dwell_ms": int(wait_seconds * 1000),
        "confidence": 0.88,
        "metadata": {
            "queue_depth": queue_position,
            "sku_zone": "BILLING",
            "session_seq": int(sess_seq),
        },
        "event_type_legacy": "BILLING_QUEUE_ABANDON" if abandoned else "BILLING_QUEUE_JOIN",
    }


# ---------------------------------------------------------------------------
# Main detection loop
# ---------------------------------------------------------------------------

def run_detection(video_path: str, model_path: str = "yolo11n.pt"):
    print(f"Initializing YOLO11 model: {model_path}")
    model = YOLO(model_path)

    if not os.path.exists(video_path):
        print(f"Error: Video file {video_path} does not exist.")
        return

    base_name = os.path.basename(video_path)

    # Resolve camera + store from path
    camera_id, sid, s_code, active_zones, s_camera_roles, cam_role = resolve_camera_id_and_store(video_path)
    print(f"📹 Camera: {camera_id} | Store: {sid} ({s_code}) | Role: {cam_role}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    print(f"Processing {video_path} | FPS: {fps} | Frames: {frame_count} | {width}x{height}")

    # Target resolution for fast inference
    target_w = 640
    target_h = int(height * (target_w / width))
    scale_x = width / target_w
    scale_y = height / target_h

    frame_skip = 3
    out_fps = fps / frame_skip

    out_path = f"annotated_{base_name}"
    temp_out_path = f"temp_annotated_{base_name}"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(temp_out_path, fourcc, out_fps, (target_w, target_h))
    print(f"Annotated output: {temp_out_path}")

    # Reset cross-camera state on entry camera
    state_file = "pipeline/session_state.json"
    if cam_role == "entry":
        if os.path.exists(state_file):
            try:
                os.remove(state_file)
                print(f"↺ Resetting cross-camera session state: {state_file}")
            except Exception as e:
                print(f"Error resetting state file: {e}")

    tracker = CrossCameraSessionTracker(state_file)

    # Session states: visitor_id -> {current_zone, enter_time, ...}
    active_sessions = {}
    # Historical exits for REENTRY detection
    historical_exits = {}
    # Billing queue state: visitor_id -> {join_ts, position, hotspot_x, hotspot_y}
    billing_queue: dict = {}
    billing_queue_depth = 0

    # Align clip start time to POS transaction window (Brigade Road, April 10 2026)
    base_time = datetime(2026, 4, 10, 12, 0, 0)

    frame_num = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1

        if frame_num % 5 == 0 or frame_num == frame_count:
            try:
                with open("pipeline/simulation_progress.json", "w") as f:
                    json.dump({
                        "video": base_name,
                        "camera_id": camera_id,
                        "store_id": sid,
                        "frame": frame_num,
                        "total": frame_count if frame_count > 0 else frame_num,
                        "percent": int((frame_num / frame_count) * 100) if frame_count > 0 else 0,
                        "status": "running"
                    }, f)
            except Exception:
                pass

        if frame_num % frame_skip != 0:
            continue

        frame = cv2.resize(frame, (target_w, target_h))

        offset_seconds = frame_num / fps
        current_time = base_time + timedelta(seconds=offset_seconds)
        ts_str = current_time.strftime("%Y-%m-%dT%H:%M:%SZ")

        results = model.track(frame, persist=True, verbose=False)

        seen_tracks = set()
        seen_vids = set()
        if results and results[0].boxes and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()

            for box, track_id, cls, conf in zip(boxes, track_ids, classes, confidences):
                if cls == 0:  # Person class
                    seen_tracks.add(track_id)
                    x1, y1, x2, y2 = box
                    px = (x1 + x2) / 2.0
                    py = y2  # foot position

                    wx, wy = map_camera_to_floor(px * scale_x, py * scale_y, camera_id, width, height)
                    zone_id, zone_name, zone_type, is_revenue_zone = determine_zone(wx, wy, camera_id, active_zones)
                    is_staff_init = is_staff_heuristic(track_id, box, frame)

                    vid = tracker.get_unified_id(
                        track_id, camera_id, wx, wy, current_time,
                        is_staff_init, box=box, frame=frame,
                        zone_id=zone_id, current_track_ids=set(track_ids),
                    )
                    seen_vids.add(vid)

                    is_staff_display = is_staff_init
                    if vid in tracker.sessions:
                        is_staff_display = tracker.sessions[vid].get("is_staff", is_staff_init)

                    # Draw bounding box
                    color = (255, 0, 255) if is_staff_display else (0, 255, 0)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    label = f"{vid} | {zone_id or 'FOH'}"
                    if is_staff_display:
                        label += " [STAFF]"
                    cv2.putText(frame, label, (int(x1), max(10, int(y1) - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

                    if frame_num % 15 == 0:
                        if vid not in active_sessions:
                            active_sessions[vid] = {
                                "current_zone": None,
                                "enter_time": current_time,
                                "first_seen_time": current_time,
                                "last_seen": current_time,
                                "dwell_sent_count": 0,
                                "seq": 1,
                                "is_staff": is_staff_init,
                                "track_id_int": int(track_id),
                                "last_hotspot_x": wx,
                                "last_hotspot_y": wy,
                                "gender": "unknown",
                                "age": None,
                            }

                            is_reentry = (vid in historical_exits) or tracker.is_reentry(vid, camera_id)
                            is_staff = update_staff_status(
                                vid, active_sessions[vid], current_time, zone_id, wx, wy, is_staff_init, tracker
                            )
                            active_sessions[vid]["is_staff"] = is_staff

                            # Emit entry or reentry event (Purplle native format)
                            if cam_role == "entry":
                                evt_type = "REENTRY" if is_reentry else "ENTRY"
                                entry_evt = make_entry_exit_event(
                                    evt_type, vid, s_code, sid, camera_id, ts_str,
                                    is_staff, float(conf), sess_seq=active_sessions[vid]["seq"],
                                )
                                post_event(entry_evt)

                        sess = active_sessions[vid]
                        sess["last_seen"] = current_time
                        sess["last_hotspot_x"] = wx
                        sess["last_hotspot_y"] = wy
                        is_staff = update_staff_status(vid, sess, current_time, zone_id, wx, wy, is_staff_init, tracker)
                        sess["is_staff"] = is_staff

                        # Zone change detection
                        old_zone = sess["current_zone"]
                        if zone_id != old_zone:
                            # Exit old zone
                            if old_zone:
                                sess["seq"] += 1
                                old_zone_meta = next((z for z in active_zones if z["zone_id"] == old_zone), {})
                                dwell_ms = int((current_time - sess["enter_time"]).total_seconds() * 1000)

                                # Handle billing queue lifecycle
                                is_billing_exit = old_zone_meta.get("zone_type") == "BILLING"
                                if is_billing_exit and vid in billing_queue:
                                    bq = billing_queue.pop(vid)
                                    billing_queue_depth = max(0, billing_queue_depth - 1)
                                    wait_secs = (current_time - bq["join_time"]).total_seconds()
                                    # Abandoned if dwell < 60s in billing and didn't reach counter
                                    abandoned = wait_secs < 60 or zone_id not in (None, "EXIT", "ENTRY")
                                    q_evt = make_queue_event(
                                        "queue_abandoned" if abandoned else "queue_completed",
                                        sess["track_id_int"], vid, sid, camera_id,
                                        old_zone, old_zone_meta.get("zone_name", "Billing Counter Queue"),
                                        bq["join_ts"], None if abandoned else ts_str, ts_str,
                                        wait_secs, bq["position"], abandoned,
                                        bq["hotspot_x"], bq["hotspot_y"],
                                        sess.get("gender", "unknown"), sess.get("age"), sess_seq=sess["seq"],
                                    )
                                    post_event(q_evt)
                                elif cam_role == "zone":
                                    # Emit zone_exited
                                    zone_exit_evt = make_zone_event(
                                        "zone_exited", sess["track_id_int"], vid, sid, camera_id, ts_str,
                                        old_zone, old_zone_meta.get("zone_name", old_zone),
                                        old_zone_meta.get("zone_type", "SHELF"),
                                        old_zone_meta.get("is_revenue_zone", "Yes"),
                                        sess["last_hotspot_x"], sess["last_hotspot_y"],
                                        is_staff, float(conf), dwell_ms, sess_seq=sess["seq"],
                                    )
                                    post_event(zone_exit_evt)

                            # Enter new zone
                            sess["current_zone"] = zone_id
                            sess["enter_time"] = current_time
                            sess["dwell_sent_count"] = 0

                            if zone_id and cam_role == "zone":
                                sess["seq"] += 1
                                zone_meta = next((z for z in active_zones if z["zone_id"] == zone_id), {})

                                # Billing queue join tracking
                                if zone_meta.get("zone_type") == "BILLING":
                                    billing_queue_depth += 1
                                    billing_queue[vid] = {
                                        "join_time": current_time,
                                        "join_ts": ts_str,
                                        "position": billing_queue_depth,
                                        "hotspot_x": wx,
                                        "hotspot_y": wy,
                                    }

                                    enter_evt = make_zone_event(
                                        "zone_entered", sess["track_id_int"], vid, sid, camera_id, ts_str,
                                        zone_id, zone_meta.get("zone_name", zone_id),
                                        zone_meta.get("zone_type", "SHELF"),
                                        zone_meta.get("is_revenue_zone", "Yes"),
                                        wx, wy, is_staff, float(conf), 0, sess_seq=sess["seq"],
                                    )
                                    post_event(enter_evt)

                        else:
                            # Same zone — emit ZONE_DWELL every 30s (scoring harness compat)
                            if zone_id and cam_role == "zone":
                                duration = (current_time - sess["enter_time"]).total_seconds()
                                expected_dwells = int(duration // 30)
                                if expected_dwells > sess["dwell_sent_count"]:
                                    sess["dwell_sent_count"] = expected_dwells
                                    sess["seq"] += 1
                                    zone_meta = next((z for z in active_zones if z["zone_id"] == zone_id), {})
                                    dwell_evt = {
                                        "event_id": str(uuid.uuid4()),
                                        "store_id": sid,
                                        "camera_id": camera_id,
                                        "visitor_id": vid,
                                        "event_type": "ZONE_DWELL",
                                        "timestamp": ts_str,
                                        "zone_id": zone_id,
                                        "dwell_ms": int(duration * 1000),
                                        "is_staff": sess["is_staff"],
                                        "confidence": float(conf),
                                        "metadata": {
                                            "queue_depth": None,
                                            "sku_zone": zone_id,
                                            "session_seq": sess["seq"],
                                        },
                                    }
                                    post_event(dwell_evt)

        # Check for completed sessions (not seen for > 15 seconds)
        if frame_num % 15 == 0:
            to_remove = []
            for vid, sess in active_sessions.items():
                if vid not in seen_vids and (current_time - sess["last_seen"]).total_seconds() > 15:
                    to_remove.append(vid)
                    sess["seq"] += 1

                    # Handle any open billing queue sessions
                    if vid in billing_queue:
                        bq = billing_queue.pop(vid)
                        billing_queue_depth = max(0, billing_queue_depth - 1)
                        wait_secs = (current_time - bq["join_time"]).total_seconds()
                        q_evt = make_queue_event(
                            "queue_completed",
                            sess["track_id_int"], vid, sid, camera_id,
                            sess["current_zone"] or "BILLING", "Billing Counter Queue",
                            bq["join_ts"], ts_str, ts_str,
                            wait_secs, bq["position"], False,
                            bq["hotspot_x"], bq["hotspot_y"], sess_seq=sess["seq"],
                        )
                        post_event(q_evt)

                    if cam_role == "entry":
                        exit_evt = make_entry_exit_event(
                            "EXIT", vid, s_code, sid, camera_id, ts_str,
                            sess["is_staff"], 0.90, sess_seq=sess["seq"],
                        )
                        post_event(exit_evt)
                    else:
                        # Exit scoring-harness compat event
                        exit_evt = {
                            "event_id": str(uuid.uuid4()),
                            "store_id": sid,
                            "camera_id": camera_id,
                            "visitor_id": vid,
                            "event_type": "EXIT",
                            "timestamp": ts_str,
                            "zone_id": sess.get("current_zone"),
                            "dwell_ms": int((current_time - sess["enter_time"]).total_seconds() * 1000) if sess.get("current_zone") else 0,
                            "is_staff": sess["is_staff"],
                            "confidence": 0.90,
                            "metadata": {"queue_depth": None, "sku_zone": None, "session_seq": sess["seq"]},
                        }
                        post_event(exit_evt)

                    historical_exits[vid] = True

            for vid in to_remove:
                del active_sessions[vid]

        out.write(frame)

    cap.release()
    out.release()
    print(f"✅ Finished processing {video_path}")

    # Transcode to H.264 for browser compatibility
    if os.path.exists(temp_out_path):
        try:
            print("Transcoding to H.264...")
            import subprocess
            cmd = [
                "ffmpeg", "-y", "-i", temp_out_path,
                "-vcodec", "libx264", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", "-an", out_path
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"H.264 video saved to: {out_path}")
            os.remove(temp_out_path)
        except Exception as e:
            print(f"Transcoding failed: {e}. Keeping MPEG-4.")
            if os.path.exists(out_path):
                try:
                    os.remove(out_path)
                except Exception:
                    pass
            os.rename(temp_out_path, out_path)

    try:
        with open("pipeline/simulation_progress.json", "w") as f:
            json.dump({
                "video": base_name, "camera_id": camera_id, "store_id": sid,
                "frame": frame_count if frame_count > 0 else frame_num,
                "total": frame_count if frame_count > 0 else frame_num,
                "percent": 100, "status": "done"
            }, f)
        print("Simulation progress marked as done.")
    except Exception as e:
        print(f"Failed to write final progress JSON: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python detect.py <video_path> [model_path]")
    else:
        model_p = sys.argv[2] if len(sys.argv) > 2 else "yolo11n.pt"
        run_detection(sys.argv[1], model_p)
