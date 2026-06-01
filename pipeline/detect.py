import json
import os
import sys
import uuid
from datetime import datetime, timedelta

import cv2
import numpy as np
import requests
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

from pipeline.adaptive_models import (
    AdaptiveModelRegistry,
    build_identity_feature_vector,
    build_staff_feature_vector,
)

# Load layouts
LAYOUT_PATH = "config/store_layout.json"
CALIBRATION_PATH = "config/calibration.json"
INGEST_URL = os.getenv("INGEST_URL", "http://localhost:8000/events/ingest")

for p in ("/workspace/config/store_layout.json", "/Users/keshabkumar/Purpple Challenge/config/store_layout.json"):
    if os.path.exists(p):
        LAYOUT_PATH = p
        break

for p in ("/workspace/config/calibration.json", "/Users/keshabkumar/Purpple Challenge/config/calibration.json"):
    if os.path.exists(p):
        CALIBRATION_PATH = p
        break

store_id = "ST1008"
zones = []
cameras_mapping = {}

if os.path.exists(LAYOUT_PATH):
    try:
        with open(LAYOUT_PATH, "r") as f:
            layout_data = json.load(f)
            store_id = layout_data.get("store_id", "ST1008")
            zones = layout_data.get("zones", [])
            cameras_mapping = layout_data.get("cameras", {})
    except Exception as e:
        print(f"Error loading layout: {e}")

# Load or initialize calibration
calibration = {}
if os.path.exists(CALIBRATION_PATH):
    try:
        with open(CALIBRATION_PATH, "r") as f:
            calibration = json.load(f)
    except Exception as e:
        print(f"Error loading calibration: {e}")

MODEL_REGISTRY = AdaptiveModelRegistry()

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
    # Check if we have calibration points for this camera
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

    # Fallback to sensible scaling mapping from 1920x1080 (standard) to 940x451 (floor plan)
    # Different cameras look at different sections of the store
    scale_x = 940.0 / frame_w
    scale_y = 451.0 / frame_h
    
    # Simple linear bounding fallbacks per camera
    if camera_id == "CAM_ENTRY_01":
        # Entry is on the left side
        wx = px * (150.0 / frame_w)
        wy = 200.0 + py * (200.0 / frame_h)
    elif camera_id == "CAM_BILLING_01":
        # Billing is on the right side
        wx = 700.0 + px * (200.0 / frame_w)
        wy = 100.0 + py * (400.0 / frame_h)
    elif camera_id == "CAM_MAIN_01":
        wx = px * (400.0 / frame_w)
        wy = py * (300.0 / frame_h)
    elif camera_id == "CAM_MAIN_02":
        wx = px * (500.0 / frame_w)
        wy = py * (300.0 / frame_h)
    else:
        wx = px * scale_x
        wy = py * scale_y
        
    return wx, wy

def determine_zone(wx, wy, camera_id):
    """Finds which zone contains the warped floor coordinates."""
    # Match camera specific zones first
    for zone in zones:
        if zone["camera_id"] == camera_id:
            coords = zone["polygon_coords"]
            if point_in_polygon(wx, wy, coords):
                return zone["zone_id"]
                
    # Global fallback match across any camera zone
    for zone in zones:
        coords = zone["polygon_coords"]
        if point_in_polygon(wx, wy, coords):
            return zone["zone_id"]
            
    return None

def is_staff_heuristic(track_id, bbox, frame):
    """Detect staff by analyzing the uniform color (e.g. black/dark grey) in the upper torso."""
    try:
        x1, y1, x2, y2 = map(int, bbox)
        # Ensure coordinates are within frame
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        # Crop to top 50% (upper torso)
        torso_y2 = y1 + max(1, (y2 - y1) // 2)
        torso = frame[y1:torso_y2, x1:x2]
        
        if torso.size == 0:
            return False
            
        # Convert to HSV
        hsv_torso = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        
        # Define color range for Black/Dark Grey uniform
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 50])
        
        mask = cv2.inRange(hsv_torso, lower_black, upper_black)
        
        # Calculate percentage of matching pixels
        match_ratio = cv2.countNonZero(mask) / (torso.shape[0] * torso.shape[1] + 1e-6)
        
        # If > 30% of torso is black, assume staff
        if match_ratio > 0.30:
            return True
    except Exception:
        pass
        
    return False

def compute_appearance_embedding(crop):
    """Extract a lightweight normalized HSV color histogram as an appearance embedding."""
    try:
        if crop is None or crop.size == 0:
            return None
        # Convert to HSV color space
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        # 3D HSV histogram: 8 Hue bins, 4 Saturation bins, 4 Value bins
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten().tolist()
    except Exception:
        return None

def compare_appearance(hist1, hist2):
    """Compare two appearance embeddings using correlation (1.0 is perfect match)."""
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
    if not previous_camera_id or not current_camera_id or previous_camera_id == current_camera_id:
        return 0.0

    transition_map = {
        "CAM_ENTRY_01": {
            "CAM_MAIN_01": 0.95,
            "CAM_MAIN_02": 0.90,
            "CAM_MAIN_03": 0.90,
            "CAM_BILLING_01": 0.75,
        },
        "CAM_MAIN_01": {
            "CAM_ENTRY_01": 0.90,
            "CAM_MAIN_02": 0.88,
            "CAM_MAIN_03": 0.88,
            "CAM_BILLING_01": 0.85,
        },
        "CAM_MAIN_02": {
            "CAM_ENTRY_01": 0.90,
            "CAM_MAIN_01": 0.88,
            "CAM_MAIN_03": 0.88,
            "CAM_BILLING_01": 0.85,
        },
        "CAM_MAIN_03": {
            "CAM_ENTRY_01": 0.90,
            "CAM_MAIN_01": 0.88,
            "CAM_MAIN_02": 0.88,
            "CAM_BILLING_01": 0.85,
        },
        "CAM_BILLING_01": {
            "CAM_ENTRY_01": 0.80,
            "CAM_MAIN_01": 0.90,
            "CAM_MAIN_02": 0.90,
            "CAM_MAIN_03": 0.90,
        },
    }

    return transition_map.get(previous_camera_id, {}).get(current_camera_id, 0.60)


def zone_transition_prior(previous_zone_id, current_zone_id):
    """Return a soft prior for whether a zone transition is plausible."""
    if not previous_zone_id or not current_zone_id:
        return 0.50
    if previous_zone_id == current_zone_id:
        return 1.0
    if "ENTRY" in (previous_zone_id, current_zone_id):
        return 0.85
    if "BILLING" in (previous_zone_id, current_zone_id):
        return 0.90
    return 0.80

class CrossCameraSessionTracker:
    def __init__(self, state_file="pipeline/session_state.json"):
        self.state_file = state_file
        self.sessions = {}
        self.local_to_unified = {}  # track_id -> unified_id for the current video run
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
        """Determine if visitor has been seen in another camera previously."""
        if unified_id in self.sessions:
            sess = self.sessions[unified_id]
            for cam in sess.get("camera_track_ids", {}):
                if cam != current_camera_id:
                    return True
        return False

    def get_unified_id(self, track_id, camera_id, wx, wy, current_time, is_staff_initial, box=None, frame=None, zone_id=None):
        """Maps a local camera track_id to a globally consistent unified visitor_id using spatial, temporal, and visual cues."""
        track_id = int(track_id)
        wx = float(wx)
        wy = float(wy)
        # Calculate appearance embedding if crop is provided
        emb = None
        if box is not None and frame is not None:
            try:
                x1, y1, x2, y2 = map(int, box)
                h, w = frame.shape[:2]
                crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                if crop is not None and crop.size > 0:
                    # Focus on the upper torso, which is usually the most stable clothing region.
                    torso_h = max(1, int(crop.shape[0] * 0.65))
                    crop = crop[:torso_h, :]
                emb = compute_appearance_embedding(crop)
            except Exception:
                pass

        if track_id in self.local_to_unified:
            unified_id = self.local_to_unified[track_id]
            if unified_id in self.sessions:
                sess = self.sessions[unified_id]
                sess["last_seen_camera"] = camera_id
                sess["last_seen_time"] = current_time.strftime("%Y-%m-%dT%H:%M:%SZ")
                sess["last_seen_x"] = wx
                sess["last_seen_y"] = wy
                sess["camera_track_ids"][camera_id] = track_id
                if zone_id is not None:
                    sess["last_zone_id"] = zone_id
                
                # Rolling update of visual embedding to handle camera angle / lighting changes
                if emb is not None:
                    old_emb = sess.get("appearance_embedding")
                    if old_emb is not None:
                        # 30% new visual signal, 70% rolling history
                        updated = (0.7 * np.array(old_emb) + 0.3 * np.array(emb)).tolist()
                        sess["appearance_embedding"] = updated
                    else:
                        sess["appearance_embedding"] = emb
                    self.save_state()
            return unified_id

        # Multi-signal match in previously exited/seen visitors from other cameras
        matched_id = None
        max_match_score = -1.0
        best_dist = float('inf')
        best_diff = 0.0
        candidate_observations = []

        for uid, sess in self.sessions.items():
            if sess["last_seen_camera"] == camera_id:
                continue  # Match across different cameras only

            last_time_str = sess["last_seen_time"]
            try:
                last_time = datetime.strptime(last_time_str, "%Y-%m-%dT%H:%M:%SZ")
            except ValueError:
                continue

            diff = abs((current_time - last_time).total_seconds())
            if diff <= 30.0:  # 30-second window
                lx, ly = sess["last_seen_x"], sess["last_seen_y"]
                dist = np.sqrt((wx - lx)**2 + (wy - ly)**2)
                
                if dist <= 150.0:  # 150-pixel spatial threshold (~2.5m)
                    # 1. Spatial proximity score (higher is better)
                    spatial_score = 1.0 - (dist / 150.0)
                    
                    # 2. Temporal closeness score
                    temporal_score = 1.0 - (diff / 30.0)
                    
                    # 3. Visual appearance correlation score
                    app_score = 0.5  # Neutral default if visual embedding is missing
                    if emb is not None and sess.get("appearance_embedding") is not None:
                        app_score = compare_appearance(emb, sess["appearance_embedding"])

                    # 4. Camera transition prior (same store traversal path)
                    cam_score = camera_transition_prior(sess["last_seen_camera"], camera_id)

                    # 5. Zone compatibility prior
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
                    
                    # Unified multi-signal match score
                    heuristic_score = (
                        spatial_score * 0.30 +
                        temporal_score * 0.22 +
                        app_score * 0.26 +
                        cam_score * 0.12 +
                        zone_score * 0.10
                    )
                    learned_prob = MODEL_REGISTRY.predict_identity_probability(feature_vec, fallback=heuristic_score)
                    match_score = (
                        0.58 * heuristic_score +
                        0.42 * learned_prob
                    )
                    candidate_observations.append((uid, feature_vec, match_score, heuristic_score, learned_prob))
                    
                    if match_score >= 0.65 and match_score > max_match_score:
                        max_match_score = match_score
                        matched_id = uid
                        best_dist = dist
                        best_diff = diff

        if matched_id:
            print(f"🔗 [Re-ID Match] Track {track_id} in {camera_id} matched to {matched_id} (score: {max_match_score:.2f}, dist: {best_dist:.1f}px, gap: {best_diff:.1f}s)")
            # Online learning: reinforce the selected match and penalize hard negatives.
            for uid, feature_vec, match_score, heuristic_score, learned_prob in candidate_observations:
                if uid == matched_id and match_score >= 0.65:
                    MODEL_REGISTRY.identity_model.update(feature_vec, 1)
                elif match_score >= 0.60 and uid != matched_id:
                    MODEL_REGISTRY.identity_model.update(feature_vec, 0)
            self.local_to_unified[track_id] = matched_id
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
                if old_emb is not None:
                    sess["appearance_embedding"] = (0.7 * np.array(old_emb) + 0.3 * np.array(emb)).tolist()
                else:
                    sess["appearance_embedding"] = emb
            self.save_state()
            return matched_id
        else:
            # When no match is found, treat the strongest candidate as a negative example
            if candidate_observations:
                top_candidate = max(candidate_observations, key=lambda item: item[2])
                if top_candidate[2] >= 0.60:
                    MODEL_REGISTRY.identity_model.update(top_candidate[1], 0)

            new_id = f"VIS_{track_id}"
            base_new_id = new_id
            counter = 1
            while new_id in self.sessions:
                new_id = f"{base_new_id}_{counter}"
                counter += 1

            self.local_to_unified[track_id] = new_id
            self.sessions[new_id] = {
                "unified_id": new_id,
                "last_seen_camera": camera_id,
                "last_seen_time": current_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "last_seen_x": wx,
                "last_seen_y": wy,
                "is_staff": is_staff_initial,
                "last_zone_id": zone_id,
                "appearance_embedding": emb,
                "camera_track_ids": {camera_id: track_id}
            }
            self.save_state()
            return new_id


def update_staff_status(vid, sess, current_time, zone_id, wx, wy, is_clothing_staff, tracker):
    """Hybrid visual-behavioral staff classifier with a lightweight learned model."""
    unified_is_staff_init = False
    camera_count = 1
    if tracker and vid in tracker.sessions:
        unified_is_staff_init = tracker.sessions[vid].get("is_staff", False)
        camera_count = len(tracker.sessions[vid].get("camera_track_ids", {}))

    # Base visual score: black uniform detection is strong but can have reflections/noise
    # We assign 0.6 for uniform match, 0.0 otherwise
    staff_score = 0.6 if is_clothing_staff or unified_is_staff_init else 0.0

    # Behavioral boosts:
    # 1. Physical location: Behind the billing register counter area (wx > 820)
    if zone_id == "BILLING" and wx > 820:
        staff_score += 0.4

    # 2. Queue Dwell: Standing in billing zone for > 90 seconds (staff stays at register; shoppers checkout and leave)
    if zone_id == "BILLING":
        billing_duration = (current_time - sess["enter_time"]).total_seconds()
        if billing_duration > 90:
            staff_score += 0.3

    # Torso match ratio is a strong signal in our store because staff uniforms are dark.
    torso_match_ratio = 1.0 if is_clothing_staff else 0.0

    # 3. Store Dwell: Staff has much higher store presence duration than customers
    total_duration = (current_time - sess.get("first_seen_time", sess["enter_time"])).total_seconds()
    if total_duration > 180:
        staff_score += 0.3

    # 4. Multi-camera activity (staff moves across cameras over long shifts)
    if camera_count >= 2:
        staff_score += 0.2

    heuristic_prob = min(max(staff_score / 1.8, 0.0), 1.0)
    feature_vec = build_staff_feature_vector(
        torso_match_ratio=torso_match_ratio,
        is_clothing_staff=is_clothing_staff,
        zone_id=zone_id,
        wx=wx,
        billing_duration_sec=(current_time - sess["enter_time"]).total_seconds() if zone_id == "BILLING" else 0.0,
        total_duration_sec=total_duration,
        camera_count=camera_count,
    )
    learned_prob = MODEL_REGISTRY.predict_staff_probability(feature_vec, fallback=heuristic_prob)
    combined_prob = (0.55 * learned_prob) + (0.45 * heuristic_prob)

    # Robust multi-stage confidence resolve with hysteresis.
    # Cold-start: rely on the conservative heuristic until the learned model has enough evidence.
    if MODEL_REGISTRY.staff_model.update_count < 5:
        if staff_score >= 0.6:
            is_staff = True
        elif staff_score < 0.4:
            is_staff = False
        else:
            # Ambiguous band [0.4, 0.6): Resolve using spatiotemporal continuity
            if total_duration > 120 or camera_count >= 2:
                is_staff = True
            else:
                is_staff = False
    else:
        if combined_prob >= 0.62:
            # High confidence staff
            is_staff = True
        elif combined_prob < 0.38:
            # High confidence customer
            is_staff = False
        else:
            # Ambiguous band [0.4, 0.6): Resolve using spatiotemporal continuity
            if total_duration > 120 or camera_count >= 2:
                is_staff = True
            else:
                is_staff = False

    # Online learning: use high-confidence pseudo-labels to train the lightweight classifier.
    if combined_prob >= 0.80 or staff_score >= 1.2:
        MODEL_REGISTRY.staff_model.update(feature_vec, 1)
    elif combined_prob <= 0.20 or staff_score <= 0.15:
        MODEL_REGISTRY.staff_model.update(feature_vec, 0)

    if tracker and vid in tracker.sessions:
        tracker.sessions[vid]["is_staff"] = is_staff
        tracker.save_state()

    return is_staff

def post_event(event):
    """Posts a structured event to the FastAPI ingest endpoint."""
    try:
        response = requests.post(INGEST_URL, json=[event], timeout=2)
        if response.status_code == 207:
            print(f"Event {event['event_type']} for VIS_{event['visitor_id']} ingested.")
        else:
            print(f"Failed ingestion: {response.text}")
    except Exception as e:
        print(f"Connection error to ingest API: {e}")

def run_detection(video_path: str, model_path: str = "yolo11n.pt"):
    print(f"Initializing YOLO11 model: {model_path}")
    model = YOLO(model_path)

    if not os.path.exists(video_path):
        print(f"Error: Video file {video_path} does not exist.")
        return

    # Determine camera_id from file name
    base_name = os.path.basename(video_path)
    camera_id = "CAM_MAIN_01" # default
    if "entry" in base_name:
        camera_id = "CAM_ENTRY_01"
    elif "billing" in base_name:
        camera_id = "CAM_BILLING_01"
    elif "main_floor_1" in base_name:
        camera_id = "CAM_MAIN_01"
    elif "main_floor_2" in base_name:
        camera_id = "CAM_MAIN_02"
    elif "main_floor_3" in base_name:
        camera_id = "CAM_MAIN_03"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    print(f"Processing {video_path} ({camera_id}) | FPS: {fps} | Total Frames: {frame_count}")

    # Setup VideoWriter
    out_path = f"annotated_{base_name}"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    print(f"Annotated output will be saved to: {out_path}")

    # Reset/clear cross-camera tracking state on entry camera run
    state_file = "pipeline/session_state.json"
    if "entry" in base_name:
        if os.path.exists(state_file):
            try:
                os.remove(state_file)
                print(f"Resetting cross-camera session state file: {state_file}")
            except Exception as e:
                print(f"Error resetting state file: {e}")

    tracker = CrossCameraSessionTracker(state_file)

    # Session states
    # visitor_id -> { "current_zone", "enter_time", "first_seen_time", "last_seen", "dwell_sent_count", "seq", "is_staff" }
    active_sessions = {}
    
    # Track visitors who have fully exited, for REENTRY detection
    # visitor_id -> True
    historical_exits = {}
    
    # Clip base start time shifted to 16:40:00 to align with POS transaction timestamps
    base_time = datetime(2026, 4, 10, 16, 40, 0)

    frame_num = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1

        # Calculate current frame timestamp
        offset_seconds = frame_num / fps
        current_time = base_time + timedelta(seconds=offset_seconds)
        ts_str = current_time.strftime("%Y-%m-%dT%H:%M:%SZ")

        # Run YOLO tracking every frame for smooth video
        results = model.track(frame, persist=True, verbose=False)
        
        seen_tracks = set()
        seen_vids = set()
        if results and results[0].boxes and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()

            for box, track_id, cls, conf in zip(boxes, track_ids, classes, confidences):
                if cls == 0: # Person
                    seen_tracks.add(track_id)
                    x1, y1, x2, y2 = box
                    px = (x1 + x2) / 2.0
                    py = y2  # foot position is bottom center

                    # Map to floor plan
                    wx, wy = map_camera_to_floor(px, py, camera_id, width, height)
                    zone_id = determine_zone(wx, wy, camera_id)
                    is_staff_init = is_staff_heuristic(track_id, box, frame)

                    # Get globally consistent unified visitor ID
                    vid = tracker.get_unified_id(
                        track_id,
                        camera_id,
                        wx,
                        wy,
                        current_time,
                        is_staff_init,
                        box=box,
                        frame=frame,
                        zone_id=zone_id,
                    )
                    seen_vids.add(vid)
                    
                    # Get display staff flag dynamically
                    is_staff_display = is_staff_init
                    if vid in tracker.sessions:
                        is_staff_display = tracker.sessions[vid].get("is_staff", is_staff_init)
                    
                    # Draw bounding box and info on frame
                    color = (255, 0, 255) if is_staff_display else (0, 255, 0)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    label = f"{vid} | {zone_id or 'UNKNOWN'}"
                    if is_staff_display:
                        label += " [STAFF]"
                    cv2.putText(frame, label, (int(x1), max(10, int(y1) - 10)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    # Event logic (evaluate API posts only every 15 frames to reduce load, but track continuously)
                    if frame_num % 15 == 0:
                        # Manage session states
                        if vid not in active_sessions:
                            active_sessions[vid] = {
                                "current_zone": None,
                                "enter_time": current_time,
                                "first_seen_time": current_time,
                                "last_seen": current_time,
                                "dwell_sent_count": 0,
                                "seq": 1,
                                "is_staff": is_staff_init
                            }
                            
                            # Determine if this is a REENTRY or first ENTRY
                            is_reentry = (vid in historical_exits) or tracker.is_reentry(vid, camera_id)
                            is_staff = update_staff_status(vid, active_sessions[vid], current_time, zone_id, wx, wy, is_staff_init, tracker)
                            active_sessions[vid]["is_staff"] = is_staff
                            
                            if is_reentry:
                                # Visitor previously exited — emit REENTRY
                                reentry_evt = {
                                    "event_id": str(uuid.uuid4()),
                                    "store_id": store_id,
                                    "camera_id": camera_id,
                                    "visitor_id": vid,
                                    "event_type": "REENTRY",
                                    "timestamp": ts_str,
                                    "zone_id": "ENTRY" if camera_id == "CAM_ENTRY_01" else zone_id,
                                    "dwell_ms": 0,
                                    "is_staff": is_staff,
                                    "confidence": float(conf),
                                    "metadata": {
                                        "queue_depth": None,
                                        "sku_zone": None,
                                        "session_seq": 1
                                    }
                                }
                                post_event(reentry_evt)
                            else:
                                # Brand new visitor — emit ENTRY
                                entry_evt = {
                                    "event_id": str(uuid.uuid4()),
                                    "store_id": store_id,
                                    "camera_id": camera_id,
                                    "visitor_id": vid,
                                    "event_type": "ENTRY",
                                    "timestamp": ts_str,
                                    "zone_id": "ENTRY" if camera_id == "CAM_ENTRY_01" else zone_id,
                                    "dwell_ms": 0,
                                    "is_staff": is_staff,
                                    "confidence": float(conf),
                                    "metadata": {
                                        "queue_depth": None,
                                        "sku_zone": None,
                                        "session_seq": 1
                                    }
                                }
                                post_event(entry_evt)
                        
                        sess = active_sessions[vid]
                        sess["last_seen"] = current_time
                        # Update staff status from latest detection (may improve over time)
                        is_staff = update_staff_status(vid, sess, current_time, zone_id, wx, wy, is_staff_init, tracker)
                        sess["is_staff"] = is_staff
                        
                        # Check zone changes
                        old_zone = sess["current_zone"]
                        if zone_id != old_zone:
                            # Exit old zone
                            if old_zone:
                                sess["seq"] += 1

                                # If leaving BILLING zone to a non-EXIT zone: BILLING_QUEUE_ABANDON
                                if old_zone == "BILLING" and zone_id not in (None, "EXIT"):
                                    abandon_evt = {
                                        "event_id": str(uuid.uuid4()),
                                        "store_id": store_id,
                                        "camera_id": camera_id,
                                        "visitor_id": vid,
                                        "event_type": "BILLING_QUEUE_ABANDON",
                                        "timestamp": ts_str,
                                        "zone_id": "BILLING",
                                        "dwell_ms": int((current_time - sess["enter_time"]).total_seconds() * 1000),
                                        "is_staff": sess["is_staff"],
                                        "confidence": float(conf),
                                        "metadata": {
                                            "queue_depth": None,
                                            "sku_zone": "BILLING",
                                            "session_seq": sess["seq"]
                                        }
                                    }
                                    post_event(abandon_evt)
                                    sess["seq"] += 1

                                exit_evt = {
                                    "event_id": str(uuid.uuid4()),
                                    "store_id": store_id,
                                    "camera_id": camera_id,
                                    "visitor_id": vid,
                                    "event_type": "ZONE_EXIT",
                                    "timestamp": ts_str,
                                    "zone_id": old_zone,
                                    "dwell_ms": int((current_time - sess["enter_time"]).total_seconds() * 1000),
                                    "is_staff": sess["is_staff"],
                                    "confidence": float(conf),
                                    "metadata": {
                                        "queue_depth": None,
                                        "sku_zone": None,
                                        "session_seq": sess["seq"]
                                    }
                                }
                                post_event(exit_evt)

                            # Enter new zone
                            sess["current_zone"] = zone_id
                            sess["enter_time"] = current_time
                            sess["dwell_sent_count"] = 0
                            
                            if zone_id:
                                sess["seq"] += 1
                                enter_type = "ZONE_ENTER"
                                q_depth = None
                                if zone_id == "BILLING":
                                    enter_type = "BILLING_QUEUE_JOIN"
                                    # Simulate queue depth: count number of other active sessions in billing
                                    q_depth = sum(1 for s in active_sessions.values() if s["current_zone"] == "BILLING")
                                
                                enter_evt = {
                                    "event_id": str(uuid.uuid4()),
                                    "store_id": store_id,
                                    "camera_id": camera_id,
                                    "visitor_id": vid,
                                    "event_type": enter_type,
                                    "timestamp": ts_str,
                                    "zone_id": zone_id,
                                    "dwell_ms": 0,
                                    "is_staff": sess["is_staff"],
                                    "confidence": float(conf),
                                    "metadata": {
                                        "queue_depth": q_depth,
                                        "sku_zone": zone_id,
                                        "session_seq": sess["seq"]
                                    }
                                }
                                post_event(enter_evt)
                        
                        else:
                            # Still in the same zone, check continuous dwell (emit every 30s)
                            if zone_id:
                                duration = (current_time - sess["enter_time"]).total_seconds()
                                expected_dwells = int(duration // 30)
                                if expected_dwells > sess["dwell_sent_count"]:
                                    sess["dwell_sent_count"] = expected_dwells
                                    sess["seq"] += 1
                                    dwell_evt = {
                                        "event_id": str(uuid.uuid4()),
                                        "store_id": store_id,
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
                                            "session_seq": sess["seq"]
                                        }
                                    }
                                    post_event(dwell_evt)

        # Check for completed sessions (no longer seen for > 15 seconds) - run every 15 frames
        if frame_num % 15 == 0:
            to_remove = []
            for vid, sess in active_sessions.items():
                if vid not in seen_vids and (current_time - sess["last_seen"]).total_seconds() > 15:
                    to_remove.append(vid)
                    
                    # Post EXIT event — use stored is_staff from session, not hardcoded False
                    sess["seq"] += 1
                    exit_evt = {
                        "event_id": str(uuid.uuid4()),
                        "store_id": store_id,
                        "camera_id": camera_id,
                        "visitor_id": vid,
                        "event_type": "EXIT",
                        "timestamp": ts_str,
                        "zone_id": "ENTRY" if camera_id == "CAM_ENTRY_01" else sess["current_zone"],
                        "dwell_ms": int((current_time - sess["enter_time"]).total_seconds() * 1000) if sess["current_zone"] else 0,
                        "is_staff": sess["is_staff"],  # FIX: use session's tracked is_staff value
                        "confidence": 0.9,
                        "metadata": {
                            "queue_depth": None,
                            "sku_zone": None,
                            "session_seq": sess["seq"]
                        }
                    }
                    post_event(exit_evt)
                    # Mark as historically exited for REENTRY detection
                    historical_exits[vid] = True
            
            for vid in to_remove:
                del active_sessions[vid]

        # Write frame to video
        out.write(frame)

    cap.release()
    out.release()
    print(f"Finished processing {video_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python detect.py <video_path>")
    else:
        run_detection(sys.argv[1])
