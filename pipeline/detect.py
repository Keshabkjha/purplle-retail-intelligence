import cv2
import torch

# PyTorch 2.6+ weights_only unpickling workaround for Ultralytics/YOLO models
original_load = torch.load
def patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    try:
        return original_load(*args, **kwargs)
    except TypeError:
        kwargs.pop('weights_only', None)
        return original_load(*args, **kwargs)
torch.load = patched_load

from ultralytics import YOLO
import sys
import os
import json
import requests
import uuid
from datetime import datetime, timedelta
import numpy as np

# Load layouts
LAYOUT_PATH = "config/store_layout.json"
CALIBRATION_PATH = "config/calibration.json"

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

def post_event(event):
    """Posts a structured event to the FastAPI ingest endpoint."""
    url = "http://localhost:8000/events/ingest"
    try:
        response = requests.post(url, json=[event], timeout=2)
        if response.status_code == 207:
            print(f"Event {event['event_type']} for VIS_{event['visitor_id']} ingested.")
        else:
            print(f"Failed ingestion: {response.text}")
    except Exception as e:
        print(f"Connection error to ingest API: {e}")

def run_detection(video_path: str, model_path: str = "yolov8n.pt"):
    print(f"Initializing YOLO model: {model_path}")
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

    # Session states
    # visitor_id -> { "current_zone": zone, "enter_time": dt, "last_seen": dt, "dwell_sent_count": int, "seq": int }
    active_sessions = {}
    
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
                    is_staff = is_staff_heuristic(track_id, box, frame)

                    vid = f"VIS_{track_id}"
                    
                    # Draw bounding box and info on frame
                    color = (255, 0, 255) if is_staff else (0, 255, 0)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    label = f"{vid} | {zone_id or 'UNKNOWN'}"
                    if is_staff:
                        label += " [STAFF]"
                    cv2.putText(frame, label, (int(x1), max(10, int(y1) - 10)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    # Event logic (evaluate API posts only every 15 frames to reduce load, but track continuously)
                    if frame_num % 15 == 0:
                        # Manage session states
                        if vid not in active_sessions:
                            # Start new session: emit ENTRY event if on entry camera
                            active_sessions[vid] = {
                                "current_zone": None,
                                "enter_time": current_time,
                                "last_seen": current_time,
                                "dwell_sent_count": 0,
                                "seq": 1
                            }
                            
                            # Post ENTRY event
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
                        
                        # Check zone changes
                        old_zone = sess["current_zone"]
                        if zone_id != old_zone:
                            # Exit old zone
                            if old_zone:
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
                                    "is_staff": is_staff,
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
                                    "is_staff": is_staff,
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
                                        "is_staff": is_staff,
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
                if vid not in seen_tracks and (current_time - sess["last_seen"]).total_seconds() > 15:
                    to_remove.append(vid)
                    
                    # Post EXIT event
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
                        "is_staff": False, # Just assuming not staff on exit for simplicity
                        "confidence": 0.9,
                        "metadata": {
                            "queue_depth": None,
                            "sku_zone": None,
                            "session_seq": sess["seq"]
                        }
                    }
                    post_event(exit_evt)
            
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
