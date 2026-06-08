import json
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy.orm import Session

from app.database import DBPOS, DBEvent


def parse_timestamp(ts_str):
    try:
        if not ts_str:
            return None
        normalized = ts_str.replace("Z", "+00:00")
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            return dt
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    except Exception:
        try:
            return datetime.strptime(ts_str.replace("Z", ""), "%Y-%m-%dT%H:%M:%S")
        except Exception:
            return None


def _load_store_layout():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    layout_path = os.path.join(base_dir, "config", "store_layout.json")
    if not os.path.exists(layout_path):
        return {}
    try:
        with open(layout_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def get_entry_camera_ids(store_id: str) -> list[str]:
    layout = _load_store_layout()
    store_config = next((s for s in layout.get("stores", []) if s.get("store_id") == store_id), None)
    if not store_config:
        return ["CAM_ENTRY_01", "cam1"]
    roles = store_config.get("camera_roles", {})
    cameras = store_config.get("cameras", {})
    entry_ids = [cam_id for cam_id in cameras if roles.get(cam_id) == "entry"]
    entry_ids.extend(["CAM_ENTRY_01", "cam1"])
    deduped = []
    for cam_id in entry_ids:
        if cam_id not in deduped:
            deduped.append(cam_id)
    return deduped or ["CAM_ENTRY_01", "cam1"]


def get_store_metrics_data(store_id: str, db: Session, camera_id: Optional[str] = None):
    # Entry cameras for this store (actual camera IDs)
    entry_cams = get_entry_camera_ids(store_id)

    # Entry event types (Purplle native + legacy)
    ENTRY_EVENT_TYPES = ("ENTRY", "REENTRY", "entry", "reentry")
    BILLING_EVENT_TYPES = (
        "BILLING_QUEUE_JOIN", "BILLING_QUEUE_ABANDON",
        "queue_completed", "queue_abandoned",
    )
    ZONE_DWELL_TYPES = ("ZONE_DWELL", "ZONE_EXIT", "zone_exited", "BILLING_QUEUE_JOIN", "queue_completed", "queue_abandoned")

    # Calculate total store visitors based on entry cameras
    total_store_visitors = (
        db.query(DBEvent.visitor_id)
        .filter(
            DBEvent.store_id == store_id,
            DBEvent.camera_id.in_(entry_cams),
            DBEvent.is_staff.is_(False)
        )
        .distinct()
        .count()
    )

    # Fallback to all unique store visitors if entry camera hasn't been run yet
    if total_store_visitors == 0:
        total_store_visitors = (
            db.query(DBEvent.visitor_id)
            .filter(
                DBEvent.store_id == store_id,
                DBEvent.is_staff.is_(False)
            )
            .distinct()
            .count()
        )

    # Fetch all events for the store that are not staff
    events = db.query(DBEvent).filter(
        DBEvent.store_id == store_id, DBEvent.is_staff.is_(False)
    ).order_by(DBEvent.timestamp).all()

    if not events:
        return {
            "store_id": store_id,
            "camera_id": camera_id,
            "unique_visitors": 0,
            "conversion_rate": 0.0,
            "average_dwell_minutes": 0.0,
            "current_queue_depth": 0,
            "abandonment_rate": 0.0,
        }

    # Group all events by visitor_id to build complete store session history
    sessions = {}
    for ev in events:
        vid = ev.visitor_id
        if vid not in sessions:
            sessions[vid] = []
        sessions[vid].append(ev)

    # If camera_id is specified, filter active sessions to visitors who visited this specific camera
    if camera_id:
        active_sessions = {
            vid: ev_list for vid, ev_list in sessions.items()
            if any(ev.camera_id == camera_id for ev in ev_list)
        }
        unique_visitors = len(active_sessions)
    else:
        active_sessions = sessions
        unique_visitors = total_store_visitors

    # 2. Fetch POS transactions
    pos_txns = db.query(DBPOS).filter(DBPOS.store_id == store_id).all()
    txn_times = []
    for tx in pos_txns:
        dt = parse_timestamp(tx.timestamp)
        if dt:
            txn_times.append(dt)

    # Calculate converted visitors and dwell times
    converted_visitors = set()
    total_dwell_seconds = 0.0
    billing_visitor_ids = set()
    visitor_billing_times: dict[str, list[datetime]] = {}

    for vid, ev_list in active_sessions.items():
        # Dwell time calculation
        cam_events = [ev for ev in ev_list if ev.camera_id == camera_id] if camera_id else ev_list
        if cam_events:
            visits = []
            current_visit = []
            for ev in cam_events:
                if ev.event_type in ENTRY_EVENT_TYPES and current_visit:
                    visits.append(current_visit)
                    current_visit = []
                current_visit.append(ev)
            if current_visit:
                visits.append(current_visit)

            for visit in visits:
                first_time = parse_timestamp(visit[0].timestamp)
                last_time = parse_timestamp(visit[-1].timestamp)
                if first_time and last_time:
                    dwell_sec = (last_time - first_time).total_seconds()
                    total_dwell_seconds += max(dwell_sec, 0.0)

        # Conversion: check billing zone events
        billing_visits = []
        for ev in ev_list:
            is_billing = (
                (ev.zone_id and "BILLING" in (ev.zone_id or "").upper())
                or ev.event_type in BILLING_EVENT_TYPES
            )
            if is_billing:
                dt = parse_timestamp(ev.timestamp)
                if dt:
                    billing_visits.append(dt)
                    billing_visitor_ids.add(vid)
        if billing_visits:
            visitor_billing_times[vid] = billing_visits

    consumed_txns = set()
    sorted_visitors = sorted(
        visitor_billing_times.items(),
        key=lambda item: (min(item[1]) if item[1] else datetime.max, item[0]),
    )
    for vid, billing_visits in sorted_visitors:
        matched_txn_idx = None
        for t_idx, t_time in enumerate(txn_times):
            if t_idx in consumed_txns:
                continue
            if any(b_time <= t_time <= b_time + timedelta(minutes=5) for b_time in billing_visits):
                matched_txn_idx = t_idx
                break
        if matched_txn_idx is not None:
            converted_visitors.add(vid)
            consumed_txns.add(matched_txn_idx)

    conversion_rate = 0.0
    if total_store_visitors > 0:
        conversion_rate = round(100.0 * len(converted_visitors) / total_store_visitors, 2)

    avg_dwell_minutes = 0.0
    if unique_visitors > 0:
        avg_dwell_minutes = round((total_dwell_seconds / 60.0) / unique_visitors, 2)

    # Current Queue Depth
    current_queue_depth = 0
    q_query = db.query(DBEvent).filter(
        DBEvent.store_id == store_id,
        DBEvent.event_type.in_(["BILLING_QUEUE_JOIN", "queue_completed", "queue_abandoned"])
    )
    if camera_id:
        q_query = q_query.filter(DBEvent.camera_id == camera_id)
    latest_queue_event = q_query.order_by(DBEvent.timestamp.desc()).first()

    if latest_queue_event and latest_queue_event.metadata_json:
        try:
            meta = latest_queue_event.metadata_json
            if isinstance(meta, str):
                meta = json.loads(meta)
            current_queue_depth = meta.get("queue_depth", 0) or 0
        except Exception:
            pass

    total_billing_visitors = len(billing_visitor_ids)
    converted_billing_visitors = len(billing_visitor_ids.intersection(converted_visitors))

    abandonment_rate = 0.0
    if total_billing_visitors > 0:
        abandoned_count = total_billing_visitors - converted_billing_visitors
        abandonment_rate = round(100.0 * abandoned_count / total_billing_visitors, 2)

    zone_dwell_secs = {}
    for ev in events:
        if ev.zone_id and ev.dwell_ms and ev.dwell_ms > 0:
            if ev.event_type in (
                "ZONE_DWELL", "ZONE_EXIT",
                "BILLING_QUEUE_JOIN", "BILLING_QUEUE_ABANDON",
                "zone_exited", "queue_completed", "queue_abandoned",
            ):
                if ev.zone_id not in zone_dwell_secs:
                    zone_dwell_secs[ev.zone_id] = []
                zone_dwell_secs[ev.zone_id].append(ev.dwell_ms / 1000.0)

    average_dwell_per_zone = {}
    for zone_id, dwell_list in zone_dwell_secs.items():
        if dwell_list:
            average_dwell_per_zone[zone_id] = round(sum(dwell_list) / len(dwell_list), 2)

    return {
        "store_id": store_id,
        "camera_id": camera_id,
        "unique_visitors": unique_visitors,
        "conversion_rate": conversion_rate,
        "average_dwell_minutes": avg_dwell_minutes,
        "average_dwell_per_zone": average_dwell_per_zone,
        "current_queue_depth": current_queue_depth,
        "abandonment_rate": abandonment_rate,
    }
