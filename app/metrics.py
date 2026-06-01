import json
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy.orm import Session

from app.database import DBPOS, DBEvent


def parse_timestamp(ts_str):
    try:
        return datetime.strptime(ts_str.replace("Z", ""), "%Y-%m-%dT%H:%M:%S")
    except Exception:
        try:
            return datetime.fromisoformat(ts_str)
        except Exception:
            return None


def get_store_metrics_data(store_id: str, db: Session, camera_id: Optional[str] = None):
    # 1. Fetch all events for the store that are not staff
    query = db.query(DBEvent).filter(
        DBEvent.store_id == store_id, DBEvent.is_staff.is_(False)
    )
    if camera_id:
        query = query.filter(DBEvent.camera_id == camera_id)

    events = query.order_by(DBEvent.timestamp).all()

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

    # Group events by visitor_id
    sessions = {}
    for ev in events:
        vid = ev.visitor_id
        if vid not in sessions:
            sessions[vid] = []
        sessions[vid].append(ev)

    unique_visitors = len(sessions)

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

    for vid, ev_list in sessions.items():
        visits = []
        current_visit = []
        for ev in ev_list:
            if ev.event_type in ("ENTRY", "REENTRY") and current_visit:
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

        billing_visits = []
        for ev in ev_list:
            if ev.zone_id == "BILLING" or ev.event_type in (
                "BILLING_QUEUE_JOIN",
                "BILLING_QUEUE_ABANDON",
            ):
                dt = parse_timestamp(ev.timestamp)
                if dt:
                    billing_visits.append(dt)
                    billing_visitor_ids.add(vid)

        is_converted = False
        for b_time in billing_visits:
            for t_time in txn_times:
                if b_time <= t_time <= b_time + timedelta(minutes=5):
                    is_converted = True
                    break
            if is_converted:
                break

        if is_converted:
            converted_visitors.add(vid)

    conversion_rate = 0.0
    if unique_visitors > 0:
        conversion_rate = round(100.0 * len(converted_visitors) / unique_visitors, 2)

    avg_dwell_minutes = 0.0
    if unique_visitors > 0:
        avg_dwell_minutes = round((total_dwell_seconds / 60.0) / unique_visitors, 2)

    # Current Queue Depth
    q_query = db.query(DBEvent).filter(
        DBEvent.store_id == store_id, DBEvent.event_type == "BILLING_QUEUE_JOIN"
    )
    if camera_id:
        q_query = q_query.filter(DBEvent.camera_id == camera_id)
    latest_queue_event = q_query.order_by(DBEvent.timestamp.desc()).first()

    current_queue_depth = 0
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
                "ZONE_DWELL",
                "ZONE_EXIT",
                "BILLING_QUEUE_JOIN",
                "BILLING_QUEUE_ABANDON",
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
