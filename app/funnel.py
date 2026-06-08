from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy.orm import Session

from app.database import DBPOS, DBEvent
from app.metrics import get_entry_camera_ids, parse_timestamp


def get_store_funnel_data(store_id: str, db: Session, camera_id: Optional[str] = None):
    # 1. Fetch all customer events for the store (no camera filter initially to construct sessions)
    all_events = db.query(DBEvent).filter(
        DBEvent.store_id == store_id, DBEvent.is_staff.is_(False)
    ).order_by(DBEvent.timestamp).all()

    if not all_events:
        return {
            "store_id": store_id,
            "camera_id": camera_id,
            "funnel": {"entry": 0, "zone_visit": 0, "billing_queue": 0, "purchase": 0},
            "dropoff_percentages": {
                "entry_to_zone": 0.0,
                "zone_to_billing": 0.0,
                "billing_to_purchase": 0.0,
            },
        }

    # Group all events by visitor_id to trace shopper session journeys
    sessions = {}
    for ev in all_events:
        vid = ev.visitor_id
        if vid not in sessions:
            sessions[vid] = []
        sessions[vid].append(ev)

    # 2. Fetch POS transactions for conversion
    pos_txns = db.query(DBPOS).filter(DBPOS.store_id == store_id).all()
    txn_times = []
    for tx in pos_txns:
        dt = parse_timestamp(tx.timestamp)
        if dt:
            txn_times.append(dt)

    entry_cams = get_entry_camera_ids(store_id)
    entry_vids = {ev.visitor_id for ev in all_events if ev.camera_id in entry_cams}
    # Fallback to all visitor IDs if no entry camera events are found yet
    if not entry_vids:
        entry_vids = set(sessions.keys())

    # If camera_id is specified, filter to visitors who visited this camera AND entered the store
    if camera_id:
        camera_vids = {
            ev.visitor_id for ev in all_events if ev.camera_id == camera_id
        }
        active_vids = entry_vids.intersection(camera_vids)
    else:
        active_vids = entry_vids

    entry_count = 0
    zone_visit_count = 0
    billing_queue_count = 0
    purchase_count = 0
    visitor_billing_times = {}

    for vid in active_vids:
        ev_list = sessions[vid]
        entry_count += 1

        has_visited_retail = False
        for ev in ev_list:
            if ev.zone_id and ev.zone_id not in ("ENTRY", "EXIT", "BILLING"):
                has_visited_retail = True
                break
        if has_visited_retail:
            zone_visit_count += 1

        has_visited_billing = False
        billing_visits = []
        for ev in ev_list:
            if ev.zone_id == "BILLING" or ev.event_type in (
                "BILLING_QUEUE_JOIN",
                "BILLING_QUEUE_ABANDON",
                "queue_completed",
                "queue_abandoned",
            ):
                has_visited_billing = True
                dt = parse_timestamp(ev.timestamp)
                if dt:
                    billing_visits.append(dt)
        if has_visited_billing:
            billing_queue_count += 1
            visitor_billing_times[vid] = billing_visits

    converted_visitors = set()
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

    purchase_count = len(converted_visitors)

    entry_to_zone = 0.0
    if entry_count > 0:
        entry_to_zone = round(100.0 * (1.0 - (zone_visit_count / entry_count)), 2)

    zone_to_billing = 0.0
    if zone_visit_count > 0:
        zone_to_billing = round(100.0 * (1.0 - (billing_queue_count / zone_visit_count)), 2)

    billing_to_purchase = 0.0
    if billing_queue_count > 0:
        billing_to_purchase = round(100.0 * (1.0 - (purchase_count / billing_queue_count)), 2)

    return {
        "store_id": store_id,
        "camera_id": camera_id,
        "funnel": {
            "entry": entry_count,
            "zone_visit": zone_visit_count,
            "billing_queue": billing_queue_count,
            "purchase": purchase_count,
        },
        "dropoff_percentages": {
            "entry_to_zone": entry_to_zone,
            "zone_to_billing": zone_to_billing,
            "billing_to_purchase": billing_to_purchase,
        },
    }
