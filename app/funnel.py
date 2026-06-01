from datetime import timedelta
from typing import Optional

from sqlalchemy.orm import Session

from app.database import DBPOS, DBEvent
from app.metrics import parse_timestamp


def get_store_funnel_data(store_id: str, db: Session, camera_id: Optional[str] = None):
    # 1. Fetch all customer events
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
            "funnel": {"entry": 0, "zone_visit": 0, "billing_queue": 0, "purchase": 0},
            "dropoff_percentages": {
                "entry_to_zone": 0.0,
                "zone_to_billing": 0.0,
                "billing_to_purchase": 0.0,
            },
        }

    sessions = {}
    for ev in events:
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

    entry_count = 0
    zone_visit_count = 0
    billing_queue_count = 0
    purchase_count = 0

    for vid, ev_list in sessions.items():
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
            ):
                has_visited_billing = True
                dt = parse_timestamp(ev.timestamp)
                if dt:
                    billing_visits.append(dt)
        if has_visited_billing:
            billing_queue_count += 1

        is_converted = False
        if has_visited_billing:
            for b_time in billing_visits:
                for t_time in txn_times:
                    if b_time <= t_time <= b_time + timedelta(minutes=5):
                        is_converted = True
                        break
                if is_converted:
                    break
        if is_converted:
            purchase_count += 1

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
