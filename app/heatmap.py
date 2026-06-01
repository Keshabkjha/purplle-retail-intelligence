from typing import Optional

from sqlalchemy.orm import Session

from app.database import DBEvent
from app.metrics import parse_timestamp


def get_store_heatmap_data(store_id: str, db: Session, camera_id: Optional[str] = None):
    # 1. Fetch all customer events (excluding ENTRY/EXIT zones)
    query = db.query(DBEvent).filter(
        DBEvent.store_id == store_id,
        DBEvent.is_staff.is_(False),
        DBEvent.zone_id.isnot(None),
        DBEvent.zone_id != "ENTRY",
        DBEvent.zone_id != "EXIT",
    )
    if camera_id:
        query = query.filter(DBEvent.camera_id == camera_id)

    events = query.order_by(DBEvent.timestamp).all()

    # Get total unique sessions to set data_confidence
    sessions_query = db.query(DBEvent.visitor_id).filter(
        DBEvent.store_id == store_id, DBEvent.is_staff.is_(False)
    )
    if camera_id:
        sessions_query = sessions_query.filter(DBEvent.camera_id == camera_id)
    unique_sessions = sessions_query.distinct().count()

    data_confidence = unique_sessions >= 20

    if not events:
        return {"store_id": store_id, "camera_id": camera_id, "zones": [], "data_confidence": data_confidence}

    zone_data = {}
    visitor_zone_visits = {}

    for ev in events:
        zone = ev.zone_id
        vid = ev.visitor_id

        if zone not in zone_data:
            zone_data[zone] = {"visitors": set(), "dwell_ms_list": []}

        zone_data[zone]["visitors"].add(vid)

        if ev.dwell_ms and ev.dwell_ms > 0:
            zone_data[zone]["dwell_ms_list"].append(ev.dwell_ms)

        key = (vid, zone)
        if key not in visitor_zone_visits:
            visitor_zone_visits[key] = {"enter": None, "exit": None, "last_timestamp": None}

        dt = parse_timestamp(ev.timestamp)
        if dt:
            visitor_zone_visits[key]["last_timestamp"] = dt
            if ev.event_type == "ZONE_ENTER":
                visitor_zone_visits[key]["enter"] = dt
            elif ev.event_type == "ZONE_EXIT":
                visitor_zone_visits[key]["exit"] = dt

    for (vid, zone), times in visitor_zone_visits.items():
        if zone in zone_data:
            calc_ms = 0
            if times["enter"] and times["exit"]:
                calc_ms = int((times["exit"] - times["enter"]).total_seconds() * 1000)
            elif times["enter"] and times["last_timestamp"]:
                calc_ms = int((times["last_timestamp"] - times["enter"]).total_seconds() * 1000)

            if calc_ms > 0:
                zone_data[zone]["dwell_ms_list"].append(calc_ms)

    zones_list = []
    max_visits = 0

    for zone, data in zone_data.items():
        visit_count = len(data["visitors"])
        if visit_count > max_visits:
            max_visits = visit_count

        avg_dwell_sec = 0.0
        if data["dwell_ms_list"]:
            avg_dwell_sec = round(
                (sum(data["dwell_ms_list"]) / len(data["dwell_ms_list"])) / 1000.0, 2
            )

        zones_list.append(
            {
                "zone_id": zone,
                "visit_count": visit_count,
                "average_dwell_seconds": avg_dwell_sec,
                "intensity": 0.0,
            }
        )

    for z in zones_list:
        if max_visits > 0:
            z["intensity"] = round(100.0 * z["visit_count"] / max_visits, 2)

    return {"store_id": store_id, "camera_id": camera_id, "zones": zones_list, "data_confidence": data_confidence}
