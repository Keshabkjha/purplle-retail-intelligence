import json
from datetime import timedelta

from sqlalchemy.orm import Session

from app.database import DBPOS, DBEvent
from app.metrics import get_store_metrics_data, parse_timestamp


def get_store_anomalies_data(store_id: str, db: Session):
    anomalies = []

    # 1. Fetch metrics to check queue depth and conversion rate
    metrics = get_store_metrics_data(store_id, db)
    
    # -----------------------------------------------------------------------
    # BILLING QUEUE SPIKE — 7-day statistical rolling average baseline
    # -----------------------------------------------------------------------
    current_depth = metrics["current_queue_depth"]
    
    # Establish temporal anchors from latest event timestamp
    latest_event = db.query(DBEvent).filter(
        DBEvent.store_id == store_id,
        DBEvent.is_staff.is_(False)
    ).order_by(DBEvent.timestamp.desc()).first()

    latest_ts = None
    if latest_event:
        latest_ts = parse_timestamp(latest_event.timestamp)

    try:
        depths = []
        if latest_ts:
            window_7d_start = latest_ts - timedelta(days=7)
            window_7d_start_str = window_7d_start.strftime("%Y-%m-%dT%H:%M:%SZ")
            
            queue_events = db.query(DBEvent.metadata_json).filter(
                DBEvent.store_id == store_id,
                DBEvent.event_type == "BILLING_QUEUE_JOIN",
                DBEvent.timestamp >= window_7d_start_str
            ).all()
            
            for ev in queue_events:
                meta = ev[0]
                if isinstance(meta, str):
                    meta = json.loads(meta)
                if meta and "queue_depth" in meta and meta["queue_depth"] is not None:
                    depths.append(meta["queue_depth"])
                    
        if len(depths) >= 5:
            avg_depth = sum(depths) / len(depths)
            variance = sum((x - avg_depth) ** 2 for x in depths) / len(depths)
            std_dev = variance ** 0.5
            threshold = max(4, avg_depth + 1.5 * std_dev)
            
            if current_depth >= threshold:
                severity = "CRITICAL" if current_depth >= threshold + std_dev else "WARN"
                anomalies.append({
                    "anomaly_type": "STATISTICAL_QUEUE_SPIKE",
                    "severity": severity,
                    "suggested_action": "Open additional billing counter immediately. Queue is higher than historical rolling average.",
                    "details": f"Queue depth {current_depth} exceeds 7-day rolling statistical threshold of {threshold:.1f} (Avg: {avg_depth:.1f})."
                })
        else:
            if current_depth >= 8:
                anomalies.append({
                    "anomaly_type": "BILLING_QUEUE_SPIKE",
                    "severity": "CRITICAL",
                    "suggested_action": "Open additional billing counter immediately.",
                    "details": f"Billing queue depth is currently {current_depth}."
                })
            elif current_depth >= 5:
                anomalies.append({
                    "anomaly_type": "BILLING_QUEUE_SPIKE",
                    "severity": "WARN",
                    "suggested_action": "Monitor billing counter closely.",
                    "details": f"Billing queue depth is currently {current_depth}."
                })
    except Exception as e:
        print(f"Error in queue anomaly calculation: {e}")

    # -----------------------------------------------------------------------
    # CONVERSION DROP — 7-day rolling Daily Business Review baseline
    # Compare current day's conversion against the 7-day daily average baseline
    # -----------------------------------------------------------------------
    try:
        current_conversion = metrics["conversion_rate"]
        unique_visitors = metrics["unique_visitors"]

        # Find all timestamps to establish a 7-day window
        all_events = db.query(DBEvent).filter(
            DBEvent.store_id == store_id,
            DBEvent.is_staff.is_(False)
        ).order_by(DBEvent.timestamp.desc()).all()

        latest_ts = None
        if all_events:
            latest_ts = parse_timestamp(all_events[0].timestamp)

        baseline_rate = 0.0
        if latest_ts:
            daily_rates = []
            for i in range(1, 8):  # Query daily slices for the past 7 days prior
                d_start = (latest_ts - timedelta(days=i)).replace(hour=0, minute=0, second=0, microsecond=0)
                d_end = d_start + timedelta(hours=23, minutes=59, seconds=59)
                
                d_start_str = d_start.strftime("%Y-%m-%dT%H:%M:%SZ")
                d_end_str = d_end.strftime("%Y-%m-%dT%H:%M:%SZ")
                
                # Daily unique visitors
                day_unique = db.query(DBEvent.visitor_id).filter(
                    DBEvent.store_id == store_id,
                    DBEvent.is_staff.is_(False),
                    DBEvent.timestamp >= d_start_str,
                    DBEvent.timestamp <= d_end_str
                ).distinct().count()
                
                # Daily POS transactions
                day_txns = db.query(DBPOS).filter(
                    DBPOS.store_id == store_id,
                    DBPOS.timestamp >= d_start_str,
                    DBPOS.timestamp <= d_end_str
                ).count()
                
                if day_unique > 0:
                    daily_rates.append(100.0 * day_txns / day_unique)
                    
            if daily_rates:
                baseline_rate = round(sum(daily_rates) / len(daily_rates), 2)

        if baseline_rate > 0:
            # Trigger WARN if current conversion is > 30% below the Daily Business Review baseline
            if current_conversion < baseline_rate * 0.70:
                anomalies.append({
                    "anomaly_type": "CONVERSION_DROP",
                    "severity": "WARN",
                    "suggested_action": "Check for checkout bottlenecks or staff availability. Conversion is significantly below the 7-day Daily Business Review baseline.",
                    "details": f"Current conversion {current_conversion}% is well below the 7-day Daily Business Review baseline of {baseline_rate}%."
                })
        elif unique_visitors >= 5 and current_conversion < 10.0:
            anomalies.append({
                "anomaly_type": "CONVERSION_DROP",
                "severity": "WARN",
                "suggested_action": "Check for checkout bottlenecks or staff availability.",
                "details": f"Conversion rate is low at {current_conversion}% with {unique_visitors} visitors."
            })
    except Exception as e:
        print(f"Error in conversion anomaly calculation: {e}")

    # -----------------------------------------------------------------------
    # DEAD ZONES — 30-minute rolling window (per rubric spec)
    # Flag retail zones that had 0 customer visits in the past 30 minutes
    # -----------------------------------------------------------------------
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    layout_path = os.path.join(base_dir, "config", "store_layout.json")

    if os.path.exists(layout_path):
        try:
            with open(layout_path, "r") as f:
                layout = json.load(f)

            retail_zones = [z["zone_id"] for z in layout.get("zones", [])
                            if z["zone_id"] not in ("ENTRY", "EXIT", "BILLING")]

            # Find the latest event timestamp to anchor the 30-min window
            latest_event = db.query(DBEvent).filter(
                DBEvent.store_id == store_id,
                DBEvent.is_staff.is_(False)
            ).order_by(DBEvent.timestamp.desc()).first()

            if latest_event:
                latest_ts = parse_timestamp(latest_event.timestamp)
                if latest_ts:
                    window_30m_start = latest_ts - timedelta(minutes=30)
                    window_30m_start_str = window_30m_start.strftime("%Y-%m-%dT%H:%M:%SZ")

                    # Query zones visited within the last 30 minutes
                    recent_visited = db.query(DBEvent.zone_id).filter(
                        DBEvent.store_id == store_id,
                        DBEvent.is_staff.is_(False),
                        DBEvent.zone_id.isnot(None),
                        DBEvent.timestamp >= window_30m_start_str
                    ).distinct().all()
                    recent_zone_ids = {z[0] for z in recent_visited}

                    # Only flag dead zones if there was active traffic (at least 1 customer) in the store during this 30m window
                    total_recent_visitors = db.query(DBEvent.visitor_id).filter(
                        DBEvent.store_id == store_id,
                        DBEvent.is_staff.is_(False),
                        DBEvent.timestamp >= window_30m_start_str
                    ).distinct().count()

                    if total_recent_visitors > 0:
                        for rz in retail_zones:
                            if rz not in recent_zone_ids:
                                anomalies.append({
                                    "anomaly_type": "DEAD_ZONE",
                                    "severity": "INFO",
                                    "suggested_action": f"Inspect product display and visibility in the {rz} zone.",
                                    "details": f"The zone '{rz}' has received 0 customer visits during the monitoring period."
                                })
        except Exception as e:
            print(f"Error in dead zone anomaly calculation: {e}")

    return {
        "store_id": store_id,
        "anomalies": anomalies
    }
