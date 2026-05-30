from sqlalchemy.orm import Session
from sqlalchemy import func
from app.database import DBEvent, DBPOS
from app.metrics import get_store_metrics_data, parse_timestamp
import json
from datetime import datetime, timedelta

def get_store_anomalies_data(store_id: str, db: Session):
    anomalies = []

    # 1. Fetch metrics to check queue depth and conversion rate
    metrics = get_store_metrics_data(store_id, db)
    
    # -----------------------------------------------------------------------
    # BILLING QUEUE SPIKE — Statistical rolling average over all history
    # -----------------------------------------------------------------------
    current_depth = metrics["current_queue_depth"]
    try:
        queue_events = db.query(DBEvent.metadata_json).filter(
            DBEvent.store_id == store_id,
            DBEvent.event_type == "BILLING_QUEUE_JOIN"
        ).all()
        
        depths = []
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
                    "suggested_action": "Open additional billing counter immediately. Queue is higher than historical average.",
                    "details": f"Queue depth {current_depth} exceeds statistical threshold of {threshold:.1f} (Avg: {avg_depth:.1f})."
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
    # CONVERSION DROP — 7-day rolling baseline (per rubric spec)
    # Compare current day's conversion against the 7-day historical average
    # -----------------------------------------------------------------------
    try:
        current_conversion = metrics["conversion_rate"]
        unique_visitors = metrics["unique_visitors"]

        # Find all timestamps to establish a 7-day window
        all_events = db.query(DBEvent).filter(
            DBEvent.store_id == store_id,
            DBEvent.is_staff == False
        ).order_by(DBEvent.timestamp.desc()).all()

        if all_events:
            latest_ts = parse_timestamp(all_events[0].timestamp)
            if latest_ts:
                window_start = latest_ts - timedelta(days=7)

                # Compute 7-day visitor count and POS transaction count
                week_events = [ev for ev in all_events if parse_timestamp(ev.timestamp) and parse_timestamp(ev.timestamp) >= window_start]
                week_visitor_ids = {ev.visitor_id for ev in week_events if not ev.is_staff}

                week_pos = db.query(DBPOS).filter(DBPOS.store_id == store_id).all()
                week_txn_times = [parse_timestamp(tx.timestamp) for tx in week_pos
                                  if parse_timestamp(tx.timestamp) and parse_timestamp(tx.timestamp) >= window_start]

                week_unique = len(week_visitor_ids)
                week_txns = len(week_txn_times)

                if week_unique > 10:
                    baseline_rate = round(100.0 * week_txns / week_unique, 2)
                    # Trigger WARN if current conversion is > 30% below 7-day baseline
                    if baseline_rate > 0 and current_conversion < baseline_rate * 0.70:
                        anomalies.append({
                            "anomaly_type": "CONVERSION_DROP",
                            "severity": "WARN",
                            "suggested_action": "Check for checkout bottlenecks or staff availability. Conversion is significantly below the 7-day baseline.",
                            "details": f"Current conversion {current_conversion}% is well below the 7-day baseline of {baseline_rate}%."
                        })
                elif unique_visitors >= 5 and current_conversion < 10.0:
                    # Fallback when not enough historical data: static threshold
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
    layout_path = "config/store_layout.json"
    if not os.path.exists(layout_path):
        for p in ("/workspace/config/store_layout.json", "/Users/keshabkumar/Purpple Challenge/config/store_layout.json",
                  "/Users/keshabkumar/purplle-retail-intelligence/config/store_layout.json"):
            if os.path.exists(p):
                layout_path = p
                break

    if os.path.exists(layout_path):
        try:
            with open(layout_path, "r") as f:
                layout = json.load(f)

            retail_zones = [z["zone_id"] for z in layout.get("zones", [])
                            if z["zone_id"] not in ("ENTRY", "EXIT", "BILLING")]

            # Find the latest event timestamp to anchor the 30-min window
            latest_event = db.query(DBEvent).filter(
                DBEvent.store_id == store_id,
                DBEvent.is_staff == False
            ).order_by(DBEvent.timestamp.desc()).first()

            if latest_event:
                latest_ts = parse_timestamp(latest_event.timestamp)
                if latest_ts:
                    window_30m_start = latest_ts - timedelta(minutes=30)
                    window_30m_start_str = window_30m_start.strftime("%Y-%m-%dT%H:%M:%SZ")

                    # Query zones visited within the last 30 minutes
                    recent_visited = db.query(DBEvent.zone_id).filter(
                        DBEvent.store_id == store_id,
                        DBEvent.is_staff == False,
                        DBEvent.zone_id.isnot(None),
                        DBEvent.timestamp >= window_30m_start_str
                    ).distinct().all()
                    recent_zone_ids = {z[0] for z in recent_visited}

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

