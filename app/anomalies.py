from sqlalchemy.orm import Session
from app.database import DBEvent
from app.metrics import get_store_metrics_data
import json

def get_store_anomalies_data(store_id: str, db: Session):
    anomalies = []

    # 1. Fetch metrics to check queue depth and conversion rate
    metrics = get_store_metrics_data(store_id, db)
    
    # Check Billing Queue Spike using Statistical Rolling Average
    current_depth = metrics["current_queue_depth"]
    try:
        # Fetch historical queue depths
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
            
            # Anomaly if current depth > avg + 2*std_dev (or absolute minimum 4)
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
            # Fallback to static if not enough data
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
                    "suggested_action": "Open additional billing counter immediately.",
                    "details": f"Billing queue depth is currently {current_depth}."
                })
    except Exception as e:
        print(f"Error in queue anomaly calculation: {e}")

    # Check Conversion Rate Drop
    # Baseline conversion rate is typically 15%. If it drops below 10%, trigger WARN
    conversion_rate = metrics["conversion_rate"]
    unique_visitors = metrics["unique_visitors"]
    # Only check if we have enough visitors to be statistically meaningful
    if unique_visitors >= 5 and conversion_rate < 10.0:
        anomalies.append({
            "anomaly_type": "CONVERSION_DROP",
            "severity": "WARN",
            "suggested_action": "Check for checkout bottlenecks or staff availability.",
            "details": f"Conversion rate is extremely low at {conversion_rate}% with {unique_visitors} visitors."
        })

    # Check Dead Zones
    # Get all zones from the store layout (we can query them from layout or define them)
    # Since we can query from layout, let's load layout
    import os
    layout_path = "config/store_layout.json"
    if not os.path.exists(layout_path):
        # Check container workspace path or fallback absolute path
        for p in ("/workspace/config/store_layout.json", "/Users/keshabkumar/Purpple Challenge/config/store_layout.json"):
            if os.path.exists(p):
                layout_path = p
                break
    if os.path.exists(layout_path):
        with open(layout_path, "r") as f:
            layout = json.load(f)
        
        retail_zones = [z["zone_id"] for z in layout.get("zones", []) if z["zone_id"] not in ("ENTRY", "EXIT", "BILLING")]
        
        # Check visits for each zone
        visited_zones = db.query(DBEvent.zone_id).filter(
            DBEvent.store_id == store_id,
            DBEvent.is_staff == False,
            DBEvent.zone_id.isnot(None)
        ).distinct().all()
        visited_zone_ids = {z[0] for z in visited_zones}

        for rz in retail_zones:
            if rz not in visited_zone_ids:
                anomalies.append({
                    "anomaly_type": "DEAD_ZONE",
                    "severity": "INFO",
                    "suggested_action": f"Inspect product display and visibility in the {rz} zone.",
                    "details": f"The zone '{rz}' has received 0 customer visits during the monitoring period."
                })

    return {
        "store_id": store_id,
        "anomalies": anomalies
    }
