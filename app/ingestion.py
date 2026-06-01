import csv
import os
import sys

from app.database import DBPOS, SessionLocal, init_db

def seed_pos_data():
    import glob
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_files = glob.glob(os.path.join(base_dir, "*.csv"))
    if not csv_files:
        print("Error: POS CSV file not found in root directory.")
        sys.exit(1)
    
    csv_file = csv_files[0]
        
    print(f"🌱 Seeding POS transactions from: {csv_file}...")
    db = SessionLocal()
    try:
        # Initialize tables
        init_db()
        
        # Check if already seeded to prevent duplication
        existing_count = db.query(DBPOS).count()
        if existing_count > 0:
            print(f"POS transactions database already seeded ({existing_count} records). Skipping.")
            return

        with open(csv_file, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            records = []
            seen_orders = set()
            
            for row in reader:
                order_id = row.get("order_id")
                if not order_id:
                    continue
                # Deduplicate by order_id
                if order_id in seen_orders:
                    continue
                seen_orders.add(order_id)
                
                # Parse date and time fields
                order_date = row.get("order_date")  # e.g., '10-04-2026'
                order_time = row.get("order_time")  # e.g., '16:55:36'
                
                # Convert '10-04-2026' and '16:55:36' to strict ISO-8601 '2026-04-10T16:55:36Z'
                timestamp = "2026-04-10T12:00:00Z"
                if order_date and order_time:
                    try:
                        day, month, year = order_date.split("-")
                        timestamp = f"{year}-{month}-{day}T{order_time}Z"
                    except Exception:
                        pass
                
                try:
                    total_amount = float(row.get("total_amount", 0.0) or 0.0)
                except ValueError:
                    total_amount = 0.0

                pos_rec = DBPOS(
                    order_id=order_id,
                    store_id=row.get("store_id", "ST1008"),
                    timestamp=timestamp,
                    brand_name=row.get("brand_name", "UNKNOWN"),
                    total_amount=total_amount
                )
                records.append(pos_rec)
                
            if records:
                db.bulk_save_objects(records)
                db.commit()
                print(f"✅ Successfully seeded {len(records)} POS transaction records into tables.")
            else:
                print("⚠️ No valid POS records parsed from the CSV.")
    except Exception as e:
        db.rollback()
        print(f"❌ Failed to seed POS data: {e}")
        sys.exit(1)
    finally:
        db.close()

if __name__ == "__main__":
    seed_pos_data()
