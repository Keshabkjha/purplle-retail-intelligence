
import csv
import io
import logging
from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session

from app.database import DBPOS

logger = logging.getLogger(__name__)


def parse_pos_datetime(order_date: str, order_time: str) -> Optional[str]:
    
    for fmt in ["%d-%m-%Y", "%Y-%m-%d", "%m/%d/%Y"]:
        try:
            dt = datetime.strptime(f"{order_date.strip()} {order_time.strip()}", f"{fmt} %H:%M:%S")
            return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            continue
    logger.warning(f"Could not parse POS datetime: {order_date} {order_time}")
    return None


def is_loyalty_scan(brand_name: str, total_amount: float) -> bool:
    return brand_name.strip().lower() == "purplle" and total_amount < 2.0


def load_pos_csv(csv_content: str, db: Session, store_id_override: Optional[str] = None) -> dict:
   
    loaded = 0
    skipped_loyalty = 0
    duplicates = 0
    errors = 0

    reader = csv.DictReader(io.StringIO(csv_content))

    # Normalize column headers (strip whitespace)
    rows = []
    for row in reader:
        rows.append({k.strip(): v.strip() if v else "" for k, v in row.items()})

    for row in rows:
        try:
            order_id = row.get("order_id", "").strip()
            order_date = row.get("order_date", "").strip()
            order_time = row.get("order_time", "").strip()
            raw_store_id = row.get("store_id", "").strip()
            product_id = row.get("product_id", "").strip()
            brand_name = row.get("brand_name", "").strip()
            total_amount_str = row.get("total_amount", "0").strip()

            store_id = store_id_override or raw_store_id

            try:
                total_amount = float(total_amount_str)
            except ValueError:
                total_amount = 0.0

            # Skip loyalty scans
            if is_loyalty_scan(brand_name, total_amount):
                skipped_loyalty += 1
                continue

            # Build ISO timestamp
            iso_ts = parse_pos_datetime(order_date, order_time)
            if not iso_ts:
                errors += 1
                continue

            # Idempotency check: skip if already exists
            existing = (
                db.query(DBPOS)
                .filter(
                    DBPOS.order_id == order_id,
                    DBPOS.store_id == store_id,
                    DBPOS.product_id == product_id,
                )
                .first()
            )
            if existing:
                duplicates += 1
                continue

            db_pos = DBPOS(
                order_id=order_id,
                store_id=store_id,
                timestamp=iso_ts,
                brand_name=brand_name,
                total_amount=total_amount,
                product_id=product_id,
                order_date=order_date,
                order_time=order_time,
            )
            db.add(db_pos)
            loaded += 1

        except Exception as e:
            logger.error(f"Error processing POS row {row}: {e}")
            errors += 1

    try:
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"DB commit failed during POS load: {e}")
        return {"loaded": 0, "skipped_loyalty": skipped_loyalty, "duplicates": duplicates, "errors": errors + loaded}

    return {
        "loaded": loaded,
        "skipped_loyalty": skipped_loyalty,
        "duplicates": duplicates,
        "errors": errors,
    }


def get_transaction_times(store_id: str, db: Session) -> list:
    from sqlalchemy import distinct
    rows = (
        db.query(distinct(DBPOS.timestamp))
        .filter(DBPOS.store_id == store_id)
        .all()
    )
    from app.metrics import parse_timestamp
    times = []
    for (ts,) in rows:
        dt = parse_timestamp(ts)
        if dt:
            times.append(dt)
    return sorted(times)


def get_brand_transaction_map(store_id: str, db: Session) -> dict:
    rows = db.query(DBPOS.brand_name, DBPOS.timestamp).filter(DBPOS.store_id == store_id).all()
    brand_map: dict = {}
    from app.metrics import parse_timestamp
    for brand, ts in rows:
        dt = parse_timestamp(ts)
        if dt and brand:
            brand_map.setdefault(brand, []).append(dt)
    return brand_map
