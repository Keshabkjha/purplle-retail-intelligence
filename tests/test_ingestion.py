import os
import csv
import pytest
from app.database import DBPOS, Base, SessionLocal, engine
from app.ingestion import seed_pos_data

def test_seed_pos_data(tmp_path, monkeypatch):
    # Setup test CSV file in tmp_path
    csv_file = os.path.join(tmp_path, "test_pos.csv")
    with open(csv_file, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["order_id", "store_id", "order_date", "order_time", "brand_name", "total_amount"])
        writer.writerow(["TX123", "ST1008", "10-04-2026", "16:55:36", "LAKME", "250.0"])
        writer.writerow(["TX124", "ST1008", "10-04-2026", "17:00:00", "MAYBELLINE", "150.0"])

    # Mock glob.glob to return our test CSV file
    import glob
    def mock_glob(*args, **kwargs):
        return [csv_file]
    monkeypatch.setattr(glob, "glob", mock_glob)

    # Initialize clean SQLite DB
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        # Run seed
        seed_pos_data()
        
        # Verify database
        count = db.query(DBPOS).count()
        assert count == 2
        
        # Verify record details
        rec1 = db.query(DBPOS).filter(DBPOS.order_id == "TX123").first()
        assert rec1 is not None
        assert rec1.brand_name == "LAKME"
        assert rec1.total_amount == 250.0
        assert rec1.timestamp == "2026-04-10T16:55:36Z"
        
        # Seed second time should skip due to existing count
        seed_pos_data()
        assert db.query(DBPOS).count() == 2
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)
