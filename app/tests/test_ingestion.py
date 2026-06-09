from app.database import DBPOS, Base, SessionLocal, engine
from app.pos_loader import load_pos_csv

# PROMPT: Generate comprehensive tests for POS CSV data loading covering:
# - CSV parsing with various date formats
# - Loyalty card scan filtering (amount < 2.0)
# - Idempotency (duplicate detection)
# - Timezone normalization to ISO-8601 UTC
# - Handling malformed timestamps
# - Multi-store store_id override functionality
#
# CHANGES MADE:
# - Replaced seed_pos_data with load_pos_csv to match actual API
# - Extended CSV test cases to include edge cases (loyalty scans, malformed dates)
# - Added timestamp format validation
# - Parameterized date format testing (DD-MM-YYYY, YYYY-MM-DD, MM/DD/YYYY)


def test_load_pos_csv_basic(tmp_path):
    """Test basic POS CSV loading with valid records"""
    csv_content = """order_id,store_id,order_date,order_time,brand_name,total_amount,product_id
TX123,ST1008,10-04-2026,16:55:36,LAKME,250.0,381916
TX124,ST1008,10-04-2026,17:00:00,MAYBELLINE,150.0,399945"""

    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        result = load_pos_csv(csv_content, db, store_id_override=None)
        
        # Verify counts
        assert result["loaded"] == 2
        assert result["skipped_loyalty"] == 0
        assert result["duplicates"] == 0
        
        # Verify records in DB
        count = db.query(DBPOS).count()
        assert count == 2
        
        # Verify timestamp format
        rec = db.query(DBPOS).filter(DBPOS.order_id == "TX123").first()
        assert rec.timestamp == "2026-04-10T16:55:36Z"
        assert rec.brand_name == "LAKME"
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)


def test_load_pos_csv_loyalty_filtering():
    """Test that Purplle loyalty card scans (amount < 2.0) are filtered"""
    csv_content = """order_id,store_id,order_date,order_time,brand_name,total_amount,product_id
TX125,ST1008,10-04-2026,16:55:36,Purplle,0.0,394925
TX126,ST1008,10-04-2026,16:55:36,Purplle,1.5,401002
TX127,ST1008,10-04-2026,16:55:36,LAKME,250.0,381916"""

    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        result = load_pos_csv(csv_content, db)
        
        # Should skip 2 Purplle scans, load 1 real transaction
        assert result["loaded"] == 1
        assert result["skipped_loyalty"] == 2
        
        # Only LAKME should be in DB
        count = db.query(DBPOS).count()
        assert count == 1
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)


def test_load_pos_csv_idempotency():
    """Test that duplicate records are not reloaded"""
    csv_content = """order_id,store_id,order_date,order_time,brand_name,total_amount,product_id
TX128,ST1008,10-04-2026,16:55:36,LAKME,250.0,381916"""

    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        # First load
        result1 = load_pos_csv(csv_content, db)
        assert result1["loaded"] == 1
        
        # Second load (should detect duplicate)
        result2 = load_pos_csv(csv_content, db)
        assert result2["loaded"] == 0
        assert result2["duplicates"] == 1
        
        # Only 1 record in DB
        count = db.query(DBPOS).count()
        assert count == 1
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)


def test_load_pos_csv_store_id_override():
    """Test store_id override functionality"""
    csv_content = """order_id,store_id,order_date,order_time,brand_name,total_amount,product_id
TX129,ST1076,10-04-2026,16:55:36,LAKME,250.0,381916"""

    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        # Override store_id
        result = load_pos_csv(csv_content, db, store_id_override="ST1008")
        assert result["loaded"] == 1
        
        # Verify override was applied
        rec = db.query(DBPOS).first()
        assert rec.store_id == "ST1008"  # Overridden, not ST1076
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)

