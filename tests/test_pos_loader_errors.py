import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.database import Base, DBPOS
from app.pos_loader import load_pos_csv, get_transaction_times, get_brand_transaction_map, parse_pos_datetime

@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()

def test_parse_pos_datetime_invalid():
    assert parse_pos_datetime("invalid_date", "invalid_time") is None

def test_load_pos_csv_invalid_date(db_session):
    csv_content = "order_id,order_date,order_time,store_id,product_id,brand_name,total_amount\n1,invalid,invalid,ST1008,p1,brand,10.0"
    result = load_pos_csv(csv_content, db_session)
    assert result["errors"] == 1
    assert result["loaded"] == 0

def test_load_pos_csv_exception_during_row_process(db_session, monkeypatch):
    csv_content = "order_id,order_date,order_time,store_id,product_id,brand_name,total_amount\n1,01-01-2023,10:00:00,ST1008,p1,brand,10.0"
    
    def mock_parse(*args, **kwargs):
        raise Exception("Mock error")
        
    monkeypatch.setattr("app.pos_loader.parse_pos_datetime", mock_parse)
    result = load_pos_csv(csv_content, db_session)
    assert result["errors"] == 1

def test_load_pos_csv_db_commit_error(db_session, monkeypatch):
    csv_content = "order_id,order_date,order_time,store_id,product_id,brand_name,total_amount\n1,01-01-2023,10:00:00,ST1008,p1,brand,10.0"
    
    def mock_commit():
        raise Exception("DB Commit Failed")
        
    monkeypatch.setattr(db_session, "commit", mock_commit)
    result = load_pos_csv(csv_content, db_session)
    assert result["errors"] == 1  # 1 loaded + 0 errors initially, then +loaded to errors in except block
    assert result["loaded"] == 0

def test_get_transaction_times_invalid_timestamp(db_session):
    db_session.add(DBPOS(order_id="1", store_id="ST1", timestamp="bad_ts", brand_name="B", total_amount=10, product_id="P", order_date="D", order_time="T"))
    db_session.commit()
    times = get_transaction_times("ST1", db_session)
    assert len(times) == 0

def test_get_brand_transaction_map_invalid_timestamp(db_session):
    db_session.add(DBPOS(order_id="1", store_id="ST1", timestamp="bad_ts", brand_name="B", total_amount=10, product_id="P", order_date="D", order_time="T"))
    db_session.commit()
    bmap = get_brand_transaction_map("ST1", db_session)
    assert len(bmap) == 0

def test_load_pos_csv_value_error(db_session):
    csv_content = "order_id,order_date,order_time,store_id,product_id,brand_name,total_amount\n1,01-01-2023,10:00:00,ST1008,p1,brand,invalid_float"
    result = load_pos_csv(csv_content, db_session)
    assert result["loaded"] == 1
    
def test_load_pos_csv_loyalty(db_session):
    csv_content = "order_id,order_date,order_time,store_id,product_id,brand_name,total_amount\n1,01-01-2023,10:00:00,ST1008,p1,purplle,1.0"
    result = load_pos_csv(csv_content, db_session)
    assert result["skipped_loyalty"] == 1

def test_load_pos_csv_duplicates(db_session):
    csv_content = "order_id,order_date,order_time,store_id,product_id,brand_name,total_amount\n1,01-01-2023,10:00:00,ST1008,p1,brand,10.0\n1,01-01-2023,10:00:00,ST1008,p1,brand,10.0"
    result = load_pos_csv(csv_content, db_session)
    assert result["loaded"] == 1
    assert result["duplicates"] == 1

