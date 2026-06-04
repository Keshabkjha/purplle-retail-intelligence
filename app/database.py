import os

from sqlalchemy import JSON, Boolean, Column, Float, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./store_intelligence.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


class DBEvent(Base):
    __tablename__ = "events"

    # Primary key is event_id — guarantees idempotency via unique/primary key constraint
    event_id = Column(String, primary_key=True, index=True)
    store_id = Column(String, index=True, nullable=False)
    camera_id = Column(String, nullable=False)
    visitor_id = Column(String, index=True, nullable=False)
    event_type = Column(String, nullable=False)
    timestamp = Column(String, index=True, nullable=False)
    zone_id = Column(String, nullable=True)
    dwell_ms = Column(Integer, default=0)
    is_staff = Column(Boolean, default=False)
    confidence = Column(Float, nullable=False)
    metadata_json = Column(JSON, nullable=False)  # queue_depth, sku_zone, session_seq

    # Purplle native enrichment fields (optional, stored when available)
    gender_pred = Column(String, nullable=True)
    age_pred = Column(Integer, nullable=True)
    age_bucket = Column(String, nullable=True)
    is_face_hidden = Column(Boolean, default=False, nullable=True)
    group_id = Column(String, nullable=True)
    group_size = Column(Integer, nullable=True)
    zone_hotspot_x = Column(Float, nullable=True)
    zone_hotspot_y = Column(Float, nullable=True)


class DBPOS(Base):
    __tablename__ = "pos_transactions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    order_id = Column(String, index=True)
    store_id = Column(String, index=True)
    timestamp = Column(String, index=True)   # ISO-8601 combined date+time
    brand_name = Column(String, nullable=True)
    total_amount = Column(Float, default=0.0)
    # Extended fields from actual CSV
    product_id = Column(String, nullable=True)
    order_date = Column(String, nullable=True)   # DD-MM-YYYY raw
    order_time = Column(String, nullable=True)   # HH:MM:SS raw


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
