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

    # ID is event_id which guarantees idempotency via unique/primary key constraints
    event_id = Column(String, primary_key=True, index=True)
    store_id = Column(String, index=True, nullable=False)
    camera_id = Column(String, nullable=False)
    visitor_id = Column(String, index=True, nullable=False)
    event_type = Column(String, nullable=False)
    timestamp = Column(String, nullable=False)
    zone_id = Column(String, nullable=True)
    dwell_ms = Column(Integer, default=0)
    is_staff = Column(Boolean, default=False)
    confidence = Column(Float, nullable=False)
    metadata_json = Column(JSON, nullable=False)  # holds queue_depth, sku_zone, session_seq


class DBPOS(Base):
    __tablename__ = "pos_transactions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    order_id = Column(String, index=True)
    store_id = Column(String, index=True)
    timestamp = Column(String, index=True)  # Combined ISO-8601 date+time
    brand_name = Column(String)
    total_amount = Column(Float)


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
