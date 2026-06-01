import json
import os
import sys
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Deque, Dict, List, Tuple

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.anomalies import get_store_anomalies_data
from app.database import DBEvent, get_db, init_db
from app.funnel import get_store_funnel_data
from app.heatmap import get_store_heatmap_data
from app.metrics import get_store_metrics_data, parse_timestamp
from app.models import EventSchema


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield

app = FastAPI(title="Store Intelligence API", lifespan=lifespan)

MAX_INGEST_BATCH = int(os.getenv("MAX_INGEST_BATCH", "500"))
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "120"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))

allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "*")
ALLOWED_ORIGINS = [origin.strip() for origin in allowed_origins_env.split(",") if origin.strip()] or ["*"]

class RateLimiter:
    def __init__(self, limit: int, window_seconds: int) -> None:
        self.limit = limit
        self.window_seconds = window_seconds
        self.hits: Dict[str, Deque[float]] = defaultdict(deque)
        self.lock = Lock()

    def allow(self, key: str) -> Tuple[bool, int]:
        now = time.time()
        with self.lock:
            window = self.hits[key]
            while window and now - window[0] > self.window_seconds:
                window.popleft()
            if len(window) >= self.limit:
                retry_after = max(1, int(self.window_seconds - (now - window[0])))
                return False, retry_after
            window.append(now)
        return True, 0

app.state.rate_limiter = RateLimiter(RATE_LIMIT_PER_MINUTE, RATE_LIMIT_WINDOW_SECONDS) if RATE_LIMIT_PER_MINUTE > 0 else None
app.state.max_ingest_batch = MAX_INGEST_BATCH

# Enable CORS for frontend flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Basic rate limiting middleware
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    limiter = request.app.state.rate_limiter
    if limiter and request.url.path not in ("/health", "/ready", "/live"):
        forwarded_for = request.headers.get("x-forwarded-for")
        forwarded_ip = forwarded_for.split(",")[0].strip() if forwarded_for else None
        client_ip = forwarded_ip or (request.client.host if request.client else "unknown")
        allowed, retry_after = limiter.allow(client_ip)
        if not allowed:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": "Rate limit exceeded"},
                headers={"Retry-After": str(retry_after)}
            )
    return await call_next(request)

# Structured logging middleware
@app.middleware("http")
async def structured_logging_middleware(request: Request, call_next):
    trace_id = str(uuid.uuid4())
    start_time = time.time()
    
    # Pre-populate state
    request.state.trace_id = trace_id
    request.state.store_id = None
    request.state.event_count = 0

    # Extract store_id from path if present
    path_parts = request.url.path.split("/")
    for i, part in enumerate(path_parts):
        if part == "stores" and i + 1 < len(path_parts):
            request.state.store_id = path_parts[i + 1]

    response = await call_next(request)
    
    latency_ms = int((time.time() - start_time) * 1000)
    
    log_record = {
        "trace_id": trace_id,
        "store_id": request.state.store_id,
        "endpoint": request.url.path,
        "latency_ms": latency_ms,
        "event_count": request.state.event_count,
        "status_code": response.status_code
    }
    
    sys.stdout.write(json.dumps(log_record) + "\n")
    sys.stdout.flush()
    return response

@app.get("/health")
def health_check(request: Request, db: Session = Depends(get_db)):
    try:
        db.execute(text("SELECT 1"))
        db_status = "healthy"
    except Exception:
        db_status = "unhealthy"
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "unhealthy", "database": "unavailable"}
        )

    # Build per-store health by querying the latest event per store_id
    last_ingest_time = getattr(request.app.state, "last_ingest_time", None)

    # Get distinct store IDs
    store_ids_rows = db.query(DBEvent.store_id).distinct().all()
    store_ids = [r[0] for r in store_ids_rows]

    stores_health = {}
    overall_stale = False

    for sid in store_ids:
        last_event = db.query(DBEvent).filter(
            DBEvent.store_id == sid
        ).order_by(DBEvent.timestamp.desc()).first()

        last_event_ts = None
        stale_feed = False

        if last_event:
            last_event_ts = last_event.timestamp
            dt = parse_timestamp(last_event_ts)
            if dt:
                if last_ingest_time:
                    lag_seconds = (datetime.now(timezone.utc).replace(tzinfo=None) - last_ingest_time).total_seconds()
                    if lag_seconds > 600:
                        stale_feed = True
                else:
                    now = datetime.now(timezone.utc).replace(tzinfo=None)
                    lag_seconds = (now - dt).total_seconds()
                    if lag_seconds > 600 and (now - dt).days < 1:
                        stale_feed = True

        stores_health[sid] = {
            "last_event_timestamp": last_event_ts,
            "stale_feed": stale_feed
        }
        if stale_feed:
            overall_stale = True

    # If no stores, fall back to global last event for backwards compatibility
    if not store_ids:
        last_event = db.query(DBEvent).order_by(DBEvent.timestamp.desc()).first()
        last_event_ts = last_event.timestamp if last_event else None
        return {
            "status": "healthy",
            "database": db_status,
            "last_event_timestamp": last_event_ts,
            "stale_feed": False,
            "stores": {}
        }

    return {
        "status": "healthy",
        "database": db_status,
        "stale_feed": overall_stale,
        "stores": stores_health
    }

@app.get("/ready")
def readiness_check(request: Request, db: Session = Depends(get_db)):
    return health_check(request, db)

@app.get("/live")
def liveness_check():
    return {"status": "alive"}


@app.post("/events/ingest", status_code=status.HTTP_207_MULTI_STATUS)
def ingest_events(events: List[Any], request: Request, db: Session = Depends(get_db)):
    request.state.event_count = len(events)
    
    if not events:
        return {
            "ingested": 0,
            "failed": 0,
            "errors": []
        }

    max_batch = request.app.state.max_ingest_batch
    if max_batch and len(events) > max_batch:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Batch too large. Max supported events per request is {max_batch}."
        )

    # Record wall-clock ingest time in app state to manage real-time stale feed health checks
    request.app.state.last_ingest_time = datetime.now(timezone.utc).replace(tzinfo=None)

    # Extract store_id for request state logging
    if len(events) > 0 and isinstance(events[0], dict):
        request.state.store_id = events[0].get("store_id")

    success_count = 0
    errors = []

    for event_data in events:
        if not isinstance(event_data, dict):
            errors.append({"event_id": "unknown", "error": "Event payload must be a JSON object"})
            continue

        event_id = event_data.get("event_id", "unknown")
        
        # Validate event schema manually using Pydantic
        try:
            event = EventSchema.model_validate(event_data)
        except Exception as ve:
            errors.append({"event_id": event_id, "error": f"Schema validation failed: {str(ve)}"})
            continue

        try:
            # Check if event already exists to maintain idempotency
            event_id_str = str(event.event_id)
            existing = db.query(DBEvent).filter(DBEvent.event_id == event_id_str).first()
            if existing:
                success_count += 1
                continue

            event_timestamp = event.timestamp.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            db_event = DBEvent(
                event_id=event_id_str,
                store_id=event.store_id,
                camera_id=event.camera_id,
                visitor_id=event.visitor_id,
                event_type=event.event_type.value,
                timestamp=event_timestamp,
                zone_id=event.zone_id,
                dwell_ms=event.dwell_ms,
                is_staff=event.is_staff,
                confidence=event.confidence,
                metadata_json={
                    "queue_depth": event.metadata.queue_depth,
                    "sku_zone": event.metadata.sku_zone,
                    "session_seq": event.metadata.session_seq
                }
            )
            db.add(db_event)
            db.commit()
            success_count += 1
        except IntegrityError:
            db.rollback()
            # If unique constraint triggered but didn't catch in query
            success_count += 1
        except Exception as e:
            db.rollback()
            errors.append({"event_id": event.event_id, "error": str(e)})

    return {
        "ingested": success_count,
        "failed": len(errors),
        "errors": errors
    }

@app.get("/stores/{store_id}/metrics")
def get_store_metrics(store_id: str, db: Session = Depends(get_db)):
    try:
        return get_store_metrics_data(store_id, db)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to calculate store metrics: {str(e)}"
        )

@app.get("/stores/{store_id}/funnel")
def get_store_funnel(store_id: str, db: Session = Depends(get_db)):
    try:
        return get_store_funnel_data(store_id, db)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to calculate store funnel: {str(e)}"
        )

@app.get("/stores/{store_id}/heatmap")
def get_store_heatmap(store_id: str, db: Session = Depends(get_db)):
    try:
        return get_store_heatmap_data(store_id, db)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to calculate store heatmap: {str(e)}"
        )

@app.get("/stores/{store_id}/anomalies")
def get_store_anomalies(store_id: str, db: Session = Depends(get_db)):
    try:
        return get_store_anomalies_data(store_id, db)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to calculate store anomalies: {str(e)}"
        )

@app.get("/dashboard", response_class=HTMLResponse)
def get_dashboard():
    import os
    dashboard_paths = ("app/dashboard.html", "/workspace/app/dashboard.html", "/Users/keshabkumar/Purpple Challenge/app/dashboard.html")
    for p in dashboard_paths:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read(), status_code=200)
    raise HTTPException(status_code=404, detail="Dashboard file not found.")
