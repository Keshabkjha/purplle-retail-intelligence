"""
Event models for Store Intelligence API.

Two schemas are supported:
1. EventSchema — the PDF-spec scoring harness schema (used by POST /events/ingest)
2. Purplle native event schemas — matching actual sample_events.jsonl format
   - PurplleEntryExitEvent  (event_type: entry / exit)
   - PurplleZoneEvent       (event_type: zone_entered / zone_exited)
   - PurplleQueueEvent      (event_type: queue_completed / queue_abandoned)
"""

import re
from datetime import datetime, timezone
from typing import Literal, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ---------------------------------------------------------------------------
# Scoring-harness schema (PDF spec) — used by POST /events/ingest
# ---------------------------------------------------------------------------

EventType = Literal[
    "ENTRY",
    "EXIT",
    "ZONE_ENTER",
    "ZONE_EXIT",
    "ZONE_DWELL",
    "BILLING_QUEUE_JOIN",
    "BILLING_QUEUE_ABANDON",
    "REENTRY",
    # Purplle native snake_case aliases (accepted for cross-compat)
    "entry",
    "exit",
    "zone_entered",
    "zone_exited",
    "zone_dwell",
    "queue_completed",
    "queue_abandoned",
    "reentry",
]


def canonical_event_type(event_type: str) -> str:
    mapping = {
        "entry": "ENTRY",
        "exit": "EXIT",
        "zone_entered": "ZONE_ENTER",
        "zone_exited": "ZONE_EXIT",
        "zone_dwell": "ZONE_DWELL",
        "queue_completed": "BILLING_QUEUE_JOIN",
        "queue_abandoned": "BILLING_QUEUE_ABANDON",
        "reentry": "REENTRY",
    }
    return mapping.get(event_type, event_type)


class EventMetadata(BaseModel):
    model_config = ConfigDict(extra="allow", str_strip_whitespace=True)

    queue_depth: Optional[int] = Field(default=None, ge=0)
    sku_zone: Optional[str] = None
    session_seq: int = Field(default=0, ge=0)


class EventSchema(BaseModel):
    model_config = ConfigDict(extra="allow", str_strip_whitespace=True)

    event_id: UUID = Field(..., description="UUID-v4 globally unique identifier")
    store_id: str = Field(..., min_length=1, description="Store ID, e.g., ST1008")
    camera_id: str = Field(..., min_length=1, description="Camera ID producing the event")
    visitor_id: str = Field(..., min_length=1, description="Visitor Re-ID token")
    event_type: EventType = Field(..., description="Structured retail event type")
    timestamp: datetime = Field(..., description="ISO-8601 UTC timestamp")
    zone_id: Optional[str] = Field(None, description="Zone ID, null for ENTRY/EXIT")
    dwell_ms: int = Field(0, ge=0, description="Dwell duration in milliseconds")
    is_staff: bool = Field(False, description="Flag indicating store staff")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Detection confidence score [0.0, 1.0]"
    )
    metadata: EventMetadata

    @field_validator("timestamp", mode="before")
    @classmethod
    def validate_iso_timestamp(cls, v):
        if not isinstance(v, str):
            raise ValueError("Timestamp must be a string")
        pattern = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$"
        if not re.match(pattern, v):
            raise ValueError(
                "Timestamp must be in strict ISO-8601 UTC format (e.g. YYYY-MM-DDTHH:MM:SSZ)"
            )
        return v

    @field_validator("timestamp", mode="after")
    @classmethod
    def normalize_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)


# ---------------------------------------------------------------------------
# Purplle native event schemas — matching actual sample_events.jsonl
# ---------------------------------------------------------------------------

AgeGroup = Literal["18-24", "25-34", "35-44", "45-54", "55+", "unknown"]
GenderPred = Literal["M", "F", "unknown"]


class PurplleEntryExitEvent(BaseModel):
    """
    Entry or exit event from the entry camera.
    Matches the format in sample_events.jsonl lines 1-4.
    """
    model_config = ConfigDict(extra="allow", str_strip_whitespace=True)

    event_type: Literal["entry", "exit"]
    id_token: str = Field(..., description="Unique visitor Re-ID token, e.g. ID_60001")
    store_code: str = Field(..., description="Store code, e.g. store_1076")
    camera_id: str
    event_timestamp: str = Field(..., description="ISO-8601 timestamp with microseconds")
    is_staff: bool = False
    gender_pred: Optional[GenderPred] = "unknown"
    age_pred: Optional[int] = Field(None, ge=0, le=100)
    age_bucket: Optional[AgeGroup] = "unknown"
    is_face_hidden: bool = False
    group_id: Optional[str] = None
    group_size: Optional[int] = Field(None, ge=1)


class PurplleZoneEvent(BaseModel):
    """
    Zone enter/exit event from zone floor cameras.
    Matches the format in sample_events.jsonl lines 5-10.
    """
    model_config = ConfigDict(extra="allow", str_strip_whitespace=True)

    event_type: Literal["zone_entered", "zone_exited"]
    track_id: int = Field(..., description="Integer tracker ID within this camera session")
    store_id: str = Field(..., description="Store ID, e.g. ST1076")
    camera_id: str
    zone_id: str = Field(..., description="Zone ID, e.g. PURPLLE_MUM_1076_Z01")
    zone_name: str
    zone_type: Literal["SHELF", "DISPLAY", "BILLING", "ENTRY", "KIOSK", "CONSULTATION"]
    is_revenue_zone: Literal["Yes", "No"] = "Yes"
    event_time: str = Field(..., description="ISO-8601 timestamp with microseconds")
    zone_hotspot_x: float = Field(..., description="X coordinate of detection centroid in frame")
    zone_hotspot_y: float = Field(..., description="Y coordinate of detection centroid in frame")
    gender: Optional[GenderPred] = "unknown"
    age: Optional[int] = Field(None, ge=0, le=100)
    age_bucket: Optional[AgeGroup] = "unknown"
    is_staff: bool = False


class PurplleQueueEvent(BaseModel):
    """
    Billing queue lifecycle event — single event covering join + served + exit.
    Matches the format in sample_events.jsonl lines 11-13.
    """
    model_config = ConfigDict(extra="allow", str_strip_whitespace=True)

    queue_event_id: str = Field(..., description="UUID for the queue event")
    event_type: Literal["queue_completed", "queue_abandoned"]
    track_id: int
    store_id: str
    camera_id: str
    zone_id: str
    zone_name: str
    zone_type: Literal["BILLING"] = "BILLING"
    is_revenue_zone: Literal["Yes", "No"] = "Yes"
    queue_join_ts: str = Field(..., description="When visitor joined the queue")
    queue_served_ts: Optional[str] = Field(None, description="When they reached counter (null if abandoned)")
    queue_exit_ts: str = Field(..., description="When visitor left the queue area")
    wait_seconds: int = Field(..., ge=0, description="Total wait time in seconds")
    queue_position_at_join: int = Field(..., ge=1, description="Position in queue when joined")
    abandoned: bool
    zone_hotspot_x: float
    zone_hotspot_y: float
    gender: Optional[GenderPred] = "unknown"
    age: Optional[int] = Field(None, ge=0, le=100)
    age_bucket: Optional[AgeGroup] = "unknown"
    is_staff: bool = False
