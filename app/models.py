from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class EventType(str, Enum):
    ENTRY = "ENTRY"
    EXIT = "EXIT"
    ZONE_ENTER = "ZONE_ENTER"
    ZONE_EXIT = "ZONE_EXIT"
    ZONE_DWELL = "ZONE_DWELL"
    BILLING_QUEUE_JOIN = "BILLING_QUEUE_JOIN"
    BILLING_QUEUE_ABANDON = "BILLING_QUEUE_ABANDON"
    REENTRY = "REENTRY"


class EventMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    queue_depth: Optional[int] = Field(default=None, ge=0)
    sku_zone: Optional[str] = None
    session_seq: int = Field(..., ge=0)

class EventSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    event_id: UUID = Field(..., description="UUID-v4 globally unique identifier")
    store_id: str = Field(..., min_length=1, description="Store ID, e.g., STORE_BLR_002")
    camera_id: str = Field(..., min_length=1, description="Camera ID producing the event")
    visitor_id: str = Field(..., min_length=1, description="Visitor Re-ID token")
    event_type: EventType = Field(..., description="Structured retail event type")
    timestamp: datetime = Field(..., description="ISO-8601 UTC timestamp")
    zone_id: Optional[str] = Field(None, description="Name of the zone, null for ENTRY/EXIT")
    dwell_ms: int = Field(0, ge=0, description="Dwell duration in milliseconds")
    is_staff: bool = Field(False, description="Flag indicating if the subject is store staff")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence score")
    metadata: EventMetadata

    @field_validator("timestamp", mode="after")
    @classmethod
    def normalize_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
