from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, Any, Literal
import re

EventType = Literal[
    "ENTRY", "EXIT", 
    "ZONE_ENTER", "ZONE_EXIT", "ZONE_DWELL", 
    "BILLING_QUEUE_JOIN", "BILLING_QUEUE_ABANDON", 
    "REENTRY"
]

class EventMetadata(BaseModel):
    queue_depth: Optional[int] = None
    sku_zone: Optional[str] = None
    session_seq: int

class EventSchema(BaseModel):
    event_id: str = Field(..., description="UUID-v4 globally unique identifier")
    store_id: str = Field(..., description="Store ID, e.g., STORE_BLR_002")
    camera_id: str = Field(..., description="Camera ID producing the event")
    visitor_id: str = Field(..., description="Visitor Re-ID token")
    event_type: EventType = Field(..., description="The type of the retail event")
    timestamp: str = Field(..., description="ISO-8601 UTC timestamp")
    zone_id: Optional[str] = Field(None, description="Name of the zone, null for ENTRY/EXIT")
    dwell_ms: int = Field(0, description="Dwell duration in milliseconds")
    is_staff: bool = Field(False, description="Flag indicating if the subject is store staff")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence score bounded [0.0, 1.0]")
    metadata: EventMetadata

    @field_validator("timestamp")
    @classmethod
    def validate_iso_timestamp(cls, v):
        pattern = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$"
        if not re.match(pattern, v):
            raise ValueError("Timestamp must be in strict ISO-8601 UTC format (e.g. YYYY-MM-DDTHH:MM:SSZ)")
        return v
