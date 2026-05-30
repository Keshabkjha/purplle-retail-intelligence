from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime

class EventMetadata(BaseModel):
    queue_depth: Optional[int] = None
    sku_zone: Optional[str] = None
    session_seq: int

class EventSchema(BaseModel):
    event_id: str = Field(..., description="UUID-v4 globally unique identifier")
    store_id: str = Field(..., description="Store ID, e.g., STORE_BLR_002")
    camera_id: str = Field(..., description="Camera ID producing the event")
    visitor_id: str = Field(..., description="Visitor Re-ID token")
    event_type: str = Field(..., description="ENTRY, EXIT, ZONE_ENTER, ZONE_EXIT, ZONE_DWELL, BILLING_QUEUE_JOIN, BILLING_QUEUE_ABANDON, REENTRY")
    timestamp: str = Field(..., description="ISO-8601 UTC timestamp")
    zone_id: Optional[str] = Field(None, description="Name of the zone, null for ENTRY/EXIT")
    dwell_ms: int = Field(0, description="Dwell duration in milliseconds")
    is_staff: bool = Field(False, description="Flag indicating if the subject is store staff")
    confidence: float = Field(..., description="Detection confidence score")
    metadata: EventMetadata
