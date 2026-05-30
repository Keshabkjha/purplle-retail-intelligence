# Store Intelligence System Architecture Design (DESIGN.md)

Apex Retail operates 40 physical stores across 8 cities. While their online channels benefit from mature, real-time analytics, physical brick-and-mortar stores have remained a critical blind spot. This Store Intelligence System addresses this by converting raw security CCTV footage into actionable, offline business intelligence metrics in real-time.

---

## 1. System Architecture Overview

The system is designed with a decoupled, event-driven architecture, cleanly dividing the resource-heavy computer vision detection layer from the fast, consumer-facing REST API layer.

```
       📹 [CCTV Video Clips] 
                 │
                 ▼  (pipeline/detect.py)
       🔍 [YOLOv8 + ByteTrack] 
                 │
                 ▼  (Perspective Warp & Zone Checking)
       ⚡ [Structured JSON Events]
                 │
                 ▼  (POST /events/ingest)
       🧠 [FastAPI Web Server] (app/main.py)
                 │
                 ├─────────────────────────┐
                 ▼                         ▼
         [(SQLite DB)]              📊 [Live Dashboard]
     (store_intelligence.db)
```

### Components

1. **Computer Vision & Spatial Mapping Layer (`/pipeline`)**:
   - **Person Detection & Tracking**: Utilizes YOLOv8 (specifically `yolov8n.pt` for optimal CPU performance during local execution) combined with ByteTrack for real-time person detection and session-aware trajectory tracking.
   - **Homography Perspective Warp**: Maps camera bounding-box foot positions (bottom-center of bounding boxes) to precise coordinates on a 2D store floor plan using calculated projective transformation matrices.
   - **Spatial Zone Checker**: Executes an optimized Ray Casting point-in-polygon algorithm to check which logical store zones (e.g. `EB_KOREAN`, `BILLING`) contain the visitor's warped coordinates.
   - **State Machine & Ingest Client**: Maintains visitor session lifecycles to emit structured behavioral events (`ENTRY`, `EXIT`, `ZONE_ENTER`, `ZONE_EXIT`, `ZONE_DWELL`, `BILLING_QUEUE_JOIN`) and posts them directly to the FastAPI server.

2. **Analytics REST API (`/app`)**:
   - **Data Store**: Uses a lightweight, high-performance SQLite engine configured with composite indexing for sub-millisecond query response times.
   - **Aggregator Modules**: Computes real-time store-level KPIs, multi-stage conversion funnels, dwell-time heatmaps, and rule-based operational anomalies dynamically from raw events and POS transaction logs.
   - **Structured Logging Middleware**: Injects a unique `trace_id` per request and outputs JSON logs recording request latency, endpoints, status codes, and database payload sizes.

---

## 2. AI-Assisted Decisions

During the development process, an LLM was actively used to brainstorm architectural patterns, evaluate library overhead, and sanity-check edge-case algorithms.

### Decision 1: Spatial Zone Checking Library vs. Pure Python
- **Context**: Selecting how to run point-in-polygon tests on warped coordinates.
- **LLM Proposal**: The LLM suggested using `shapely.geometry.Polygon` for robust spatial geometry calculations.
- **Our Action (Override)**: While `shapely` is clean, it introduces heavy native C-library dependencies (`GEOS`), which often fail or slow down during Docker container compilation on various developer operating systems. Instead, we implemented a pure Python Ray Casting algorithm for point-in-polygon checks. It has zero external dependencies, compiles instantly, and runs with negligible latency.

### Decision 2: In-Memory SQLite Testing connection overrides
- **Context**: Structuring the pytest database fixture to test FastAPI endpoints concurrently.
- **LLM Proposal**: The LLM suggested standard in-memory connection URLs: `sqlite:///:memory:`.
- **Our Action (Refinement)**: During implementation, standard connection pooling in SQLAlchemy spawned multiple separate connections, each getting a distinct in-memory SQLite sandbox, which caused mock data table missing errors. We refined the LLM's design by introducing SQLAlchemy's `StaticPool` to enforce a single, persistent connection shared across all API calls in test environments.

### Decision 3: POS Transaction Customer Correlation
- **Context**: Matching anonymous transaction records in POS CSV with camera visitor sessions.
- **LLM Proposal**: The LLM suggested building a probabilistic matching algorithm using billing exit timestamps and order values.
- **Our Action (Agreed)**: We agreed with the LLM's recommendation to use a 5-minute pre-transaction correlation window matching store layouts, which is the industry standard for matching transaction systems with spatial camera tracking.
