# System Architecture & Design — Purplle Retail Intelligence Hub

> **Author:** Keshab Kumar ([@keshabkjha](https://github.com/keshabkjha))  
> **Project:** Purplle Tech Challenge 2026 — Round 2  
> **Store:** Brigade Road, Bangalore (ID: `ST1008`)

---

## 1. Problem Statement

Purplle's 40+ physical stores generated zero structured behavioral data. While the e-commerce platform had mature analytics (funnel tracking, A/B tests, heatmaps), the physical stores were a complete blind spot. Store managers had no data-driven answers to:

- **How many unique customers** entered today vs. last week?
- **Which product zones** are dead spots that attract no browsing?
- **When does the billing queue** get so long that customers abandon it?
- **What is the true in-store conversion rate** vs. the POS transaction count?

This system converts existing CCTV infrastructure into a real-time intelligence sensor network — requiring **zero hardware changes** and **zero customer opt-in**.

---

## 2. High-Level Architecture

```mermaid
graph TB
    subgraph CV["🎥 Computer Vision Pipeline (pipeline/)"]
        A[CCTV Video Clips\nMultiple Cameras] --> B[YOLO11n\nPerson Detection]
        B --> C[ByteTrack\nPersistent Session Tracking]
        C --> D[Homography Warp\nCamera → Floor Plan]
        D --> E[Ray Casting\nPoint-in-Polygon Zone Check]
        E --> F[Session State Machine\nvisitor lifecycle tracking]
    end

    subgraph EVENTS["⚡ Structured Event Stream"]
        F --> G[ENTRY / REENTRY\nZONE_ENTER / ZONE_EXIT\nZONE_DWELL / EXIT\nBILLING_QUEUE_JOIN\nBILLING_QUEUE_ABANDON]
    end

    subgraph API["🧠 Analytics API (app/)"]
        G -->|POST /events/ingest| H[FastAPI Server\napp/main.py]
        H --> I[(SQLite\nstore_intelligence.db)]
        J[POS Transactions CSV] -->|Seeded at startup| I
        I --> K[metrics.py\nKPI Aggregation]
        I --> L[funnel.py\nConversion Funnel]
        I --> M[heatmap.py\nZone Intensity]
        I --> N[anomalies.py\nAnomaly Detection]
    end

    subgraph UI["📊 Live Dashboard"]
        K & L & M & N --> O[GET /dashboard\nReal-time Operations UI\nPolls every 2 seconds]
    end
```

---

## 3. Data Flow Sequence

```mermaid
sequenceDiagram
    participant CAM as CCTV Camera
    participant DET as detect.py
    participant API as FastAPI
    participant DB as SQLite DB
    participant DASH as Dashboard

    loop Every video frame (throttled @ every 15th frame)
        CAM->>DET: Video frame
        DET->>DET: YOLO11n detect + ByteTrack IDs
        DET->>DET: Homography warp foot position
        DET->>DET: Ray cast → zone_id

        alt First appearance
            DET->>API: POST ENTRY (new visitor)
        end
        alt Previously exited
            DET->>API: POST REENTRY
        end
        alt Zone changed from BILLING to non-exit
            DET->>API: POST BILLING_QUEUE_ABANDON
        end
        alt Zone changed
            DET->>API: POST ZONE_EXIT + ZONE_ENTER
        end
        alt Same zone, 30s elapsed
            DET->>API: POST ZONE_DWELL
        end
        alt Track lost > 15 seconds
            DET->>API: POST EXIT
        end

        API->>DB: INSERT events (idempotent via event_id)
    end

    loop Every 2 seconds
        DASH->>API: GET /stores/ST1008/metrics
        DASH->>API: GET /stores/ST1008/funnel
        DASH->>API: GET /stores/ST1008/heatmap
        DASH->>API: GET /stores/ST1008/anomalies
        API->>DB: SQL aggregation queries
        API->>DASH: JSON response
        DASH->>DASH: Animate KPI cards + heatmap
    end
```

---

## 4. Spatial Mapping & Cross-Camera Re-ID Pipeline

The core technical challenge: mapping a 2D bounding box in camera pixel coordinates to a logical store zone on a 2D floor plan, and maintaining identity continuity as visitors walk between camera feeds.

```
Camera Frame (1920 × 1080 px)         Store Floor Plan (940 × 451 px)
┌───────────────────────────┐          ┌────────────────────┐
│                           │          │  ENTRY │ KOREAN │  │
│   [Person BBox]           │  ─────►  │        │        │  │
│   foot @ (px, py)         │ WARP     │  LAKME │ BILLING│  │
│                           │          └────────────────────┘
└───────────────────────────┘
         │
         ▼
   Homography Matrix H
   (4-point correspondence)
         │
         ▼
   cv2.perspectiveTransform()
         │
         ▼
   Floor coordinates (wx, wy)
         │
         ├───► Ray Casting: point_in_polygon(wx, wy, zone_polygon) ───► zone_id = "EB_KOREAN" | "BILLING" | None
         │
         └───► Cross-Camera Session Matcher (Spatial-Temporal Proximity)
```

**Calibration**: 4 corresponding real-world points are clicked in both the camera frame and the floor plan image using `pipeline/calibrate.py`. OpenCV calculates the 3×3 homography matrix `H`.  
**Fallback**: If no calibration exists for a camera, a proportional linear scaling matrix is used.

### 4.1 Multi-Signal Cross-Camera Re-ID

Since different video streams are processed sequentially by our background pipeline, the system tracks identity continuity across camera boundaries using a persistent, file-based state log (`pipeline/session_state.json`) and a unified multi-signal matching function.

When a new track (person) appears in any camera:
1. It registers the local `track_id` and checks the global session list.
2. It extracts an **appearance-based visual signature** from the person's bounding box crop using a 3D HSV color histogram (3D bins across Hue, Saturation, and Value channels).
3. It filters matching candidate sessions from **other cameras** that were active within a **30-second window** ($|t_{\text{current}} - t_{\text{last\_seen}}| \le 30$ seconds).
4. For each candidate session, it calculates a **Unified Match Score** ($S_{\text{match}} \in [0, 1]$) combining five signals and a lightweight online-learned identity classifier:
   - **Spatial Proximity (40% weight)**: Based on homography-mapped 2D floor plan Euclidean distance. If the distance $d \le 150$ pixels (~2.5 meters), $S_{\text{spatial}} = 1 - (d / 150)$, else $0$.
   - **Temporal Closeness (30% weight)**: Based on the absolute time delta $dt \le 30$ seconds. $S_{\text{temporal}} = 1 - (dt / 30)$.
   - **Visual Appearance Correlation (30% weight)**: Computed via the correlation of the HSV color histograms ($S_{\text{visual}} = \text{compare\_appearance}(\text{hist}_1, \text{hist}_2)$).
   - **Camera Transition Prior (12% weight)**: Uses the store's camera topology and known movement paths to reward plausible inter-camera transitions and penalize impossible ones.
   - **Zone Compatibility Prior (10% weight)**: Compares the candidate session's last seen zone with the current zone to keep identity continuity aligned with real retail movement.
   - **Learned Match Probability (blended 42%)**: An online logistic classifier learns from high-confidence pseudo-labels generated during the run, improving over the raw heuristic score as more clips are processed.
5. If the unified match score $S_{\text{match}} \ge 0.65$, it selects the best matching candidate. The local track is assigned the existing unified `visitor_id`, and subsequent events are logged under this ID. To handle camera angle and lighting variations, it continuously performs a rolling update of the visual signature ($70\%$ historical average + $30\%$ current frame signature).
6. If no match is found, a brand-new unified ID is created.
7. When `"entry_camera"` starts processing, the session file is automatically reset to start a fresh tracking run.

---

## 5. Staff Detection & Exclusions

Staff members wear identifiable dark uniforms. Rather than relying entirely on a static clothing check, the system uses a **Hybrid Visual-Behavioral Classifier** that combines visual features with physical behaviors to achieve high accuracy:

```
                  ┌────────────────────────────────────────┐
                  │ Upper Torso Crop (top 50% of BBox)     │
                  └───────────────────┬────────────────────┘
                                      │
                                      ▼
                      [HSV Color Space Transformation]
                      Lower Black: [0, 0, 0] | Upper Black: [180, 255, 50]
                                      │
                                      ▼
                        Torso Uniform Match Ratio
                                      │
               ┌──────────────────────┴──────────────────────┐
               ▼                                             ▼
        [Match Ratio > 30%]                          [Match Ratio <= 30%]
         Base Score: 0.6                              Base Score: 0.0
               │                                             │
               └──────────────────────┬──────────────────────┘
                                      │
                                      ▼
                      [Add Behavioral Confidence Boosts]
                      +0.4: Position behind register counter (wx > 820)
                      +0.3: High dwell in BILLING area (>90 seconds)
                      +0.3: Long cumulative presence (>180 seconds)
                      +0.2: Seen in >=2 different cameras
                                      │
                                      ▼
                       Final Staff Score (0.0 to 1.8)
                                      │
                                      ▼
                        Is Staff Score >= 0.5?
                                 /      \
                               Yes       No
                             /              \
                     [STAFF = True]         [STAFF = False]
```

### 5.1 Robust Behavioral Rules

1. **Torso Uniform Check**: Crops the top 50% of the person's bounding box, converts to the HSV color space, and applies a dark color mask. A match ratio $>30\%$ yields a base confidence score of `0.6`.
2. **Billing Counter Check**: Staff registers are located behind the counter at $w_x > 820$ inside the BILLING zone. Standing in this area adds a `0.4` boost.
3. **Queue Dwell Check**: A shopper in a queue moves steadily. A staff member standing still behind the counter for $>90$ seconds without buying adds a `0.3` boost.
4. **Presence duration**: Staff members remain active in the store far longer than average shoppers. An active session $>180$ seconds adds `0.3`, and presence in $\ge 2$ different cameras adds `0.2`.

When the unified `staff_score` exceeds the learned threshold, they are flagged as staff. The system also trains a lightweight online staff classifier from high-confidence pseudo-labels so the decision becomes more stable over time. This status is persistently logged inside the global tracker state so it carries over to other cameras and all session events (including `EXIT` and `REENTRY`), filtering them out of conversion metrics, heatmap displays, and retail KPIs.

---

## 6. Metric Calculation Logic

### 6.1 Conversion Rate

```mermaid
flowchart LR
    A[All Events\nis_staff=False] --> B[Group by visitor_id]
    B --> C{Did visitor\nvisit BILLING\nzone?}
    C -->|Yes| D{Is there a POS\ntransaction within\n5 min of billing visit?}
    C -->|No| E[Not Converted]
    D -->|Yes| F[✅ Converted Visitor]
    D -->|No| G[❌ Not Converted]
    F --> H[Conversion Rate =\nConverted ÷ Total Visitors × 100]
```

### 6.2 Anomaly Detection Windows

| Anomaly | Data Window | Method |
|---|---|---|
| Queue Spike | All historical `BILLING_QUEUE_JOIN` events | Statistical: depth > μ + 1.5σ |
| Conversion Drop | **Last 7 days** of events + POS transactions | Comparison to 7-day rolling baseline |
| Dead Zone | **Last 30 minutes** of events | Zero visits to retail zones |

---

## 7. Engineering Decisions (AI-Assisted)

### Decision 0: YOLO Model Version — v8 vs. v11 vs. v12

| Model | Released | Params | CPU Speed | GPU Req. | ByteTrack API | Selected |
|---|---|---|---|---|---|---|
| YOLOv8n | Jan 2023 | 3.2M | ~6ms/frame | No | Native, stable | — |
| **YOLO11n** | Oct 2024 | 2.6M | ~5ms/frame ✅ | No | Native, stable | ✅ |
| YOLOv12n | Feb 2025 | 6.5M | ~18ms/frame ⚠️ | Flash Attn | Changed API | ❌ |

**Context:** The system was initially built with YOLOv8n and has been upgraded to YOLO11n — Ultralytics' latest stable release with 22% fewer parameters, same drop-in API, and better CPU efficiency.

**Why YOLOv8, not v11 or v12:**

1. **ByteTrack stability**: `model.track(persist=True, tracker="bytetrack.yaml")` — YOLOv8's tracking API is the most battle-tested. YOLOv12 moved the `boxes.id` attribute across minor versions.

2. **CPU-only inference**: The challenge runs on standard hardware without GPU. YOLOv12 uses Flash Attention in its R-ELAN backbone, which has no CPU fallback — making it **3× slower** than v8 on CPU.

3. **Self-contained weight file**: `yolov8n.pt` (6MB) ships with the repository. YOLO12 weights are 9MB+ and the model format requires a newer Ultralytics version that has breaking API changes.

4. **Marginal accuracy gain**: For pedestrian detection in retail CCTV footage (640px input, ~50–200 people/session), the mAP difference between v8 and v11 is **less than 1.5%** — not meaningful at this scale.

**YOLO11 upgrade path** (when deploying to production with more resources):
```python
# Upgrade is a single line — same API, 22% fewer parameters
model = YOLO("yolo11n.pt")  # Drop-in replacement for yolov8n.pt
```

**Decision**: Upgraded to YOLO11n — same drop-in API as v8, 22% fewer parameters (2.6M vs 3.2M), and marginally faster CPU inference (~5ms vs ~6ms/frame). YOLO12 remains excluded due to GPU-only Flash Attention requirements.

---

### Decision 1: Zone Detection — Shapely vs. Pure Python Ray Casting


| Option | Pros | Cons | Selected |
|---|---|---|---|
| `shapely.geometry` | Clean API, robust edge cases | Heavy GEOS native dependency, Docker compile failures | ❌ |
| Pure Python Ray Casting | Zero deps, instant Docker build | Manual implementation | ✅ |

**Rationale:** Shapely's underlying GEOS C library frequently causes `pip install` failures in Alpine Linux Docker containers. The pure Python Ray Casting algorithm is O(n) on polygon vertices and adds negligible latency for the small zone polygons in this store layout.

### Decision 2: SQLite vs. PostgreSQL

| Option | Pros | Cons | Selected |
|---|---|---|---|
| PostgreSQL | Production-grade, JSON operators | Requires separate container, complex setup | ❌ |
| SQLite + SQLAlchemy | Zero-config, single file, fast reads | No concurrent writes, limited scale | ✅ |

**Rationale:** For a hackathon evaluation with a single store and ~2,000 events/day, SQLite provides adequate performance. The SQLAlchemy ORM abstraction means migrating to PostgreSQL for production would require only a connection string change.

### Decision 3: POS Correlation — Probabilistic vs. Time Window

| Option | Pros | Cons | Selected |
|---|---|---|---|
| Probabilistic ML matching | More accurate | Requires labeled training data | ❌ |
| 5-minute time window | Deterministic, explainable | Some false positives | ✅ |

**Rationale:** The 5-minute correlation window (billing zone visit → POS transaction) is the industry standard for brick-and-mortar analytics systems and requires no labeled training data.

### Decision 4: In-Memory Test Isolation — Standard vs. StaticPool

| Option | Pros | Cons | Selected |
|---|---|---|---|
| Standard `sqlite:///:memory:` | Simple | Each connection gets separate DB; tables disappear between test calls | ❌ |
| `StaticPool` + `check_same_thread=False` | Single persistent connection shared across TestClient calls | Slightly non-standard | ✅ |

---

## 8. API Middleware Stack

```
HTTP Request
    │
    ▼
[CORS Middleware]
Allows all origins for hackathon cross-origin dashboard access
    │
    ▼
[Structured Logging Middleware]
Injects trace_id, records latency_ms, event_count, status_code to stdout as JSON
    │
    ▼
[FastAPI Route Handler]
Validates Pydantic schema → SQLAlchemy query → JSON response
    │
    ▼
HTTP Response
```

---

## 9. Database Schema

```sql
-- Events table (core sensor data)
CREATE TABLE events (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id     TEXT UNIQUE NOT NULL,   -- Idempotency key
    store_id     TEXT NOT NULL,
    camera_id    TEXT NOT NULL,
    visitor_id   TEXT NOT NULL,
    event_type   TEXT NOT NULL,
    timestamp    TEXT NOT NULL,
    zone_id      TEXT,
    dwell_ms     INTEGER,
    is_staff     BOOLEAN DEFAULT FALSE,
    confidence   REAL,
    metadata_json JSON
);

-- POS transactions (seeded from CSV at startup)
CREATE TABLE pos_transactions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    txn_id       TEXT UNIQUE NOT NULL,
    store_id     TEXT NOT NULL,
    timestamp    TEXT NOT NULL,
    amount       REAL
);

-- Indexes for optimized query performance
CREATE INDEX idx_store_staff ON events(store_id, is_staff);
CREATE INDEX idx_visitor     ON events(visitor_id);
CREATE INDEX idx_timestamp   ON events(timestamp);
CREATE INDEX idx_event_type  ON events(event_type);
```

---

*Built by [Keshab Kumar](https://github.com/keshabkjha) for the Purplle Tech Challenge 2026.*
