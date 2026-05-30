<p align="center">
  <img src="docs/assets/banner.png" alt="Purplle Retail Intelligence Hub" width="100%">
</p>

<p align="center">
  <a href="https://github.com/keshabkjha/purplle-retail-intelligence/actions"><img alt="Tests" src="https://img.shields.io/badge/tests-12%20passed-brightgreen?style=for-the-badge&logo=pytest&logoColor=white"></a>
  <a href="#"><img alt="Python" src="https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white"></a>
  <a href="#"><img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white"></a>
  <a href="#"><img alt="YOLOv8" src="https://img.shields.io/badge/YOLOv8-Ultralytics-FF6B35?style=for-the-badge&logo=pytorch&logoColor=white"></a>
  <a href="#"><img alt="Docker" src="https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-MIT-purple?style=for-the-badge"></a>
</p>

<p align="center">
  <strong>An end-to-end AI-powered retail store operations platform.</strong><br>
  Converts raw CCTV footage into real-time KPIs: visitor funnels, dwell heatmaps, queue management, and anomaly detection.
</p>

<p align="center">
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-api-reference">API Reference</a> •
  <a href="#-event-contract">Event Contract</a> •
  <a href="#-tech-stack">Tech Stack</a> •
  <a href="docs/DESIGN.md">Design Doc</a>
</p>

---

## 🎯 What This System Does

Purplle's physical stores were a complete analytics blind spot. This system solves that by treating each CCTV camera as a real-time sensor. Every customer movement — from store entry to zone browsing to billing queue — is converted into a structured event stream, aggregated into business KPIs, and surfaced on a live operations dashboard.

| Capability | Description |
|---|---|
| 🎥 **Person Tracking** | YOLOv8 + ByteTrack identifies and tracks individuals across multi-camera store footage |
| 📐 **Spatial Mapping** | Homography perspective warp maps camera foot-positions to a 2D store floor plan |
| ⚡ **Real-time Ingestion** | Structured events posted to FastAPI REST endpoint at sub-100ms latency |
| 📊 **KPI Aggregation** | Conversion rate, dwell time per zone, queue depth, abandonment rate |
| 🚨 **Anomaly Detection** | Statistical queue spikes, 7-day conversion baseline drops, 30-min dead zones |
| 🗺️ **Live Dashboard** | Real-time operations control room with animated spatial heatmap |

---

## 🏗️ Architecture

<p align="center">
  <img src="docs/assets/architecture.png" alt="System Architecture" width="85%">
</p>

The system uses a **decoupled, event-driven architecture** that cleanly separates the resource-heavy computer vision layer from the low-latency API layer.

```mermaid
graph TD
    A[📹 CCTV Camera Feeds] --> B[pipeline/detect.py]
    B --> C{YOLOv8 + ByteTrack\nPerson Detection}
    C --> D[Homography Warp\nFloor Plan Mapping]
    D --> E{Point-in-Polygon\nZone Assignment}
    E --> F[Structured JSON Events\nENTRY / ZONE_ENTER / ZONE_DWELL\nZONE_EXIT / EXIT / REENTRY\nBILLING_QUEUE_JOIN / ABANDON]
    F -->|POST /events/ingest| G[FastAPI Web Server\napp/main.py]
    G --> H[(SQLite Database\nstore_intelligence.db)]
    G --> I[REST API Endpoints\n/metrics /funnel /heatmap /anomalies]
    I --> J[📊 Live Operations Dashboard\n/dashboard]
    K[🧾 POS Transaction CSV] -->|Seed at startup| H
```

### Component Breakdown

```
purplle-retail-intelligence/
├── pipeline/
│   ├── detect.py          # YOLOv8 tracker → structured event emitter
│   ├── calibrate.py       # Interactive homography calibration tool
│   └── run.sh             # Multi-camera parallel pipeline launcher
├── app/
│   ├── main.py            # FastAPI router + middleware + health checks
│   ├── database.py        # SQLAlchemy ORM models + POS seeder
│   ├── models.py          # Pydantic event schema validation
│   ├── metrics.py         # KPI aggregation (visitors, dwell, conversion)
│   ├── funnel.py          # Customer journey funnel computation
│   ├── heatmap.py         # Zone intensity heatmap + confidence scoring
│   ├── anomalies.py       # Statistical anomaly detection engine
│   └── dashboard.html     # Self-contained real-time operations UI
├── config/
│   ├── store_layout.json  # Zone polygon definitions (Brigade Road, Bangalore)
│   └── calibration.json   # Camera homography transform matrices
├── tests/
│   ├── test_pipeline.py   # Core metric + pipeline logic tests
│   ├── test_metrics.py    # KPI aggregation coverage
│   └── test_anomalies.py  # Anomaly detection scenario tests
└── docs/
    ├── DESIGN.md          # Architecture & AI decision rationale
    └── CHOICES.md         # Trade-off analysis & engineering reasoning
```

---

## 🚀 Quick Start

### Option A: Docker (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/keshabkjha/purplle-retail-intelligence.git
cd purplle-retail-intelligence

# 2. Start the API and database
docker compose up -d --build

# 3. Verify the API is live
curl http://localhost:8000/health

# 4. Run the CV pipeline on a video
python3 pipeline/detect.py "CCTV Footage/entry_camera.mp4"

# 5. Open the live dashboard
open http://localhost:8000/dashboard
```

### Option B: Local Development

```bash
# Install dependencies
pip install -r api-requirements.txt

# Start the API server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# In a separate terminal, run the pipeline
python3 pipeline/detect.py "CCTV Footage/entry_camera.mp4"
```

### Option C: Run Tests

```bash
# Run the full test suite (12 tests)
python3 -m pytest tests/ -v

# Expected output:
# tests/test_pipeline.py::test_entry_exit_metrics     PASSED
# tests/test_pipeline.py::test_staff_exclusion        PASSED
# tests/test_metrics.py::test_conversion_rate         PASSED
# tests/test_anomalies.py::test_billing_queue_spike   PASSED
# ... 12 passed in 2.5s ✅
```

---

## 📡 API Reference

Base URL: `http://localhost:8000`

Interactive docs available at: `http://localhost:8000/docs` (Swagger UI)

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Per-store health status, feed latency, stale detection |
| `POST` | `/events/ingest` | Bulk ingest structured visitor events (idempotent) |
| `GET` | `/stores/{id}/metrics` | KPIs: visitors, conversion rate, dwell time, queue depth |
| `GET` | `/stores/{id}/funnel` | 4-stage visitor conversion funnel with drop-off rates |
| `GET` | `/stores/{id}/heatmap` | Zone visit frequency, dwell seconds, intensity (0-100) |
| `GET` | `/stores/{id}/anomalies` | Active operational warnings with severity + actions |
| `GET` | `/dashboard` | Self-contained live operations control room UI |

### Sample: Metrics Response

```json
{
  "store_id": "ST1008",
  "unique_visitors": 87,
  "conversion_rate": 24.14,
  "average_dwell_minutes": 12.4,
  "average_dwell_per_zone": {
    "EB_KOREAN": 185.3,
    "LAKME": 142.7,
    "BILLING": 97.2
  },
  "current_queue_depth": 3,
  "abandonment_rate": 8.2
}
```

### Sample: Anomalies Response

```json
{
  "store_id": "ST1008",
  "anomalies": [
    {
      "anomaly_type": "STATISTICAL_QUEUE_SPIKE",
      "severity": "CRITICAL",
      "suggested_action": "Open additional billing counter immediately.",
      "details": "Queue depth 9 exceeds statistical threshold of 6.2 (Avg: 3.1)."
    },
    {
      "anomaly_type": "DEAD_ZONE",
      "severity": "INFO",
      "suggested_action": "Inspect product display in the MINIMALIST zone.",
      "details": "Zone 'MINIMALIST' has received 0 customer visits in the past 30 minutes."
    }
  ]
}
```

---

## 📋 Event Contract

All events posted to `POST /events/ingest` must conform to this schema:

| Field | Type | Required | Description |
|---|---|---|---|
| `event_id` | `UUID` | ✅ | Unique event identifier (idempotency key) |
| `store_id` | `string` | ✅ | Store identifier (e.g. `ST1008`) |
| `camera_id` | `string` | ✅ | Source camera ID (e.g. `CAM_ENTRY_01`) |
| `visitor_id` | `string` | ✅ | Tracker-assigned session ID (e.g. `VIS_42`) |
| `event_type` | `enum` | ✅ | See event type table below |
| `timestamp` | `ISO8601` | ✅ | Event UTC timestamp |
| `zone_id` | `string \| null` | ✅ | Target zone identifier |
| `dwell_ms` | `integer` | ✅ | Time spent in zone (milliseconds) |
| `is_staff` | `boolean` | ✅ | Whether visitor is staff (excluded from metrics) |
| `confidence` | `float` | ✅ | YOLO detection confidence score (0.0–1.0) |
| `metadata.queue_depth` | `integer \| null` | — | Current billing queue length |
| `metadata.sku_zone` | `string \| null` | — | Product category in current zone |
| `metadata.session_seq` | `integer` | — | Sequential event counter per visitor session |

### Event Types

| Event Type | Trigger |
|---|---|
| `ENTRY` | Visitor appears for the first time in the store |
| `REENTRY` | Visitor who previously exited reappears |
| `ZONE_ENTER` | Visitor moves into a new retail zone |
| `ZONE_DWELL` | Visitor still in zone after every 30-second interval |
| `ZONE_EXIT` | Visitor leaves a zone |
| `BILLING_QUEUE_JOIN` | Visitor enters the BILLING zone |
| `BILLING_QUEUE_ABANDON` | Visitor leaves BILLING zone without completing purchase |
| `EXIT` | Visitor disappears from all camera feeds (session end) |

---

## 🔍 Detection Pipeline

```mermaid
sequenceDiagram
    participant V as Video Frame
    participant Y as YOLOv8 Tracker
    participant H as Homography Mapper
    participant Z as Zone Checker
    participant SM as Session Manager
    participant API as FastAPI /ingest

    V->>Y: Frame (every 15 frames)
    Y->>H: Bounding boxes + track_ids
    H->>Z: Warped floor coordinates
    Z->>SM: zone_id per visitor

    alt New visitor
        SM->>API: POST ENTRY or REENTRY
    end

    alt Zone changed
        alt Leaving BILLING without exit
            SM->>API: POST BILLING_QUEUE_ABANDON
        end
        SM->>API: POST ZONE_EXIT (old zone)
        SM->>API: POST ZONE_ENTER or BILLING_QUEUE_JOIN
    end

    alt Still in same zone (30s elapsed)
        SM->>API: POST ZONE_DWELL
    end

    alt Track lost for > 15 seconds
        SM->>API: POST EXIT
    end
```

---

## 📊 KPI Calculation Logic

```mermaid
flowchart LR
    A[Raw Events DB] --> B[Filter: is_staff = False]
    B --> C[Group by visitor_id]
    C --> D[Unique Visitors Count]
    C --> E[Dwell Time: last_event - first_event]
    C --> F[Billing Visits: BILLING_QUEUE_JOIN events]

    G[POS Transactions DB] --> H[Parse timestamps]
    F --> I{Match billing visit\nto POS transaction\nwithin 5-min window?}
    H --> I
    I -->|Yes| J[Converted Visitor]
    I -->|No| K[Abandoned Visitor]

    J --> L[Conversion Rate = Converted / Total]
    K --> M[Abandonment Rate = Abandoned / Billing]
    E --> N[Avg Dwell = Total Dwell / Unique Visitors]
```

---

## 🚨 Anomaly Detection Rules

| Anomaly | Trigger Condition | Severity | Action |
|---|---|---|---|
| `STATISTICAL_QUEUE_SPIKE` | Queue depth > mean + 1.5σ over all history | `WARN` / `CRITICAL` | Open additional billing counter |
| `CONVERSION_DROP` | Current rate < 70% of 7-day rolling baseline | `WARN` | Check checkout bottlenecks |
| `DEAD_ZONE` | Zero visits to a retail zone in past **30 minutes** | `INFO` | Inspect product display & visibility |

---

## 🛠️ Tech Stack

| Layer | Technology | Rationale |
|---|---|---|
| **Object Detection** | YOLOv8n (Ultralytics) | Best accuracy/speed tradeoff on CPU hardware |
| **Object Tracking** | ByteTrack (built-in) | Low-ID-switch rate for persistent session tracking |
| **Spatial Mapping** | OpenCV Homography | Perspective-correct floor projection from camera |
| **Zone Logic** | Pure Python Ray Casting | Zero native dependencies vs. Shapely/GEOS |
| **API Framework** | FastAPI + Uvicorn | Async, typed, auto-documented REST API |
| **ORM & DB** | SQLAlchemy + SQLite | Zero-config persistence for hackathon scope |
| **Data Validation** | Pydantic v2 | Schema enforcement on every ingest event |
| **Containerization** | Docker + Compose | Reproducible single-command deployment |
| **Testing** | Pytest + TestClient | In-memory SQLite isolation per test fixture |
| **Logging** | Structured JSON (stdout) | Trace IDs, latency, event counts per request |

---

## 🧪 Test Coverage

```
tests/
├── test_pipeline.py     # Entry/exit metrics, staff exclusion
├── test_metrics.py      # Conversion rate, dwell time, queue depth
└── test_anomalies.py    # Queue spike (WARN + CRITICAL), conversion drop, dead zones

Total: 12 tests | Status: ✅ All Passing
```

Run with:
```bash
python3 -m pytest tests/ -v --tb=short
```

---

## 🗂️ Store Layout (Brigade Road, Bangalore — Store ID: ST1008)

```
┌─────────────────────────────────────────────────────┐
│  ENTRY   │  EB_KOREAN │ THE_FACE_SHOP │  MINIMALIST │
│  PORTAL  ├────────────┼──────────────┼─────────────┤  BILLING /
│  [CAM_   │   LAKME    │  MAYBELLINE  │  GOOD_VIBES │  CHECKOUT
│  ENTRY]  │            │              │             │  [CAM_BILLING]
└──────────┴────────────┴──────────────┴─────────────┘
     [CAM_MAIN_01]        [CAM_MAIN_02]  [CAM_MAIN_03]
```

Zone coordinates are stored in `config/store_layout.json` as polygon arrays matched to the `Revised.png` floor plan.

---

## 📐 Calibration Tool

To map camera footage to the floor plan for a new store:

```bash
# Step 1: Collect source points from camera frame
python3 pipeline/calibrate.py "CCTV Footage/entry_camera.mp4"

# Step 2: Collect destination points from floor plan image
python3 pipeline/calibrate.py Revised.png

# Step 3: Save to config/calibration.json
{
  "CAM_ENTRY_01": {
    "src": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
    "dst": [[mx1,my1], [mx2,my2], [mx3,my3], [mx4,my4]]
  }
}
```

> **Note:** The pipeline automatically falls back to sensible linear scaling matrices if a camera is not yet calibrated, so the system works out-of-the-box without calibration.

---

## 🚢 Deployment

### Render.com (Recommended)

1. Connect your GitHub repository on [render.com](https://render.com)
2. Set **Build Command**: `pip install -r api-requirements.txt`
3. Set **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port 10000`
4. Dashboard will be live at `https://your-service.onrender.com/dashboard`

---

## 📄 Documentation

| Document | Description |
|---|---|
| [DESIGN.md](docs/DESIGN.md) | Full architecture overview, AI-assisted decision log, trade-off rationale |
| [CHOICES.md](docs/CHOICES.md) | Library selection reasoning, engineering trade-offs |

---

## 👤 Author

**Keshab Kumar** — [@keshabkjha](https://github.com/keshabkjha)

> Built for the **Purplle Tech Challenge 2026 – Round 2**

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
