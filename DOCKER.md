# Docker-Based Development & Testing Guide

This project uses **Docker exclusively** for development and testing, as required by the challenge specification. Follow this guide to run the Store Intelligence system.

## Quick Start (5 Commands)

```bash
# 1. Clone the repository
git clone <repo-url>
cd purplle-retail-intelligence

# 2. Build the Docker image
docker build -f api.Dockerfile -t store-intelligence .

# 3. Start the API via docker-compose
docker compose up api

# 4. In a new terminal, run tests
docker run --rm -v "$(pwd):/workspace" -w /workspace store-intelligence \
  sh -c "pip install -q -r requirements-dev.txt && python -m pytest tests/ -v"

# 5. Access the dashboard
# Visit: http://localhost:8000/dashboard
```

## Using Helper Scripts

### Run Tests (Recommended)

```bash
./run-tests.sh                                   # Run all tests
./run-tests.sh tests/test_metrics.py            # Run specific test file
./run-tests.sh -k test_staff_exclusion          # Run tests matching pattern
./run-tests.sh -v --cov                         # Run with coverage report
```

### Run API

```bash
./run-api.sh                                     # Starts API on port 8000
```

## Docker Architecture

### Build Image

```bash
docker build -f api.Dockerfile -t store-intelligence .
```

**Image Details:**
- Base: Python 3.11 slim
- Size: ~2.4GB (includes CUDA for GPU support)
- Dependencies: OpenCV, YOLO, FastAPI, SQLAlchemy, PyTorch
- System: ffmpeg, libgl1, build-essential

### Run API Service

```bash
docker run -p 8000:8000 -v "$(pwd):/workspace" store-intelligence \
  uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**API Endpoints:**
- `GET /health` - Health status
- `GET /dashboard` - Live metrics dashboard
- `POST /events/ingest` - Ingest events (idempotent)
- `GET /stores/{store_id}/metrics` - Store metrics
- `GET /stores/{store_id}/funnel` - Conversion funnel
- `GET /stores/{store_id}/heatmap` - Zone heatmap
- `GET /stores/{store_id}/anomalies` - Active anomalies

### Run Tests

```bash
# Via docker run
docker run --rm -v "$(pwd):/workspace" -w /workspace store-intelligence \
  sh -c "pip install -q -r requirements-dev.txt && python -m pytest tests/ -v"

# Via docker-compose
docker compose run --rm api sh -c "pip install -q -r requirements-dev.txt && python -m pytest tests/ -v"
```

## Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | `sqlite:///./store_intelligence.db` | SQLite database path |
| `ENV` | `dev` | Environment (dev/test/prod) |
| `PORT` | `8000` | API port |
| `RATE_LIMIT_PER_MINUTE` | `0` | Rate limiting (0 = disabled) |
| `MAX_INGEST_BATCH` | `500` | Max events per ingest request |
| `ALLOWED_ORIGINS` | `*` | CORS allowed origins |

## Running the Detection Pipeline

### Option 1: Docker Run

```bash
# Process a single video clip
docker run --rm -v "$(pwd):/workspace" -w /workspace store-intelligence \
  python pipeline/detect.py "Store 1/CAM 1 - zone.mp4"
```

### Option 2: Docker Interactive

```bash
# Start interactive container
docker run -it --rm -v "$(pwd):/workspace" -w /workspace store-intelligence bash

# Inside container
python pipeline/detect.py "Store 1/CAM 1 - zone.mp4"
```

## Troubleshooting

### Container won't start
```bash
# Check logs
docker-compose logs api

# Rebuild without cache
docker build --no-cache -f api.Dockerfile -t store-intelligence .
```

### Import errors
```bash
# Ensure requirements are installed
docker run --rm -v "$(pwd):/workspace" store-intelligence \
  pip install -r requirements.txt
```

### Database locked
```bash
# SQLite WAL mode should handle this, but if stuck:
rm store_intelligence.db*
```

### Tests won't run
```bash
# Install dev requirements and run
docker run --rm -v "$(pwd):/workspace" -w /workspace store-intelligence \
  sh -c "pip install -q pytest pytest-cov && python -m pytest tests/ -v"
```

## Docker Compose Usage

### Start entire stack

```bash
docker compose up
```

Services available:
- `api` on `http://localhost:8000`

### Stop services

```bash
docker compose down
```

### View logs

```bash
docker compose logs -f api
```

## Development Workflow

### 1. Make code changes
```bash
# Edit files locally - volumes mounted in Docker
vim app/main.py
```

### 2. Run tests to verify
```bash
./run-tests.sh -k "your_test_name"
```

### 3. Start API and check manually
```bash
./run-api.sh
# In another terminal
curl http://localhost:8000/health
```

### 4. Iterate until satisfied

## Performance Notes

- **API latency**: ~50-200ms per request (SQLite in WAL mode)
- **Test suite**: ~30-60 seconds (full coverage)
- **Detection pipeline**: ~5ms per frame at 1 FPS (YOLO11n on CPU)

## What's Happening Under the Hood

1. **Dockerfile** defines the environment (Python 3.11, dependencies)
2. **docker-compose.yml** orchestrates services (API, optional test service)
3. **Volumes** mount local code into container (live editing)
4. **Helper scripts** simplify common operations

All code runs inside isolated Docker containers - no local virtual environment needed.

## Next Steps

- Process CCTV clips: See [OPERATIONS.md](../docs/OPERATIONS.md)
- Understand architecture: See [DESIGN.md](../DESIGN.md)
- Review decisions: See [CHOICES.md](../CHOICES.md)
