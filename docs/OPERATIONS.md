# Operations Guide

## Configuration

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `sqlite:///./store_intelligence.db` | Database connection string |
| `ENV` | `dev` | Environment name |
| `ALLOWED_ORIGINS` | `*` | Comma-separated CORS origins |
| `RATE_LIMIT_PER_MINUTE` | `120` | Per-IP request limit |
| `RATE_LIMIT_WINDOW_SECONDS` | `60` | Rate-limit window in seconds |
| `MAX_INGEST_BATCH` | `500` | Maximum events per ingest request |
| `INGEST_URL` | `http://localhost:8000/events/ingest` | Pipeline ingest endpoint |

## Health, Readiness & Liveness

- `GET /health`: Database connectivity plus per-store feed staleness.
- `GET /ready`: Same as `/health` (readiness probe).
- `GET /live`: Basic liveness indicator for orchestration.

## Observability

- Logs are structured JSON on stdout, suitable for log aggregation.
- To enable distributed tracing, integrate with OpenTelemetry (`opentelemetry-instrumentation-fastapi`) and export to your tracing backend.
- Export API metrics via your preferred middleware (Prometheus, Datadog, etc.)—no vendor lock-in is baked in.

## Production Checklist

- Lock dependencies with `pip-compile`.
- Configure `ALLOWED_ORIGINS` explicitly for your frontend domains.
- Tune `RATE_LIMIT_PER_MINUTE` and `MAX_INGEST_BATCH` for expected traffic.
- Run Uvicorn with multiple workers behind a reverse proxy (e.g., `--workers 2`).

> Note: The built-in rate limiter is in-memory per instance. For distributed deployments, use a shared store (Redis) or edge rate limiting.
