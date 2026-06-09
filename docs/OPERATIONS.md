# Operations Guide

## Configuration

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `sqlite:///./store_intelligence.db` | Database connection string |
| `ENV` | `dev` | Environment name |
| `ALLOWED_ORIGINS` | `*` | Comma-separated CORS origins |
| `ALLOWED_HOSTS` | unset | Optional comma-separated Host header allowlist |
| `APP_PUBLIC_BASE_URL` | `http://localhost:8000` | Canonical public URL used by sitemap and SEO metadata |
| `API_KEY` | unset | Enables API-key protection for write/admin endpoints when set |
| `MAX_REQUEST_BODY_BYTES` | `5242880` | Maximum accepted HTTP request body size |
| `ENABLE_HSTS` | `0` | Sends HSTS header when `1`; also enabled automatically when `ENV=prod` |
| `RATE_LIMIT_PER_MINUTE` | `120` | Per-IP request limit |
| `RATE_LIMIT_WINDOW_SECONDS` | `60` | Rate-limit window in seconds |
| `MAX_INGEST_BATCH` | `500` | Maximum events per ingest request |
| `INGEST_URL` | `http://localhost:8000/events/ingest` | Pipeline ingest endpoint |
| `RESET_DB_ON_STARTUP` | `1` | Resets default local SQLite DB on startup; use `0` for persistent deployments |

## Health, Readiness & Liveness

- `GET /health`: Database connectivity plus per-store feed staleness.
- `GET /ready`: Same as `/health` (readiness probe).
- `GET /live`: Basic liveness indicator for orchestration.

## Observability

- Logs are structured JSON on stdout, suitable for log aggregation.
- To enable distributed tracing, integrate with OpenTelemetry (`opentelemetry-instrumentation-fastapi`) and export to your tracing backend.
- Export API metrics via your preferred middleware (Prometheus, Datadog, etc.)—no vendor lock-in is baked in.

## Security Controls

- Security headers are added on every response: CSP, `X-Content-Type-Options`, `X-Frame-Options`, referrer policy, and permissions policy.
- Set `API_KEY` in production. Protected endpoints are `POST /events/ingest`, `POST /api/load-pos`, and `POST /api/simulate`.
- Send API keys with `X-API-Key: <key>` or `Authorization: Bearer <key>`.
- Set `ALLOWED_HOSTS` for public deployments to reduce Host-header abuse.
- Set `MAX_REQUEST_BODY_BYTES` and `MAX_INGEST_BATCH` based on expected event payload sizes.
- Use HTTPS at the platform or reverse-proxy layer. Set `ENABLE_HSTS=1` after confirming the domain is HTTPS-only.

## SEO & Public Metadata

- `GET /` serves a SEO-focused landing page for Purplle Retail Intelligence by Keshab Kumar `@keshabkjha`.
- `GET /robots.txt` advertises the sitemap.
- `GET /sitemap.xml` uses `APP_PUBLIC_BASE_URL` to emit canonical URLs.
- `GET /site.webmanifest` exposes installable web app metadata.
- `GET /api/project-profile` returns machine-readable project, author, and official profile links.
- For Hugging Face Spaces, set `APP_PUBLIC_BASE_URL` to the active Space URL or custom domain.

## Production Checklist

- Lock dependencies with `pip-compile`.
- Configure `ALLOWED_ORIGINS` explicitly for your frontend domains.
- Configure `ALLOWED_HOSTS`, `APP_PUBLIC_BASE_URL`, `API_KEY`, `MAX_REQUEST_BODY_BYTES`, `RATE_LIMIT_PER_MINUTE`, and `MAX_INGEST_BATCH` for expected traffic.
- Set `RESET_DB_ON_STARTUP=0` for persistent deployments.
- Run Uvicorn with multiple workers behind a reverse proxy (e.g., `--workers 2`).

> Note: The built-in rate limiter is in-memory per instance. For distributed deployments, use a shared store (Redis) or edge rate limiting.
