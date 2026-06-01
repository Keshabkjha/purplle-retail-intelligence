# Roadmap: Purplle Retail Intelligence

This document outlines the strategic vision, planned enhancements, and technical debt for the Purplle Retail Intelligence project.

## Current Features (v1.0)
- ✅ Real-time CCTV ingestion and behavioral spatial mapping.
- ✅ Hybrid Re-ID (Spatiotemporal + Visual HSV histograms).
- ✅ Live Operations Dashboard with interactive heatmaps.
- ✅ Store anomalies (statistical queue spikes, dead zones, conversion drops).
- ✅ Zero-hardware integration architecture.

## Q3 2026: Scale and Analytics Depth
- [ ] **Multi-Tenancy Support:** Move from SQLite to PostgreSQL to support 40+ stores concurrently in a single deployed instance.
- [ ] **Redis Pub/Sub:** Offload the real-time WebSocket dashboard streaming from standard polling to a Redis event bus.
- [ ] **YOLOv12 Upgrade:** Investigate upgrading edge nodes with entry-level GPUs to support YOLOv12 with Flash Attention.
- [ ] **Automated Calibration:** Replace the manual 4-point homography calibration script (`calibrate.py`) with an auto-calibrating transformer model that understands retail shelving constraints.

## Q4 2026: Advanced Customer Insights
- [ ] **Demographic Analysis (Opt-in):** Add anonymized, aggregated age and gender demographic estimates.
- [ ] **Cross-Store Re-ID:** Implement a highly secure, hashed identity verification system to track customer loyalty across different physical locations without storing PII.
- [ ] **Actionable Alerting:** Integrate with store-manager mobile apps (via Push Notifications / WhatsApp) for instant anomaly resolution (e.g. "Open Register 3").

## Technical Debt & Maintenance
- **Native Dependency Management:** Shift from `pip` and `requirements.txt` to `Poetry` or `uv` for more deterministic production lockfiles.
- **Frontend Refactor:** Migrate the vanilla HTML/JS `dashboard.html` into a modern Next.js/React application for better component reusability and testing.
- **Coverage Scaling:** Increase unit test coverage from the current 75% baseline to >90% across the API layer.
