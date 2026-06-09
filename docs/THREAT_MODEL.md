# Threat Model

## Assets

- Visitor event stream (PII-adjacent behavioral data)
- Store performance KPIs and anomaly signals
- API availability (ingest + dashboard)

## Assumptions

- Ingest endpoints are called by trusted pipeline components.
- Databases are protected by infrastructure-level controls.
- HTTPS termination is handled by the ingress/proxy layer.

## Threats & Mitigations

| Threat | Mitigation |
|---|---|
| Unauthorized event injection | Input validation, UUID enforcement, rate limiting, network ACLs |
| Write endpoint abuse | Optional `API_KEY` protection for ingest, POS load, and simulation endpoints |
| Data exfiltration | Principle of least privilege for DB credentials, private networks |
| API abuse / DoS | Rate limiting, max ingest batch size, request body size guard, autoscaling |
| Host-header abuse | Optional `ALLOWED_HOSTS` allowlist |
| Browser injection / clickjacking | CSP, `X-Frame-Options`, `X-Content-Type-Options`, referrer policy |
| Dependency vulnerabilities | Dependabot + CodeQL scanning |
| Secret leakage | Secret scanning in CI, avoid secrets in repo |

## Residual Risk

This repository ships reference defaults plus lightweight API-key protection for write/admin endpoints. Production deployments should still add user auth (JWT/OAuth or identity-aware proxy), IP allowlists, shared rate limiting, HTTPS-only access, encrypted storage, and retention controls according to local compliance requirements.
