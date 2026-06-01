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
| Data exfiltration | Principle of least privilege for DB credentials, private networks |
| API abuse / DoS | Rate limiting, max ingest batch size, autoscaling |
| Dependency vulnerabilities | Dependabot + CodeQL scanning |
| Secret leakage | Secret scanning in CI, avoid secrets in repo |

## Residual Risk

This repository ships reference defaults. Production deployments should add auth (JWT/OAuth), IP allowlists, and encrypted storage according to local compliance requirements.
