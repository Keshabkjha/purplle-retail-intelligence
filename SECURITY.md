# Security Policy

## Supported Versions

Currently, the `main` branch is the only officially supported version receiving security updates.

| Version | Supported          |
| ------- | ------------------ |
| v1.0.x  | :white_check_mark: |
| < v1.0  | :x:                |

## Threat Model

Please review our [Threat Model](docs/THREAT_MODEL.md) document to understand the assumptions, boundaries, and accepted risks inherent to the Purplle Retail Intelligence architecture.

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability within Purplle Retail Intelligence, please do **NOT** open a public issue.

Instead, please send an email to `security@purplle-retail.example.com`. We will strive to acknowledge your report within 48 hours and will provide a timeline for a resolution.

### What to include in your report:
- A detailed description of the vulnerability.
- Steps to reproduce the issue.
- Potential impact and an assessment of severity.
- Any suggested mitigations (if available).

## Responsible Disclosure

We ask that you do not share or publish details of any unresolved vulnerabilities with third parties or the public until we have had time to investigate and issue a patch.

## Automated Scans

We employ GitHub Actions to automatically run:
- **Dependabot** for dependency vulnerability auditing.
- **CodeQL** for static application security testing (SAST).
- **Gitleaks** for secret detection.

Pull requests will not be merged if they fail any of these security gates.
