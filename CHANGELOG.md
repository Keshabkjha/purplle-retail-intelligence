# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- CI workflows for linting, typing, testing, and Docker builds.
- Security automation (CodeQL, Dependabot, gitleaks) and threat model docs.
- Pre-commit tooling and operational documentation.
## [2.0.0] - 2026-06-09

### Added
- **Production Infrastructure:** Multi-stage Docker builds, non-root user execution, and container health checks.
- **Enhanced Security:** Implemented `X-Request-ID`, `X-XSS-Protection`, improved CSP, and strict `store_id` path validation.
- **Premium UI:** Redesigned landing page (`/`) with a glassmorphism interface and comprehensive social integrations.
- **CI/CD Hardening:** Added Codecov coverage reporting, Trivy container scanning, SBOM generation, and automated GitHub Release workflows.
- **Documentation:** Full author attribution, Hugging Face deployment badges, and enhanced security advisory channels.

### Changed
- Switched default `RESET_DB_ON_STARTUP` behavior to `0` to protect persistent data.
- Split requirements to provide a lightweight API-only image.
- Updated `sitemap.xml` and `robots.txt` for better SEO indexing.
### Changed
- Consolidated dependency management with lockfiles.
