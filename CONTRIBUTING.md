# Contributing to Purplle Retail Intelligence Hub

Thank you for your interest in contributing! This document provides guidelines to get you started quickly.

---

## 🚀 Development Setup

```bash
# 1. Fork and clone the repository
git clone https://github.com/keshabkjha/purplle-retail-intelligence.git
cd purplle-retail-intelligence

# 2. Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install all dependencies
pip install -r requirements-dev.txt

# 4. Run quality checks
ruff check . --fix
ruff format .
mypy

# 5. Run the test suite to confirm everything works
python3 -m pytest
```

### Updating Dependencies

Edit `requirements.in` / `requirements-dev.in` and regenerate lockfiles:

```bash
pip-compile requirements.in --output-file=requirements.txt
pip-compile requirements-dev.in --output-file=requirements-dev.txt
```

---

## 🌿 Branch Conventions

| Branch Pattern | Purpose |
|---|---|
| `main` | Stable, deployable code |
| `feat/description` | New features |
| `fix/description` | Bug fixes |
| `docs/description` | Documentation updates |
| `test/description` | Test additions |

---

## ✅ Pull Request Checklist

Before submitting a PR, confirm:

- [ ] All 26 tests pass (`python3 -m pytest tests/ -v`)
- [ ] New features include corresponding unit tests
- [ ] API changes are reflected in `README.md` endpoint table
- [ ] New event types are added to the Event Contract table in `README.md`
- [ ] Code follows existing style (PEP 8, type hints on functions)

---

## 🧪 Writing Tests

Tests live in `tests/` and use `pytest` with FastAPI's `TestClient` and an in-memory SQLite `StaticPool` fixture.

```python
# Example test structure
def test_my_new_feature(client, db_session):
    # 1. Seed test data into db_session
    db_session.add(DBEvent(...))
    db_session.commit()

    # 2. Hit the endpoint
    response = client.get("/stores/ST1008/metrics")
    assert response.status_code == 200

    # 3. Assert expected values
    data = response.json()
    assert data["unique_visitors"] == 1
```

---

## 📝 Commit Message Format

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add REENTRY event detection to pipeline
fix: correct staff exclusion on EXIT event
docs: update API reference table in README
test: add conversion drop anomaly test case
refactor: extract zone dwell logic into helper
```

---

## 🧹 Pre-commit Hooks

```bash
pre-commit install
pre-commit run --all-files
```

---

*Questions? Open an issue or reach out to [@keshabkjha](https://github.com/keshabkjha).*
