#!/bin/bash

# Run tests inside Docker container
# Usage: ./run-tests.sh [pytest-args]
# Examples:
#   ./run-tests.sh                      # Run all tests
#   ./run-tests.sh tests/test_metrics.py::test_staff_exclusion_in_metrics
#   ./run-tests.sh -v --cov            # Run with coverage

set -e

# Ensure docker is available
if ! command -v docker &> /dev/null; then
    echo "❌ docker not found. Please install Docker."
    exit 1
fi

echo "🧪 Running tests in Docker container..."
echo ""

# Build the image
docker build -f api.Dockerfile -t store-intelligence-test . > /dev/null 2>&1

# Run tests
docker run --rm -v "$(pwd):/workspace" -w /workspace store-intelligence-test \
    sh -c "pip install -q -r requirements-dev.txt && python -m pytest tests/ $@ -v"

echo ""
echo "✅ Test run complete"
