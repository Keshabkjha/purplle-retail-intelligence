#!/bin/bash

# Start the API using docker-compose
# Usage: ./run-api.sh

set -e

# Ensure docker-compose is available
if ! command -v docker-compose &> /dev/null && ! command -v docker compose &> /dev/null; then
    echo "❌ docker-compose not found. Please install Docker."
    exit 1
fi

# Use either docker compose or docker-compose
if command -v docker-compose &> /dev/null; then
    DOCKER_COMPOSE="docker-compose"
else
    DOCKER_COMPOSE="docker compose"
fi

echo "🚀 Starting Store Intelligence API via Docker..."
echo ""
echo "API will be available at: http://localhost:8000"
echo "Dashboard: http://localhost:8000/dashboard"
echo "Health: http://localhost:8000/health"
echo ""
echo "Press Ctrl+C to stop the API"
echo ""

# Start the API service
$DOCKER_COMPOSE up api

echo ""
echo "✅ API stopped"
