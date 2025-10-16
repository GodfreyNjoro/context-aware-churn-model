#!/bin/bash

echo "=========================================="
echo "Churn Prediction System - Deployment"
echo "=========================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed"
    echo "Please install Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

echo "✓ Docker found: $(docker --version)"

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "Error: docker-compose is not installed"
    echo "Please install docker-compose: https://docs.docker.com/compose/install/"
    exit 1
fi

echo "✓ docker-compose found: $(docker-compose --version)"

# Build and start containers
echo ""
echo "Building Docker images..."
docker-compose build

echo ""
echo "Starting containers..."
docker-compose up -d

# Wait for services to be ready
echo ""
echo "Waiting for services to start..."
sleep 10

# Check service health
echo ""
echo "Checking service health..."
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✓ API service is healthy"
else
    echo "⚠ API service may not be ready yet. Check logs with: docker-compose logs"
fi

echo ""
echo "=========================================="
echo "✓ Deployment complete!"
echo "=========================================="
echo ""
echo "API is running at: http://localhost:8000"
echo "API documentation: http://localhost:8000/docs"
echo ""
echo "Useful commands:"
echo "  View logs:        docker-compose logs -f"
echo "  Stop services:    docker-compose down"
echo "  Restart services: docker-compose restart"
echo ""
