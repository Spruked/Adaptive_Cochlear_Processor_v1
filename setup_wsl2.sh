#!/bin/bash
# ACP 1.0 WSL2 Setup Script
# Builds and starts Docker containers for stable ACP execution

set -e

echo "🚀 Setting up ACP 1.0 in WSL2 + Docker"

# Check if in WSL2
if [[ ! -f /proc/version ]] || ! grep -q "Microsoft" /proc/version; then
    echo "❌ This script must be run in WSL2"
    exit 1
fi

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Install Docker Desktop with WSL2 integration"
    exit 1
fi

# Build and start
echo "🏗️  Building Docker image..."
docker-compose build

echo "🚀 Starting services..."
docker-compose up -d acp

# Wait for container
echo "⏳ Waiting for container to be ready..."
sleep 5

# Run validation
echo "🔍 Running validation tests..."
docker-compose exec -T acp python validate_setup.py

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Setup complete! ACP 1.0 is ready."
    echo ""
    echo "Next steps:"
    echo "  • Test mic: docker-compose exec acp python realtime/mic_capture.py 3"
    echo "  • Check learning: docker-compose exec acp python check_metrics.py"
    echo "  • Push-to-talk: docker-compose exec acp python realtime/keyboard_gate.py"
else
    echo ""
    echo "⚠️  Setup completed but some tests failed."
    echo "Check the output above and fix issues before proceeding."
    exit 1
fi