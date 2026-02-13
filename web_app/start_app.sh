#!/bin/bash
# Startup script for the NetworkX Graph Visualizer web app

echo "🚀 Starting NetworkX Graph Visualizer..."
echo "📍 Navigate to: http://localhost:5000"
echo "⚠️  Press Ctrl+C to stop the server"
echo ""

cd "$(dirname "$0")"
uv run python app.py