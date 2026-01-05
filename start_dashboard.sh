#!/bin/bash
# Quick start script for RootCNN v2 Dashboard

echo "======================================"
echo "RootCNN v2 Dashboard - Quick Start"
echo "======================================"
echo ""

# Check if sample logs exist, if not create them
if [ ! -f "output/logs/sample_detection.json" ]; then
    echo "Generating sample log files..."
    env/bin/python example_logging.py
    echo ""
fi

echo "Starting dashboard..."
echo "Dashboard will be available at: http://127.0.0.1:8050"
echo ""
echo "Press Ctrl+C to stop the server"
echo "======================================"
echo ""

# Start the dashboard
env/bin/python dashboard.py
