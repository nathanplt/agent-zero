#!/bin/bash
# AgentZero Container Entrypoint
# Starts virtual display, VNC server, and then runs the provided command

set -e

echo "AgentZero Container Starting..."
echo "================================"

# Start the display services
/start-display.sh

# Wait for display to be ready
echo "Waiting for display to be ready..."
timeout 10 bash -c 'until xdpyinfo -display :99 >/dev/null 2>&1; do sleep 0.5; done' || {
    echo "ERROR: Display failed to start"
    exit 1
}
echo "Display is ready!"

# Print environment info
echo ""
echo "Environment:"
echo "  DISPLAY=$DISPLAY"
echo "  Python: $(python --version)"
echo "  VNC: Port 5900"
echo ""
echo "================================"
echo "Running command: $@"
echo "================================"
echo ""

# Execute the provided command
exec "$@"
