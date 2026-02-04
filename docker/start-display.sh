#!/bin/bash
# Start virtual display and VNC server
# This script is called by the entrypoint

set -e

# Configuration with defaults
DISPLAY_NUM=${DISPLAY_NUM:-99}
WIDTH=${DISPLAY_WIDTH:-1920}
HEIGHT=${DISPLAY_HEIGHT:-1080}
DEPTH=${DISPLAY_DEPTH:-24}

export DISPLAY=:${DISPLAY_NUM}

echo "Starting virtual display on :${DISPLAY_NUM} (${WIDTH}x${HEIGHT}x${DEPTH})"

# Start Xvfb (X Virtual Framebuffer)
Xvfb :${DISPLAY_NUM} \
    -screen 0 ${WIDTH}x${HEIGHT}x${DEPTH} \
    -ac \
    +extension GLX \
    +render \
    -noreset &

XVFB_PID=$!
echo "Xvfb started with PID ${XVFB_PID}"

# Wait for Xvfb to be ready
sleep 1

# Start a lightweight window manager (helps with some apps)
if command -v fluxbox &> /dev/null; then
    fluxbox -display :${DISPLAY_NUM} &
    echo "Fluxbox window manager started"
    sleep 0.5
fi

# Start x11vnc for remote viewing
echo "Starting VNC server on port 5900..."
x11vnc \
    -display :${DISPLAY_NUM} \
    -forever \
    -shared \
    -rfbport 5900 \
    -nopw \
    -xkb \
    -noxrecord \
    -noxfixes \
    -noxdamage \
    -bg

echo "VNC server started on port 5900"
echo "Connect with: vncviewer localhost:5900"
