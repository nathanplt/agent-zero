# Agent Zero Container
# Provides a graphical environment for running the Roblox game agent
#
# Features:
# - Python 3.11+
# - Virtual display (Xvfb on :99)
# - VNC server for remote viewing
# - Required X11 and browser dependencies

FROM python:3.11-slim-bookworm

# Set Python environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Display configuration
ENV DISPLAY=:99 \
    DISPLAY_WIDTH=1920 \
    DISPLAY_HEIGHT=1080 \
    DISPLAY_DEPTH=24

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Virtual display
    xvfb \
    # VNC server
    x11vnc \
    # X11 utilities
    x11-utils \
    x11-xserver-utils \
    xdotool \
    # Window manager (lightweight)
    fluxbox \
    # Fonts
    fonts-liberation \
    fonts-dejavu-core \
    # Browser dependencies (for Playwright/Chromium)
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    libpango-1.0-0 \
    libcairo2 \
    # Utilities
    procps \
    curl \
    wget \
    # Cleanup
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy project files
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install Python dependencies (including browser extras for Playwright)
RUN pip install -e ".[browser,vision]"

# Install Playwright browsers (Chromium)
# This downloads and installs the browser binaries
RUN playwright install chromium --with-deps

# Copy startup scripts
COPY docker/entrypoint.sh /entrypoint.sh
COPY docker/start-display.sh /start-display.sh
RUN chmod +x /entrypoint.sh /start-display.sh

# Expose VNC port
EXPOSE 5900

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD xdpyinfo -display :99 >/dev/null 2>&1 || exit 1

# Default entrypoint starts the display and runs the agent
ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "-c", "print('Agent Zero container ready. Override CMD to run your command.')"]
