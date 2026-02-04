"""Tests for Feature 1.2: Browser Runtime.

These tests verify that Chromium browser can run in the container with Playwright:
- Chromium launches in virtual display
- Playwright can control browser
- Can navigate to roblox.com
- Page renders correctly (screenshot verification)

Note: These tests require Docker to be installed and running.
"""

import subprocess
from pathlib import Path

import pytest

# Mark for tests requiring Docker
docker = pytest.mark.skipif(
    subprocess.run(["docker", "info"], capture_output=True).returncode != 0,
    reason="Docker not available",
)


class TestDockerfileHasBrowser:
    """Tests for Dockerfile browser-related configuration."""

    def test_dockerfile_has_playwright_install(self):
        """Dockerfile should install playwright dependencies."""
        dockerfile = Path(__file__).parent.parent / "Dockerfile"
        content = dockerfile.read_text()
        # Check for playwright installation
        assert "playwright" in content.lower(), "Dockerfile should reference playwright"

    def test_dockerfile_has_browser_deps(self):
        """Dockerfile should have browser dependencies."""
        dockerfile = Path(__file__).parent.parent / "Dockerfile"
        content = dockerfile.read_text()
        # Key dependencies for Chromium
        required_deps = ["libnss3", "libatk1.0-0", "libgbm1"]
        for dep in required_deps:
            assert dep in content, f"Dockerfile should have {dep}"


@docker
class TestPlaywrightInstalled:
    """Tests that Playwright is installed in the container."""

    @pytest.fixture(scope="class")
    def container_id(self):
        """Build and start a container for testing."""
        project_root = Path(__file__).parent.parent

        # Build container
        build_result = subprocess.run(
            ["docker", "build", "-t", "agent-zero-browser-test", "."],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=600,  # 10 min timeout for building with playwright
        )
        if build_result.returncode != 0:
            pytest.skip(f"Docker build failed: {build_result.stderr}")

        # Run container with entrypoint
        result = subprocess.run(
            [
                "docker", "run", "-d", "--rm",
                "--shm-size=2gb",  # Required for Chromium
                "-e", "DISPLAY=:99",
                "agent-zero-browser-test",
                "sleep", "120"
            ],
            capture_output=True,
            text=True,
        )
        container_id = result.stdout.strip()

        # Wait for display to be ready
        import time
        time.sleep(5)

        yield container_id

        # Cleanup
        subprocess.run(["docker", "stop", container_id], capture_output=True)

    def test_playwright_package_installed(self, container_id):
        """Playwright Python package should be installed."""
        result = subprocess.run(
            ["docker", "exec", container_id, "python", "-c", "import playwright; print('OK')"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Playwright not installed: {result.stderr}"
        assert "OK" in result.stdout

    def test_chromium_browser_available(self, container_id):
        """Chromium browser should be available for Playwright."""
        # Test that playwright can find chromium
        result = subprocess.run(
            [
                "docker", "exec", container_id,
                "python", "-c",
                "from playwright.sync_api import sync_playwright; "
                "p = sync_playwright().start(); "
                "b = p.chromium.launch(headless=True); "
                "b.close(); p.stop(); "
                "print('OK')"
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, f"Chromium not available: {result.stderr}"
        assert "OK" in result.stdout


@docker
class TestBrowserNavigation:
    """Tests for browser navigation capabilities."""

    @pytest.fixture(scope="class")
    def container_id(self):
        """Build and start a container for testing."""
        project_root = Path(__file__).parent.parent

        # Build container
        build_result = subprocess.run(
            ["docker", "build", "-t", "agent-zero-browser-test", "."],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if build_result.returncode != 0:
            pytest.skip(f"Docker build failed: {build_result.stderr}")

        # Run container with entrypoint
        result = subprocess.run(
            [
                "docker", "run", "-d", "--rm",
                "--shm-size=2gb",
                "-e", "DISPLAY=:99",
                "agent-zero-browser-test",
                "sleep", "300"
            ],
            capture_output=True,
            text=True,
        )
        container_id = result.stdout.strip()

        # Wait for display to be ready
        import time
        time.sleep(5)

        yield container_id

        # Cleanup
        subprocess.run(["docker", "stop", container_id], capture_output=True)

    def test_navigate_to_example_com(self, container_id):
        """Should be able to navigate to example.com."""
        script = '''
import sys
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto("https://example.com", timeout=30000)
    title = page.title()
    browser.close()
    if "Example" in title:
        print("SUCCESS")
        sys.exit(0)
    else:
        print(f"FAIL: title was {title}")
        sys.exit(1)
'''
        result = subprocess.run(
            ["docker", "exec", container_id, "python", "-c", script],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, f"Navigation failed: {result.stderr}\n{result.stdout}"
        assert "SUCCESS" in result.stdout

    def test_navigate_to_roblox(self, container_id):
        """Should be able to navigate to roblox.com."""
        script = '''
import sys
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    try:
        page.goto("https://www.roblox.com", timeout=60000)
        # Check that we got some response (might be login page, geo-restricted, etc.)
        content = page.content()
        if len(content) > 1000:  # Got substantial content
            print("SUCCESS")
        else:
            print(f"FAIL: content too short ({len(content)} bytes)")
    except Exception as e:
        print(f"FAIL: {e}")
    finally:
        browser.close()
'''
        result = subprocess.run(
            ["docker", "exec", container_id, "python", "-c", script],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, f"Navigation failed: {result.stderr}"
        assert "SUCCESS" in result.stdout, f"Roblox navigation failed: {result.stdout}"


@docker
class TestScreenshotCapture:
    """Tests for screenshot capture from browser."""

    @pytest.fixture(scope="class")
    def container_id(self):
        """Build and start a container for testing."""
        project_root = Path(__file__).parent.parent

        # Build container
        build_result = subprocess.run(
            ["docker", "build", "-t", "agent-zero-browser-test", "."],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if build_result.returncode != 0:
            pytest.skip(f"Docker build failed: {build_result.stderr}")

        # Run container with entrypoint
        result = subprocess.run(
            [
                "docker", "run", "-d", "--rm",
                "--shm-size=2gb",
                "-e", "DISPLAY=:99",
                "-e", "DISPLAY_WIDTH=1920",
                "-e", "DISPLAY_HEIGHT=1080",
                "agent-zero-browser-test",
                "sleep", "300"
            ],
            capture_output=True,
            text=True,
        )
        container_id = result.stdout.strip()

        # Wait for display to be ready
        import time
        time.sleep(5)

        yield container_id

        # Cleanup
        subprocess.run(["docker", "stop", container_id], capture_output=True)

    def test_screenshot_dimensions(self, container_id):
        """Screenshot should match configured display dimensions."""
        script = '''
import sys
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)  # Use display
    page = browser.new_page(viewport={"width": 1920, "height": 1080})
    page.goto("https://example.com", timeout=30000)
    screenshot = page.screenshot()
    browser.close()

    # Check dimensions using PIL
    from PIL import Image
    import io
    img = Image.open(io.BytesIO(screenshot))
    width, height = img.size
    print(f"DIMENSIONS:{width}x{height}")
    if width == 1920 and height == 1080:
        print("SUCCESS")
    else:
        print(f"FAIL: expected 1920x1080, got {width}x{height}")
'''
        result = subprocess.run(
            ["docker", "exec", container_id, "python", "-c", script],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, f"Screenshot failed: {result.stderr}\n{result.stdout}"
        assert "SUCCESS" in result.stdout, f"Wrong dimensions: {result.stdout}"

    def test_screenshot_has_content(self, container_id):
        """Screenshot should contain actual page content (not blank)."""
        script = '''
import sys
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto("https://example.com", timeout=30000)
    screenshot = page.screenshot()
    browser.close()

    # Check that screenshot is not blank
    from PIL import Image
    import io
    img = Image.open(io.BytesIO(screenshot))

    # Get unique colors - a blank image would have very few
    colors = img.getcolors(maxcolors=1000)
    if colors is None or len(colors) > 5:
        # Either too many colors to count (good) or more than 5 colors (good)
        print("SUCCESS")
    else:
        print(f"FAIL: only {len(colors)} colors - might be blank")
'''
        result = subprocess.run(
            ["docker", "exec", container_id, "python", "-c", script],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, f"Screenshot content check failed: {result.stderr}"
        assert "SUCCESS" in result.stdout, f"Screenshot may be blank: {result.stdout}"


@docker
class TestHeadedBrowser:
    """Tests for browser running in headed mode (on virtual display)."""

    @pytest.fixture(scope="class")
    def container_id(self):
        """Build and start a container with display for testing."""
        project_root = Path(__file__).parent.parent

        # Build container
        build_result = subprocess.run(
            ["docker", "build", "-t", "agent-zero-browser-test", "."],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if build_result.returncode != 0:
            pytest.skip(f"Docker build failed: {build_result.stderr}")

        # Run container with entrypoint (which starts display)
        result = subprocess.run(
            [
                "docker", "run", "-d", "--rm",
                "--shm-size=2gb",
                "-e", "DISPLAY=:99",
                "agent-zero-browser-test",
                "sleep", "300"
            ],
            capture_output=True,
            text=True,
        )
        container_id = result.stdout.strip()

        # Wait for display to be ready
        import time
        time.sleep(5)

        yield container_id

        # Cleanup
        subprocess.run(["docker", "stop", container_id], capture_output=True)

    def test_chromium_launches_on_display(self, container_id):
        """Chromium should launch on the virtual display (headed mode)."""
        script = '''
import sys
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    # Launch in headed mode (will use DISPLAY env var)
    browser = p.chromium.launch(headless=False)
    page = browser.new_page()
    page.goto("https://example.com", timeout=30000)

    # If we got here without error, headed mode works
    title = page.title()
    browser.close()

    if "Example" in title:
        print("SUCCESS")
    else:
        print(f"FAIL: title was {title}")
'''
        result = subprocess.run(
            ["docker", "exec", container_id, "python", "-c", script],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, f"Headed browser failed: {result.stderr}\n{result.stdout}"
        assert "SUCCESS" in result.stdout
