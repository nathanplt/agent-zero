"""Tests for Feature 1.1: Container Definition.

These tests verify that the Docker container is properly configured with:
- Virtual display (Xvfb on :99)
- VNC server for debugging
- Python 3.11+ environment

Note: These tests require Docker to be installed and running.
Some tests are marked with @pytest.mark.docker to indicate they require Docker.
"""

import subprocess
from pathlib import Path

import pytest


# Mark for tests requiring Docker
def _docker_available() -> bool:
    """Check if Docker is available."""
    try:
        result = subprocess.run(["docker", "info"], capture_output=True)
        return result.returncode == 0
    except (FileNotFoundError, OSError):
        return False

docker = pytest.mark.skipif(
    not _docker_available(),
    reason="Docker not available",
)


class TestDockerfileExists:
    """Tests for Dockerfile presence and structure."""

    def test_dockerfile_exists(self):
        """Dockerfile should exist in project root."""
        dockerfile = Path(__file__).parent.parent / "Dockerfile"
        assert dockerfile.exists(), "Dockerfile not found in project root"

    def test_docker_compose_exists(self):
        """docker-compose.yml should exist in project root."""
        compose_file = Path(__file__).parent.parent / "docker-compose.yml"
        assert compose_file.exists(), "docker-compose.yml not found in project root"

    def test_dockerfile_has_python(self):
        """Dockerfile should specify Python 3.11+."""
        dockerfile = Path(__file__).parent.parent / "Dockerfile"
        content = dockerfile.read_text()
        # Check for Python 3.11+ base image or installation
        assert any(
            version in content
            for version in ["python:3.11", "python:3.12", "python:3.13", "python3.11", "python3.12"]
        ), "Dockerfile should use Python 3.11+"

    def test_dockerfile_has_xvfb(self):
        """Dockerfile should install Xvfb for virtual display."""
        dockerfile = Path(__file__).parent.parent / "Dockerfile"
        content = dockerfile.read_text()
        assert "xvfb" in content.lower(), "Dockerfile should install xvfb"

    def test_dockerfile_has_vnc(self):
        """Dockerfile should install VNC server."""
        dockerfile = Path(__file__).parent.parent / "Dockerfile"
        content = dockerfile.read_text()
        # Check for x11vnc or tigervnc
        assert any(vnc in content.lower() for vnc in ["x11vnc", "tigervnc", "vnc"]), (
            "Dockerfile should install VNC server"
        )

    def test_dockerfile_exposes_vnc_port(self):
        """Dockerfile should expose VNC port 5900."""
        dockerfile = Path(__file__).parent.parent / "Dockerfile"
        content = dockerfile.read_text()
        assert "5900" in content, "Dockerfile should expose VNC port 5900"

    def test_dockerfile_sets_display(self):
        """Dockerfile should set DISPLAY environment variable."""
        dockerfile = Path(__file__).parent.parent / "Dockerfile"
        content = dockerfile.read_text()
        assert "DISPLAY" in content, "Dockerfile should set DISPLAY environment variable"
        assert ":99" in content or ":0" in content, "DISPLAY should be set to :99 or :0"


class TestDockerCompose:
    """Tests for docker-compose.yml configuration."""

    def test_docker_compose_valid_yaml(self):
        """docker-compose.yml should be valid YAML."""
        import yaml

        compose_file = Path(__file__).parent.parent / "docker-compose.yml"
        content = compose_file.read_text()
        # Should not raise
        config = yaml.safe_load(content)
        assert config is not None

    def test_docker_compose_has_services(self):
        """docker-compose.yml should define services."""
        import yaml

        compose_file = Path(__file__).parent.parent / "docker-compose.yml"
        config = yaml.safe_load(compose_file.read_text())
        assert "services" in config, "docker-compose.yml should have services"

    def test_docker_compose_has_agent_service(self):
        """docker-compose.yml should define agent service."""
        import yaml

        compose_file = Path(__file__).parent.parent / "docker-compose.yml"
        config = yaml.safe_load(compose_file.read_text())
        assert "agent" in config.get("services", {}), "Should have 'agent' service"

    def test_docker_compose_exposes_vnc_port(self):
        """docker-compose.yml should expose VNC port."""
        import yaml

        compose_file = Path(__file__).parent.parent / "docker-compose.yml"
        config = yaml.safe_load(compose_file.read_text())
        agent = config.get("services", {}).get("agent", {})
        ports = agent.get("ports", [])
        port_strs = [str(p) for p in ports]
        assert any("5900" in p for p in port_strs), "Should expose VNC port 5900"


@docker
class TestContainerBuild:
    """Tests for Docker container build (requires Docker)."""

    @pytest.fixture(scope="class")
    def build_result(self):
        """Build the container and return the result."""
        project_root = Path(__file__).parent.parent
        result = subprocess.run(
            ["docker", "build", "-t", "agent-zero-test", "."],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )
        return result

    def test_container_builds_successfully(self, build_result):
        """Container should build without errors."""
        assert build_result.returncode == 0, f"Docker build failed:\n{build_result.stderr}"


@docker
class TestContainerRuntime:
    """Tests for running container (requires Docker)."""

    @pytest.fixture(scope="class")
    def container_id(self):
        """Start a container and return its ID."""
        project_root = Path(__file__).parent.parent

        # Build first
        subprocess.run(
            ["docker", "build", "-t", "agent-zero-test", "."],
            cwd=project_root,
            capture_output=True,
            timeout=300,
        )

        # Run container in background
        result = subprocess.run(
            ["docker", "run", "-d", "--rm", "agent-zero-test", "sleep", "60"],
            capture_output=True,
            text=True,
        )
        container_id = result.stdout.strip()
        yield container_id

        # Cleanup
        subprocess.run(["docker", "stop", container_id], capture_output=True)

    def test_python_version(self, container_id):
        """Container should have Python 3.11+."""
        result = subprocess.run(
            ["docker", "exec", container_id, "python", "--version"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Python not found: {result.stderr}"
        version = result.stdout.strip()
        # Parse version number
        import re

        match = re.search(r"Python (\d+)\.(\d+)", version)
        assert match, f"Could not parse Python version: {version}"
        major, minor = int(match.group(1)), int(match.group(2))
        assert major >= 3 and minor >= 11, f"Python 3.11+ required, got {version}"

    def test_pip_available(self, container_id):
        """Container should have pip."""
        result = subprocess.run(
            ["docker", "exec", container_id, "pip", "--version"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"pip not found: {result.stderr}"

    def test_xvfb_installed(self, container_id):
        """Container should have Xvfb installed."""
        result = subprocess.run(
            ["docker", "exec", container_id, "which", "Xvfb"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, "Xvfb not installed"

    def test_vnc_installed(self, container_id):
        """Container should have VNC server installed."""
        result = subprocess.run(
            ["docker", "exec", container_id, "which", "x11vnc"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, "x11vnc not installed"


@docker
class TestVirtualDisplay:
    """Tests for virtual display functionality (requires Docker)."""

    @pytest.fixture(scope="class")
    def running_container(self):
        """Start container with entrypoint that starts Xvfb."""
        project_root = Path(__file__).parent.parent

        # Build first
        subprocess.run(
            ["docker", "build", "-t", "agent-zero-test", "."],
            cwd=project_root,
            capture_output=True,
            timeout=300,
        )

        # Run container with default entrypoint
        result = subprocess.run(
            ["docker", "run", "-d", "--rm", "-p", "5900:5900", "agent-zero-test"],
            capture_output=True,
            text=True,
        )
        container_id = result.stdout.strip()

        # Wait for services to start
        import time

        time.sleep(3)

        yield container_id

        # Cleanup
        subprocess.run(["docker", "stop", container_id], capture_output=True)

    def test_xvfb_running(self, running_container):
        """Xvfb should be running on :99."""
        result = subprocess.run(
            ["docker", "exec", running_container, "pgrep", "-f", "Xvfb"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, "Xvfb not running"

    def test_display_env_set(self, running_container):
        """DISPLAY environment should be set correctly."""
        result = subprocess.run(
            ["docker", "exec", running_container, "printenv", "DISPLAY"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, "DISPLAY not set"
        display = result.stdout.strip()
        assert display in [":99", ":0"], f"Unexpected DISPLAY value: {display}"

    def test_can_run_graphical_app(self, running_container):
        """Should be able to run a graphical application."""
        # Try to run xdpyinfo which requires a working display
        result = subprocess.run(
            ["docker", "exec", running_container, "xdpyinfo"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Cannot run graphical apps: {result.stderr}"

    def test_vnc_server_running(self, running_container):
        """VNC server should be listening on port 5900."""
        result = subprocess.run(
            ["docker", "exec", running_container, "pgrep", "-f", "x11vnc"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, "x11vnc not running"
