"""Secure loading of runtime secrets from dotenv files."""

from __future__ import annotations

import os
import stat
from pathlib import Path

from dotenv import load_dotenv


def _project_root() -> Path:
    """Resolve project root from source tree layout."""
    return Path(__file__).resolve().parents[2]


def _resolve_env_file(
    *,
    env_file: str | Path | None,
    start_dir: Path,
) -> Path | None:
    """Resolve dotenv file path from explicit value or defaults."""
    env_path = env_file or os.environ.get("AGENTZERO_ENV_FILE")
    if env_path:
        resolved = Path(env_path).expanduser()
        if not resolved.is_absolute():
            resolved = start_dir / resolved
        return resolved.resolve()

    cwd_env = (start_dir / ".env").resolve()
    if cwd_env.exists():
        return cwd_env

    root_env = (_project_root() / ".env").resolve()
    if root_env.exists():
        return root_env

    return None


def _validate_permissions(env_file: Path) -> None:
    """Reject insecure dotenv file permissions on POSIX systems."""
    if os.name == "nt":
        return

    if env_file.is_symlink():
        raise PermissionError(
            f"Refusing to load dotenv symlink: {env_file}. Use a real file with chmod 600."
        )

    file_stat = env_file.stat()
    if hasattr(os, "getuid") and file_stat.st_uid != os.getuid():
        raise PermissionError(
            f"Refusing to load dotenv owned by another user: {env_file}. "
            "Use a file owned by the current user."
        )

    insecure_mask = stat.S_IRWXG | stat.S_IRWXO
    if file_stat.st_mode & insecure_mask:
        raise PermissionError(
            f"Insecure dotenv permissions for {env_file}. "
            "Restrict access with chmod 600."
        )


def load_environment_secrets(
    env_file: str | Path | None = None,
    *,
    override: bool = False,
    strict: bool = True,
    start_dir: Path | None = None,
) -> Path | None:
    """Load runtime secrets from dotenv while enforcing secure defaults.

    Args:
        env_file: Optional dotenv path. If omitted, checks `AGENTZERO_ENV_FILE`,
            then `.env` in current working directory, then project root.
        override: Whether dotenv values should override existing environment vars.
        strict: Whether an explicit but missing dotenv path should raise.
        start_dir: Optional base directory for resolving relative paths.

    Returns:
        Loaded dotenv path, or None when no dotenv file is found.
    """
    base_dir = (start_dir or Path.cwd()).resolve()
    resolved = _resolve_env_file(env_file=env_file, start_dir=base_dir)
    if resolved is None:
        return None

    if not resolved.exists():
        if strict:
            raise FileNotFoundError(f"Dotenv file not found: {resolved}")
        return None
    if resolved.is_dir():
        raise ValueError(
            f"Dotenv path is a directory: {resolved}. "
            "Remove or rename that directory and create a .env file (chmod 600)."
        )
    if not resolved.is_file():
        raise ValueError(f"Dotenv path is not a regular file: {resolved}")

    _validate_permissions(resolved)
    load_dotenv(dotenv_path=str(resolved), override=override)
    return resolved
