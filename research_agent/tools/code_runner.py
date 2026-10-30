"""
CodeRunnerTool — safe sandboxed code execution tool.

Executes code in a subprocess with:
  - Hard wall-clock timeout (SIGKILL after timeout seconds)
  - Stdout / stderr capture
  - Environment scrubbing (no inherited API keys passed to child process)
  - Optional network-access warning (subprocess still has host network;
    true kernel-level network isolation requires a container/namespace)

Supported languages: python, bash/sh, javascript (node)

Environment variables:
    CODE_RUNNER_ENABLED : "true" | "false"  (default: true)
                          Set to "false" to disable all execution and return a
                          disabled message instead.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import sys
import time
from typing import Optional

from research_agent.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_TIMEOUT = 10  # seconds
_MAX_OUTPUT_BYTES = 128 * 1024  # 128 KB per stream

# Allowed language identifiers → (executable, file extension, arg template)
_LANGUAGES: dict[str, dict] = {
    "python": {"exe": sys.executable, "ext": ".py", "args": ["{file}"]},
    "python3": {"exe": sys.executable, "ext": ".py", "args": ["{file}"]},
    "bash": {"exe": "bash", "ext": ".sh", "args": ["{file}"]},
    "sh": {"exe": "sh", "ext": ".sh", "args": ["{file}"]},
    "javascript": {"exe": "node", "ext": ".js", "args": ["{file}"]},
    "js": {"exe": "node", "ext": ".js", "args": ["{file}"]},
}

# Minimum safe environment for child process — no secrets
_SAFE_ENV_KEYS = {
    "PATH", "HOME", "TMPDIR", "TEMP", "TMP",
    "LANG", "LC_ALL", "LC_CTYPE", "USER", "LOGNAME",
    "PYTHONDONTWRITEBYTECODE", "PYTHONIOENCODING",
}


def _build_safe_env() -> dict[str, str]:
    """Return a minimal, scrubbed environment for the subprocess."""
    safe: dict[str, str] = {}
    host_env = os.environ
    for key in _SAFE_ENV_KEYS:
        if key in host_env:
            safe[key] = host_env[key]
    # Ensure Python output is unbuffered
    safe["PYTHONUNBUFFERED"] = "1"
    safe["PYTHONIOENCODING"] = "utf-8"
    return safe


# ---------------------------------------------------------------------------
# Main tool
# ---------------------------------------------------------------------------


class CodeRunnerTool(BaseTool):
    """
    Executes code snippets in a sandboxed subprocess.

    Usage::

        tool = CodeRunnerTool()
        result = await tool.run("print('hello')", language="python", timeout=5)
        print(result.data["stdout"])   # "hello\\n"

    When CODE_RUNNER_ENABLED=false the tool returns a disabled result
    without raising an error, so callers can handle it gracefully.
    """

    name = "code_runner"
    description = (
        "Execute a code snippet and return its output. "
        "Supports Python, Bash, and JavaScript (Node). "
        "Runs in an isolated subprocess with a hard timeout."
    )

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "Source code to execute"},
                        "language": {"type": "string", "description": "Language: python | bash | javascript", "default": "python"},
                        "timeout": {"type": "integer", "description": "Execution timeout in seconds (default 10)", "default": 10},
                    },
                    "required": ["code"],
                },
            },
        }

    def __init__(self, enabled: Optional[bool] = None) -> None:
        if enabled is not None:
            self._enabled = enabled
        else:
            raw = os.getenv("CODE_RUNNER_ENABLED", "false").strip().lower()
            self._enabled = raw in ("true", "1", "yes", "on")
        if not self._enabled:
            logger.info(
                "CodeRunnerTool: Execution is disabled by default for safety. "
                "Set CODE_RUNNER_ENABLED=true or initialize with enabled=True to opt in."
            )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def run(  # type: ignore[override]
        self,
        code: str,
        language: str = "python",
        timeout: int = _DEFAULT_TIMEOUT,
    ) -> ToolResult:
        """
        Execute code and return the result.

        Args:
            code:     Source code to execute.
            language: One of "python", "bash", "javascript".
            timeout:  Hard timeout in seconds (max 60).

        Returns:
            ToolResult where .data is::

                {
                    "stdout":      str,
                    "stderr":      str,
                    "exit_code":   int,
                    "duration_ms": float,
                    "language":    str,
                    "timed_out":   bool,
                }
        """
        # ------------------------------------------------------------------
        # Guard: disabled
        # ------------------------------------------------------------------
        if not self._enabled:
            return self._err(
                "Code execution is disabled. Set CODE_RUNNER_ENABLED=true to enable it.",
                disabled=True,
            )

        # ------------------------------------------------------------------
        # Guard: input validation
        # ------------------------------------------------------------------
        if not code or not code.strip():
            return self._err("Code must not be empty.")

        lang_key = language.strip().lower()
        if lang_key not in _LANGUAGES:
            return self._err(
                f"Unsupported language '{language}'. "
                f"Supported: {', '.join(_LANGUAGES)}"
            )

        lang_cfg = _LANGUAGES[lang_key]
        exe: str = lang_cfg["exe"]
        ext: str = lang_cfg["ext"]

        # Check executable is available
        if exe != sys.executable and not shutil.which(exe):
            return self._err(
                f"Interpreter '{exe}' not found on PATH. "
                f"Install it to run {language} code."
            )

        # Cap timeout
        timeout = max(1, min(timeout, 60))

        # ------------------------------------------------------------------
        # Write code to a temp file and execute
        # ------------------------------------------------------------------
        import tempfile
        import pathlib

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=ext,
            prefix="research_agent_run_",
            delete=False,
            encoding="utf-8",
        ) as tmp:
            tmp.write(code)
            tmp_path = tmp.name

        cmd = [exe] + [
            arg.format(file=tmp_path) for arg in lang_cfg["args"]
        ]

        start = time.perf_counter()
        timed_out = False
        stdout_data = b""
        stderr_data = b""
        exit_code = -1

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=_build_safe_env(),
                # Run inside a new process group so we can kill the whole tree
                start_new_session=True,
            )

            try:
                stdout_data, stderr_data = await asyncio.wait_for(
                    process.communicate(),
                    timeout=float(timeout),
                )
                exit_code = process.returncode or 0
            except asyncio.TimeoutError:
                timed_out = True
                # Kill entire process group
                try:
                    import signal
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except ProcessLookupError:
                    process.kill()
                await process.wait()
                exit_code = -1

        except Exception as exc:  # noqa: BLE001
            duration_ms = (time.perf_counter() - start) * 1000
            logger.exception("CodeRunnerTool subprocess error")
            return self._err(
                f"Failed to execute code: {exc}",
                duration_ms=duration_ms,
            )
        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        duration_ms = (time.perf_counter() - start) * 1000

        # Decode and cap outputs
        def _decode(b: bytes) -> str:
            return b[:_MAX_OUTPUT_BYTES].decode("utf-8", errors="replace")

        stdout_str = _decode(stdout_data)
        stderr_str = _decode(stderr_data)

        if timed_out:
            return self._err(
                f"Execution timed out — process killed after {timeout}s timeout",
                timed_out=True,
                stdout=stdout_str,
                stderr=stderr_str,
                duration_ms=duration_ms,
            )

        logger.info(
            "CodeRunnerTool: ran %s code in %.1fms, exit=%d, timed_out=%s",
            language, duration_ms, exit_code, timed_out,
        )

        return self._ok(
            {
                "stdout": stdout_str,
                "stderr": stderr_str,
                "exit_code": exit_code,
                "duration_ms": round(duration_ms, 2),
                "language": language,
                "timed_out": timed_out,
                "disabled": False,
            },
            duration_ms=duration_ms,
        )
