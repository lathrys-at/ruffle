"""The typing gate: the package and its tests pass mypy --strict.

The package advertises inline types (``py.typed``); this test keeps that promise
enforced from the test suite itself, so a hole in the annotations fails CI the same
way a behavioural regression does.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]


def test_mypy_strict_package_and_tests() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "--strict",
            "-p",
            "ruffle",
            "-p",
            "tests",
        ],
        cwd=PROJECT_DIR,
        env={**os.environ, "MYPYPATH": "python"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"mypy --strict failed:\n{result.stdout}{result.stderr}"
