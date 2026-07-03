"""Cross-implementation interoperability: states written by this binding reconcile
under the Rust crate's command-line tool, and the CLI's output loads back here.

Requires a Rust toolchain; skipped when cargo is unavailable.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from ruffle import (
    ChannelConfig,
    ChannelId,
    ChannelInput,
    Direction,
    Fuser,
    MergePolicy,
    RuffleState,
)

REPO_ROOT = Path(__file__).resolve().parents[3]

pytestmark = pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo is not available")


def make_state(offset: int) -> RuffleState:
    semantic = ChannelConfig(ChannelId("semantic", "v1"), Direction.HIGHER_IS_BETTER)
    fuser = Fuser([semantic])
    pool = [(f"doc{i:03}", 0.01 * (i + offset)) for i in range(30)]
    pool += [("hit0", 10.0), ("hit1", 10.5)]
    fuser.fuse([ChannelInput.scored(semantic, pool)])
    return fuser.state


def test_python_states_reconcile_under_the_rust_cli(tmp_path: Path) -> None:
    a, b = make_state(0), make_state(100)
    (tmp_path / "a.json").write_text(a.to_json(), encoding="utf-8")
    (tmp_path / "b.json").write_text(b.to_json(), encoding="utf-8")
    out = tmp_path / "merged.json"

    subprocess.run(
        [
            "cargo",
            "run",
            "--quiet",
            "--features",
            "cli",
            "--bin",
            "ruffle",
            "--",
            "reconcile",
            str(tmp_path / "a.json"),
            str(tmp_path / "b.json"),
            "-o",
            str(out),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    cli_merged = RuffleState.from_json(out.read_text(encoding="utf-8"))
    py_merged, _ = RuffleState.merge([a, b], MergePolicy.STRICT)
    # Byte-for-byte: both implementations run the same merge on the same code.
    assert cli_merged.to_json() == py_merged.to_json()
