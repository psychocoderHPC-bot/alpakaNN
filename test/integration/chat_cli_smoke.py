#!/usr/bin/env python3

import pathlib
import shutil
import subprocess
import sys
import tempfile


def run_selftest(binary: pathlib.Path, model_dir: pathlib.Path) -> None:
    with tempfile.TemporaryDirectory(prefix="alpakaNN-chatcli-") as tmpdir:
        cwd = pathlib.Path(tmpdir)
        models_dir = cwd / "models" / "tiny_llama"
        models_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(model_dir, models_dir)
        result = subprocess.run([str(binary)], cwd=cwd, text=True, capture_output=True, check=False)
        if result.returncode != 0:
            raise SystemExit(result.stderr or result.stdout)
        if "SELF-TEST PASSED" not in result.stdout:
            raise SystemExit(result.stdout)


def run_interactive(binary: pathlib.Path, model_dir: pathlib.Path) -> None:
    result = subprocess.run(
        [str(binary), "--interactive", "--model", str(model_dir)],
        input="Hello there\nexit\n",
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(result.stderr or result.stdout)
    if "Interactive chat mode." not in result.stdout:
        raise SystemExit(result.stdout)
    if "SELF-TEST PASSED" in result.stdout:
        raise SystemExit(result.stdout)


def main() -> int:
    mode = sys.argv[1]
    binary = pathlib.Path(sys.argv[2])
    model_dir = pathlib.Path(sys.argv[3])
    if mode == "selftest":
        run_selftest(binary, model_dir)
    elif mode == "interactive":
        run_interactive(binary, model_dir)
    else:
        raise SystemExit(f"unknown mode: {mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
