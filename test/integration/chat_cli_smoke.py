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


def run_tinyllama_chat(binary: pathlib.Path, repo_root: pathlib.Path) -> None:
    with tempfile.TemporaryDirectory(prefix="alpakaNN-tinyllama-") as tmpdir:
        model_dir = pathlib.Path(tmpdir) / "models" / "tinyllama-1.1b-chat"
        result = subprocess.run(
            [
                "python3",
                str(repo_root / "tools" / "download_tiny_llama.py"),
                "tinyllama-1.1b-chat",
                str(model_dir),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise SystemExit(result.stderr or result.stdout)

        cli = subprocess.run(
            [str(binary), "--interactive", "--model", str(model_dir)],
            input="Hello there\nexit\n",
            text=True,
            capture_output=True,
            check=False,
        )
        if cli.returncode != 0:
            raise SystemExit(cli.stderr or cli.stdout)
        lines = [line.strip() for line in cli.stdout.splitlines() if line.strip() and not line.startswith(">")]
        if not any("Interactive chat mode." in line for line in lines):
            raise SystemExit(cli.stdout)
        payload_lines = [line for line in lines if "Interactive chat mode." not in line and "Note:" not in line]
        if not payload_lines:
            raise SystemExit(cli.stdout)


def main() -> int:
    mode = sys.argv[1]
    binary = pathlib.Path(sys.argv[2])
    model_dir = pathlib.Path(sys.argv[3])
    if mode == "selftest":
        run_selftest(binary, model_dir)
    elif mode == "interactive":
        run_interactive(binary, model_dir)
    elif mode == "tinyllama-chat":
        run_tinyllama_chat(binary, pathlib.Path(sys.argv[4]))
    else:
        raise SystemExit(f"unknown mode: {mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
