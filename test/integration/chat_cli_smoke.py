#!/usr/bin/env python3

import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import json


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
    with tempfile.TemporaryDirectory(prefix="alpakaNN-chatcli-interactive-") as tmpdir:
        local_model = pathlib.Path(tmpdir) / "tiny_llama"
        shutil.copytree(model_dir, local_model)
        tokenizer_config_path = local_model / "tokenizer_config.json"
        tokenizer_config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
        tokenizer_config.pop("chat_template", None)
        tokenizer_config_path.write_text(json.dumps(tokenizer_config, indent=2) + "\n", encoding="utf-8")

        result = subprocess.run(
            [str(binary), "--interactive", "--model", str(local_model)],
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


def extract_assistant_lines(stdout: str) -> list[str]:
    replies: list[str] = []
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        match = re.match(r"^(?:>\s*)?Assistant:\s*(.+?)\s*$", line)
        if match:
            replies.append(match.group(1).strip())
    return replies


def is_readable_reply(text: str) -> bool:
    return bool(text.strip()) and any(char.isalpha() for char in text)


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
            [
                str(binary),
                "--interactive",
                "--model",
                str(model_dir),
                "--system-prompt",
                "You are a concise assistant.",
                "--max-new-tokens",
                "64",
            ],
            input="Say hello in one sentence.\nNow ask a short follow-up question.\nexit\n",
            text=True,
            capture_output=True,
            check=False,
        )
        if cli.returncode != 0:
            raise SystemExit(cli.stderr or cli.stdout)
        if "Interactive chat mode." not in cli.stdout:
            raise SystemExit(cli.stdout)
        replies = extract_assistant_lines(cli.stdout)
        if len(replies) != 2:
            raise SystemExit(cli.stdout)
        if not all(is_readable_reply(reply) for reply in replies):
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
