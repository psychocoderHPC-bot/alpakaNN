#!/usr/bin/env python3

import json
import pathlib
import subprocess
import sys


def main() -> int:
    helper = pathlib.Path(sys.argv[1])
    model_dir = pathlib.Path(sys.argv[2])
    text = "Hello world"

    encoded = subprocess.run(
        ["python3", str(helper), "encode", str(model_dir), "--bos"],
        input=text,
        text=True,
        capture_output=True,
        check=True,
    )
    token_ids = json.loads(encoded.stdout)
    if not token_ids:
        raise SystemExit("encode returned no token ids")

    decoded = subprocess.run(
        ["python3", str(helper), "decode", str(model_dir)],
        input=json.dumps(token_ids),
        text=True,
        capture_output=True,
        check=True,
    )
    if decoded.stdout.strip() != text:
        raise SystemExit(f"round-trip mismatch: {decoded.stdout!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
