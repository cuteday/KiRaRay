"""Capture native output and contain a GPU test in its own process."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", required=True, type=Path)
    parser.add_argument("--timeout", type=float, default=240)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a test command is required after --")
    args.artifacts.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    timed_out = False
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                timeout=args.timeout, check=False)
        output = result.stdout
        code = result.returncode
    except subprocess.TimeoutExpired as error:
        output = (error.stdout or b"") + b"\nGPU test exceeded its timeout.\n"
        code = 124
        timed_out = True
    except OSError as error:
        output = str(error).encode("utf-8")
        code = 1
    log = output.decode("utf-8", errors="replace")
    (args.artifacts / "output.log").write_text(log, encoding="utf-8")
    (args.artifacts / "execution.json").write_text(json.dumps({
        "command": command, "exit_code": code, "timed_out": timed_out,
        "elapsed_seconds": time.perf_counter() - start,
    }, indent=2) + "\n", encoding="utf-8")
    print(log, end="")
    print(f"Artifacts: {args.artifacts.resolve()}")
    return code if 0 <= code <= 255 else 1


if __name__ == "__main__":
    sys.exit(main())
