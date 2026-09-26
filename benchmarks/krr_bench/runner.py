"""CPU-only orchestration; renderer imports belong to the worker process."""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASE = ROOT / "benchmarks/cases/cornell_wavefront.json"


def validate_options(frames, warmup, repeats, seed):
    for name, value, minimum in (("frames", frames, 1), ("warmup", warmup, 0),
                                  ("repeats", repeats, 1), ("seed", seed, 0)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if frames + warmup > 2**32 - 1:
        raise ValueError("frames + warmup must fit in an unsigned 32-bit frame index")
    if seed >= 2**64:
        raise ValueError("seed must fit in an unsigned 64-bit integer")


def summarize(runs):
    if not runs:
        raise ValueError("At least one completed run is required")
    durations, per_frame, rates = [], [], []
    for run in runs:
        duration = run["timings"]["render_ms"]
        frames = run["frames"]
        validate_options(frames, 0, 1, 0)
        if isinstance(duration, bool) or not math.isfinite(duration) or duration <= 0:
            raise ValueError("Measured render time must be finite and positive")
        durations.append(duration)
        per_frame.append(duration / frames)
        rates.append(1000 * frames / duration)

    def stats(values):
        return {"min": min(values), "median": statistics.median(values),
                "mean": statistics.mean(values), "max": max(values),
                "stdev": statistics.stdev(values) if len(values) > 1 else 0.0}

    return {"repeats": len(runs), "render_ms": stats(durations),
            "ms_per_frame": stats(per_frame), "frames_per_second": stats(rates)}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_cache(build_dir):
    cache = build_dir / "CMakeCache.txt"
    if not cache.is_file():
        raise ValueError(f"No CMake cache in {build_dir}")
    values = {}
    for line in cache.read_text(encoding="utf-8").splitlines():
        if line.startswith(("//", "#")) or "=" not in line or ":" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.split(":", 1)[0]] = value
    return values


def revision():
    def git(*args):
        result = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True,
                                text=True, timeout=10, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    try:
        status = git("status", "--porcelain")
        return {"commit": git("rev-parse", "HEAD"),
                "dirty": None if status is None else bool(status)}
    except (OSError, subprocess.TimeoutExpired):
        return {"commit": None, "dirty": None}


def worker_environment(build_dir, configuration):
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["KRR_BUILD_DIR"] = str(build_dir)
    env.pop("KRR_MODULE_DIR", None)
    for directory in (build_dir / "lib" / configuration, build_dir / "bin" / configuration):
        if directory.is_dir() and any(directory.glob("pykrr.*")):
            env["KRR_MODULE_DIR"] = str(directory)
            break
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT / "benchmarks"),
        str(ROOT / "common/scripts"), env.get("PYTHONPATH", "")]))
    return env


def run_worker(command, directory, env, timeout):
    from .profilers import run_command

    write_json(directory / "command.json", command)
    return run_command(command, directory, "worker", env=env, timeout=timeout)


def run(args):
    validate_options(args.frames, args.warmup, args.repeats, args.seed)
    if args.timeout <= 0 or not math.isfinite(args.timeout):
        raise ValueError("timeout must be finite and positive")
    if args.command == "profile" and args.repeats != 1:
        raise ValueError("Profile captures use one batch; use run for timing repetitions")
    if args.command == "profile" and args.tool == "nsight-python" and (args.metrics or args.kernel_regex):
        raise ValueError("The nsight-python adapter captures duration metrics; use ncu for custom metrics or kernel filters")
    if args.command == "profile" and args.nsys_trace is not None and args.tool != "nsys":
        raise ValueError("--nsys-trace applies only to nsys")
    build_dir = args.build_dir.resolve()
    cache = read_cache(build_dir)
    configuration = args.configuration or cache.get("CMAKE_BUILD_TYPE")
    if not configuration:
        raise ValueError("Select --configuration for a multi-configuration build")
    if not args.allow_debug and configuration.lower() not in ("release", "relwithdebinfo"):
        raise ValueError("Use Release/RelWithDebInfo for benchmarks, or --allow-debug for development checks")
    python = args.python or cache.get("Python_EXECUTABLE") or cache.get("_Python_EXECUTABLE")
    if not python or not Path(python).is_file():
        raise ValueError("The build's Python interpreter is unavailable; select --python explicitly")
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Config must contain a JSON object")
    if args.resolution:
        if min(args.resolution) <= 0:
            raise ValueError("Resolution must be positive")
        config["resolution"] = args.resolution
    asset_root = (args.asset_root or ROOT).resolve()
    if not asset_root.is_dir():
        raise ValueError("asset_root must be an existing directory")
    identifier = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    directory = (args.output_dir or build_dir / "benchmarks" / configuration / identifier).resolve()
    if directory.exists() and any(directory.iterdir()):
        raise ValueError(f"Output directory must be empty: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    request = {"schema_version": 1, "config": config, "config_path": str(config_path),
               "build_dir": str(build_dir), "configuration": configuration,
               "asset_root": str(asset_root), "frames": args.frames, "warmup": args.warmup,
               "repeats": args.repeats, "seed": args.seed, "validation": args.validation,
               "allow_debug": args.allow_debug, "revision": revision(),
               "backend": args.tool if args.command == "profile" else "timing",
               "tool_path": args.tool_path if args.command == "profile" else None,
               "nsys_trace": (args.nsys_trace or "cuda,vulkan,nvtx")
                             if args.command == "profile" and args.tool == "nsys" else None}
    write_json(directory / "request.json", request)
    write_json(directory / "config.json", config)
    command = [str(Path(python).resolve()), "-u", "-m", "krr_bench.worker", "--request",
               str(directory / "request.json"), "--output-dir", str(directory)]
    env = worker_environment(build_dir, configuration)
    try:
        capture = None
        if args.command == "profile" and args.tool != "nsight-python":
            from .profilers import capture as capture_profile
            capture = capture_profile(args.tool, command, directory, env=env,
                metrics=args.metrics.split(",") if args.metrics else None,
                kernel_regex=args.kernel_regex, timeout=args.timeout, tool_path=args.tool_path,
                nsys_trace=args.nsys_trace)
            write_json(directory / "capture.json", capture)
        else:
            run_worker(command, directory, env, args.timeout)
        result_path = directory / "result.json"
        if not result_path.is_file():
            raise RuntimeError("Worker produced no result.json; inspect the capture log")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "passed":
            raise RuntimeError("Worker did not report successful completion")
        if args.command == "profile":
            from .analysis import analyze
            if capture is None:
                capture = result["profile"]
                write_json(directory / "capture.json", capture)
            report = Path(capture["report"])
            analysis = analyze(report, directory, tool_path=args.tool_path)
            print(Path(analysis["summary"]).read_text(encoding="utf-8"))
            print(f"Profile and analysis saved to {directory}")
            return {"capture": capture, "analysis": analysis}
        summary = summarize(result["runs"])
        write_json(directory / "summary.json", summary)
        print(f"{summary['ms_per_frame']['median']:.3f} ms/frame median "
              f"({summary['repeats']} repetitions); results: {directory}")
        return summary
    except Exception as error:
        write_json(directory / "failure.json", {"status": "failed", "error": str(error)})
        raise


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    for name in ("run", "profile"):
        command = commands.add_parser(name)
        command.add_argument("--build-dir", type=Path, required=True)
        command.add_argument("--configuration")
        command.add_argument("--python", help="Override the build's Python interpreter")
        command.add_argument("--config", type=Path, default=DEFAULT_CASE)
        command.add_argument("--frames", type=int, default=64 if name == "run" else 1)
        command.add_argument("--warmup", type=int, default=8)
        command.add_argument("--repeats", type=int, default=3 if name == "run" else 1)
        command.add_argument("--seed", type=int, default=0)
        command.add_argument("--resolution", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"))
        command.add_argument("--asset-root", type=Path)
        command.add_argument("--output-dir", type=Path)
        command.add_argument("--allow-debug", action="store_true")
        command.add_argument("--validation", action="store_true")
        command.add_argument("--timeout", type=float, default=900)
        if name == "profile":
            command.add_argument("--tool", choices=("ncu", "nsys", "nsight-python"), required=True)
            command.add_argument("--tool-path")
            command.add_argument("--metrics", help="Comma-separated NCU metric names")
            command.add_argument("--kernel-regex", help="NCU kernel name filter")
            command.add_argument("--nsys-trace", choices=("cuda,vulkan,nvtx", "cuda,nvtx"),
                                 help="NSYS APIs to trace (default: cuda,vulkan,nvtx)")
    analyze = commands.add_parser("analyze", help="Analyze a saved report without rendering")
    analyze.add_argument("report", type=Path)
    analyze.add_argument("--output-dir", type=Path)
    analyze.add_argument("--tool-path")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "analyze":
            from .analysis import analyze
            report = args.report.resolve()
            directory = (args.output_dir or report.parent / (report.stem + "-analysis")).resolve()
            analysis = analyze(report, directory, tool_path=args.tool_path)
            print(Path(analysis["summary"]).read_text(encoding="utf-8"))
            print(f"Analysis saved to {directory}")
        else:
            run(args)
    except Exception as error:
        print(f"Benchmark failed: {error}", file=sys.stderr)
        return 1
    return 0
