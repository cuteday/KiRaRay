"""Launch optional profiler tools without importing the renderer."""

import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import subprocess


def _version_key(path):
    return tuple(int(part) for part in re.findall(r"\d+", str(path)))


def _target_patterns(tool):
    if os.name == "nt":
        arch = "armv8" if platform.machine().lower() in ("arm64", "aarch64") else "x64"
        return (f"target/*{arch}/{tool}.exe", f"target-windows-{arch}/{tool}.exe")
    arch = "sbsa" if platform.machine().lower() in ("arm64", "aarch64") else "x64"
    return (f"target/*{arch}/{tool}", f"target-linux-{arch}/{tool}")


def _executable(path, tool):
    path = Path(path)
    if path.suffix.lower() in (".bat", ".cmd"):
        candidates = [path.with_suffix(".exe")]
        for pattern in _target_patterns(tool):
            candidates += sorted(path.parent.glob(pattern))
        return next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)
    return path.resolve() if path.is_file() else None


def find_tool(tool, tool_path=None, env=None):
    """Resolve a CLI executable, including NVIDIA's Windows batch wrappers."""
    if tool not in ("ncu", "nsys"):
        raise ValueError("tool must be ncu or nsys")
    env = os.environ if env is None else env
    selected = tool_path or env.get(f"KRR_{tool.upper()}")
    if selected:
        candidate = _executable(selected, tool)
        if candidate:
            return candidate
        raise RuntimeError(f"Invalid {tool} executable: {selected}")
    for name in (tool, tool + ".exe", tool + ".bat"):
        found = shutil.which(name, path=env.get("PATH", ""))
        if found:
            candidate = _executable(found, tool)
            if candidate:
                return candidate
    roots = []
    if os.name == "nt":
        program_files = Path(env.get("ProgramFiles", "C:/Program Files"))
        product = "Nsight Compute*" if tool == "ncu" else "Nsight Systems*"
        roots += list((program_files / "NVIDIA Corporation").glob(product))
        roots += list((program_files / "NVIDIA GPU Computing Toolkit/CUDA").glob("v*/nsight*"))
    else:
        product = "nsight-compute*" if tool == "ncu" else "nsight-systems*"
        roots += list(Path("/opt/nvidia").glob(product))
        roots += list(Path("/usr/local").glob("cuda*/" + product))
    for root in sorted(roots, key=_version_key, reverse=True):
        for pattern in (tool, tool + ".exe", tool + ".bat", f"bin/{tool}", *_target_patterns(tool)):
            for path in root.glob(pattern):
                candidate = _executable(path, tool)
                if candidate:
                    return candidate
    raise RuntimeError(f"{tool} was not found. Install it or set KRR_{tool.upper()} to its executable.")


def run_command(command, output_dir, name, *, env=None, timeout=900):
    """Keep commands and logs even when a profiler or its target fails."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [os.fspath(arg) for arg in command]
    stdout_path = output_dir / f"{name}.stdout.log"
    stderr_path = output_dir / f"{name}.stderr.log"
    manifest_path = output_dir / f"{name}.command.json"
    log_hint = f"see {output_dir} ({name}.stdout.log and {name}.stderr.log)"
    manifest = {"command": command, "timeout_seconds": timeout,
                "stdout": str(stdout_path), "stderr": str(stderr_path)}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    options = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {
        "start_new_session": True}
    try:
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            with subprocess.Popen(command, stdout=stdout, stderr=stderr, env=env, **options) as process:
                try:
                    code = process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    try:
                        if os.name == "nt":
                            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                           check=False, timeout=30)
                        else:
                            os.killpg(process.pid, signal.SIGKILL)
                    finally:
                        if process.poll() is None:
                            process.kill()
                    process.wait()
                    raise RuntimeError(f"Command timed out after {timeout}s; {log_hint}")
        manifest["returncode"] = code
        if code:
            raise RuntimeError(f"{Path(command[0]).name} exited with code {code}; {log_hint}")
    except Exception as error:
        manifest["error"] = str(error)
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def tool_version(tool, output_dir, *, env=None):
    result = run_command([str(tool), "--version"], output_dir, "version", env=env, timeout=30)
    return Path(result["stdout"]).read_text(encoding="utf-8", errors="replace").strip()


def capture(tool, command, output_dir, *, env, metrics=None, kernel_regex=None,
            timeout=900, tool_path=None, nsys_trace=None):
    """Profile one isolated worker, gated by its CUDA profiler start/stop calls."""
    if not command:
        raise ValueError("A worker command is required")
    if tool == "nsys" and (metrics is not None or kernel_regex is not None):
        raise ValueError("metrics and kernel_regex apply only to ncu")
    if nsys_trace is not None and tool != "nsys":
        raise ValueError("nsys_trace applies only to nsys")
    if nsys_trace not in (None, "cuda,vulkan,nvtx", "cuda,nvtx"):
        raise ValueError("nsys_trace must be cuda,vulkan,nvtx or cuda,nvtx")
    executable = find_tool(tool, tool_path, env)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report = output_dir / ("profile.ncu-rep" if tool == "ncu" else "profile.nsys-rep")
    if report.exists():
        raise FileExistsError(f"Refusing to overwrite {report}")
    version = tool_version(executable, output_dir, env=env)
    print(f"Profiler: {executable}\n{version}", flush=True)
    if tool == "ncu":
        args = [str(executable), "--profile-from-start", "off", "--target-processes", "all",
                "--replay-mode", "kernel", "--clock-control", "none", "--nvtx",
                "--kernel-name-base", "demangled", "--export", str(report)]
        if metrics is not None:
            selected = metrics.split(",") if isinstance(metrics, str) else list(metrics)
            if not selected or any(not isinstance(value, str) or not value.strip() for value in selected):
                raise ValueError("metrics must contain nonempty NCU metric names")
            args += ["--metrics", ",".join(value.strip() for value in selected)]
        else:
            args += ["--set", "basic"]
        if kernel_regex:
            args += ["--kernel-name", "regex:" + kernel_regex]
    else:
        args = [str(executable), "profile", "--trace=" + (nsys_trace or "cuda,vulkan,nvtx"),
                "--sample=none", "--cpuctxsw=none",
                "--capture-range=cudaProfilerApi", "--capture-range-end=stop", "--stats=false",
                "--output", str(report.with_suffix(""))]
    args += [os.fspath(arg) for arg in command]
    logs = run_command(args, output_dir, "capture", env=env, timeout=timeout)
    if not report.is_file() or report.stat().st_size == 0:
        raise RuntimeError(f"{tool} completed without a nonempty report; see {logs['stdout']}")
    result = {"tool": tool, "version": version, "report": str(report),
              "command": args, "logs": logs, "timings_are_profiled": True}
    (output_dir / "capture.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
