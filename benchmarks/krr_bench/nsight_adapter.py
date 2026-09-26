"""Optional Nsight Python adapter, loaded only in a fresh profiling worker."""

from importlib import metadata
import json
import os
from pathlib import Path
import re
import sys

from .profilers import find_tool, tool_version


def profile(config, *, frames, warmup, seed, output_dir, asset_root=None,
            validation=False, tool_path=None, allow_debug=False, configuration=None):
    """Capture warmed frames; keep raw launches alongside duration-only DataFrames."""
    if sys.version_info < (3, 10):
        raise RuntimeError("nsight-python requires Python 3.10+ and bindings built for that interpreter.")
    if "krr" in sys.modules or "pykrr" in sys.modules:
        raise RuntimeError("Run the nsight-python adapter in a fresh worker, before importing krr.")
    try:
        distribution = metadata.distribution("nsight-python")
        package_version = distribution.version
    except metadata.PackageNotFoundError as error:
        raise RuntimeError("Install benchmarks/requirements-nsight.txt in the selected build's Python environment.") from error
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if list(output_dir.glob("*.ncu-rep")):
        raise FileExistsError("Use a fresh output directory for nsight-python profiling.")
    executable = find_tool("ncu", tool_path)
    version = tool_version(executable, output_dir)
    settings = {"tool": "nsight-python", "nsight_python_version": package_version,
                "version": version, "metrics": ["gpu__time_duration.sum"],
                "replay_mode": "kernel", "clock_control": "none", "cache_control": "all",
                "frames": frames, "warmup": warmup, "seed": seed, "timings_are_profiled": True}
    provenance = distribution.read_text("direct_url.json")
    if provenance:
        settings["nsight_python_source"] = json.loads(provenance)
    (output_dir / "nsight-settings.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    match = re.search(r"Version\s+(\d+)\.(\d+)\.(\d+)", version, re.IGNORECASE)
    if match and tuple(int(part) for part in match.groups()) < (2026, 2, 1):
        raise RuntimeError("nsight-python requires Nsight Compute 2026.2.1 or later.")
    os.environ["PATH"] = str(executable.parent) + os.pathsep + os.environ.get("PATH", "")
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / "matplotlib"))

    import nsight
    from nsight.collection import ncu as collector

    if collector.injection_load_error is not None:
        raise RuntimeError(f"Nsight Python cannot initialize {executable}: {collector.injection_load_error}. "
                           "Use --tool ncu, or select a public Nsight Compute 2026.2.1+ build with "
                           "nvInjBeginProfiling/nvInjEndProfiling exports via --tool-path.") from collector.injection_load_error
    import krr

    build = krr.get_build_info()
    if build["build_type"].lower() not in ("release", "relwithdebinfo") and not allow_debug:
        raise RuntimeError("Selected native module is not an optimized build; use --allow-debug only for development")
    if configuration is not None and build["build_type"] != configuration:
        raise RuntimeError("Native module build type differs from the selected configuration")
    batches = []

    @nsight.analyze.kernel(runs=1, metrics=settings["metrics"], replay_mode="kernel",
                           clock_control="none", cache_control="all", thermal_mode="off",
                           combine_kernel_metrics=lambda left, right: left + right,
                           output_prefix=str(output_dir / "nsight-"), output_csv=True)
    def render_batch():
        active = []

        def begin():
            annotation = nsight.annotate("KiRaRay measured frames")
            annotation.__enter__()
            active.append(annotation)

        def end():
            if active:
                active.pop().__exit__(None, None, None)

        try:
            with krr.HeadlessRenderer(config, asset_root=asset_root, validation=validation) as renderer:
                batches.append(renderer.benchmark(frames=frames, warmup=warmup, seed=seed,
                               capture=False, on_capture_begin=begin, on_capture_end=end))
        finally:
            if active:
                active.pop().__exit__(*sys.exc_info())

    results = render_batch()
    reports = sorted(output_dir.glob("*.ncu-rep"))
    if results is None or not batches or len(reports) != 1 or reports[0].stat().st_size == 0:
        raise RuntimeError("nsight-python did not produce one complete batch and a nonempty NCU report.")
    dataframe = results.to_dataframe()
    if dataframe.empty:
        raise RuntimeError("nsight-python returned no profiling measurements.")
    dataframe_path = output_dir / "nsight-dataframe.json"
    dataframe.to_json(dataframe_path, orient="table", index=False, indent=2)
    settings.update({"report": str(reports[0]), "dataframe": str(dataframe_path),
                     "logs": [str(path) for path in sorted(output_dir.glob("*.log"))],
                     "note": "Only kernel duration is summed; raw launches remain in the NCU report."})
    batches[0]["profile"] = settings
    return batches[0]
