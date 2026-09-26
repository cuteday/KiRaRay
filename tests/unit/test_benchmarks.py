from contextlib import closing
import json
import math
from pathlib import Path
import sqlite3
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))

from krr_bench.runner import summarize, validate_options
from krr_bench import analysis, profilers


class BenchmarkOptionsTest(unittest.TestCase):
    def test_valid_boundaries(self):
        validate_options(frames=1, warmup=0, repeats=1, seed=0)
        validate_options(frames=2**32 - 1, warmup=0, repeats=1, seed=2**64 - 1)
        validate_options(frames=1, warmup=2**32 - 2, repeats=3, seed=17)

    def test_invalid_arguments(self):
        defaults = {"frames": 4, "warmup": 2, "repeats": 3, "seed": 17}
        invalid = {
            "frames": (0, -1, 2**32, True, 1.5, "4", None),
            "warmup": (-1, 2**32 - 4, False, 1.5, "2", None),
            "repeats": (0, -1, True, 1.5, "3", None),
            "seed": (-1, 2**64, False, 1.5, "17", None),
        }
        for name, values in invalid.items():
            for value in values:
                options = dict(defaults, **{name: value})
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    validate_options(**options)


class BenchmarkStatisticsTest(unittest.TestCase):
    @staticmethod
    def run_result(render_ms, frames=4):
        return {"frames": frames, "timings": {"render_ms": render_ms}}

    def test_known_timings_and_units(self):
        result = summarize([self.run_result(value) for value in (8.0, 12.0, 16.0)])
        self.assertEqual(result["repeats"], 3)
        self.assertEqual(result["render_ms"],
                         {"min": 8.0, "median": 12.0, "mean": 12.0, "max": 16.0, "stdev": 4.0})
        self.assertEqual(result["ms_per_frame"],
                         {"min": 2.0, "median": 3.0, "mean": 3.0, "max": 4.0, "stdev": 1.0})
        fps = result["frames_per_second"]
        self.assertAlmostEqual(fps["min"], 250.0)
        self.assertAlmostEqual(fps["median"], 1000.0 / 3)
        self.assertAlmostEqual(fps["mean"], (500.0 + 1000.0 / 3 + 250.0) / 3)
        self.assertAlmostEqual(fps["max"], 500.0)

    def test_single_run(self):
        result = summarize([self.run_result(10.0, frames=2)])
        for name, expected in (("render_ms", 10.0), ("ms_per_frame", 5.0),
                               ("frames_per_second", 200.0)):
            for statistic in ("min", "median", "mean", "max"):
                self.assertEqual(result[name][statistic], expected)
            self.assertEqual(result[name]["stdev"], 0.0)

    def test_invalid_timings(self):
        with self.assertRaises(ValueError):
            summarize([])
        for value in (0.0, -1.0, math.nan, math.inf, -math.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                summarize([self.run_result(value)])


class ProfilerProcessTest(unittest.TestCase):
    @unittest.skipUnless(profilers.os.name == "nt", "Windows installation layout")
    def test_discovery_selects_native_windows_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "NVIDIA Corporation/Nsight Systems 2026.5.1"
            for arch in ("armv8", "x64"):
                target = root / ("target-windows-" + arch) / "nsys.exe"
                target.parent.mkdir(parents=True)
                target.touch()
            with patch.object(profilers.shutil, "which", return_value=None), \
                    patch.object(profilers.platform, "machine", return_value="AMD64"):
                selected = profilers.find_tool("nsys", env={"ProgramFiles": directory, "PATH": ""})
            self.assertEqual(selected, (root / "target-windows-x64/nsys.exe").resolve())

    def test_nsys_trace_selection_is_explicit(self):
        for trace in (None, "cuda,nvtx"):
            with self.subTest(trace=trace), tempfile.TemporaryDirectory() as directory:
                def complete(command, *args, **kwargs):
                    Path(directory, "profile.nsys-rep").write_bytes(b"report")
                    return {"stdout": "capture.stdout.log"}

                with patch.object(profilers, "find_tool", return_value=Path("nsys.exe")), \
                        patch.object(profilers, "tool_version", return_value="test version"), \
                        patch.object(profilers, "run_command", side_effect=complete):
                    result = profilers.capture("nsys", ["python", "worker.py"], directory,
                                               env={}, nsys_trace=trace)
                self.assertIn("--trace=" + (trace or "cuda,vulkan,nvtx"), result["command"])
                self.assertIn("--cpuctxsw=none", result["command"])

    def test_failed_command_retains_logs_and_status(self):
        with tempfile.TemporaryDirectory() as directory:
            command = [sys.executable, "-c",
                       "import sys; print('partial output'); print('failure detail', file=sys.stderr); sys.exit(7)"]
            with self.assertRaisesRegex(RuntimeError, "code 7"):
                profilers.run_command(command, directory, "failed", timeout=10)
            root = Path(directory)
            manifest = json.loads((root / "failed.command.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["returncode"], 7)
            self.assertEqual(manifest["command"], command)
            self.assertIn("partial output", (root / "failed.stdout.log").read_text())
            self.assertIn("failure detail", (root / "failed.stderr.log").read_text())

    def test_partial_report_does_not_hide_failed_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "profile.ncu-rep"

            def failed_capture(*args, **kwargs):
                report.write_bytes(b"partial report")
                raise RuntimeError("target failed")

            with patch.object(profilers, "find_tool", return_value=Path("ncu.exe")), \
                    patch.object(profilers, "tool_version", return_value="test version"), \
                    patch.object(profilers, "run_command", side_effect=failed_capture):
                with self.assertRaisesRegex(RuntimeError, "target failed"):
                    profilers.capture("ncu", [sys.executable, "worker.py"], directory, env={})
            self.assertTrue(report.is_file())
            self.assertFalse((Path(directory) / "capture.json").exists())

    def test_success_without_report_is_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(profilers, "find_tool", return_value=Path("ncu.exe")), \
                    patch.object(profilers, "tool_version", return_value="test version"), \
                    patch.object(profilers, "run_command", return_value={"stdout": "capture.log"}):
                with self.assertRaisesRegex(RuntimeError, "without a nonempty report"):
                    profilers.capture("ncu", [sys.executable, "worker.py"], directory, env={})


class NsysReportTest(unittest.TestCase):
    def test_overlapping_kernels_have_union_busy_time(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.sqlite"
            with closing(sqlite3.connect(path)) as connection:
                connection.executescript("""
                    CREATE TABLE StringIds (id INTEGER, value TEXT);
                    INSERT INTO StringIds VALUES (1, 'trace rays'), (2, 'shade');
                    CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL
                        (start INTEGER, end INTEGER, demangledName INTEGER, deviceId INTEGER);
                    INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES
                        (0, 10, 1, 0), (5, 15, 2, 0), (20, 30, 1, 0);
                """)
            result = analysis.analyze_nsys_sqlite(path)
            kernels = {kernel["name"]: kernel for kernel in result["kernels"]}
            self.assertEqual(kernels["trace rays"]["calls"], 2)
            self.assertEqual(kernels["trace rays"]["total_ns"], 20)
            self.assertEqual(kernels["trace rays"]["mean_ns"], 10)
            self.assertEqual(kernels["shade"]["total_ns"], 10)
            timeline = result["cuda_timeline"][0]
            self.assertEqual(timeline["span_ns"], 30)
            self.assertEqual(timeline["busy_ns"], 25)
            self.assertEqual(timeline["gap_ns"], 5)

    def test_missing_cuda_tables_are_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.sqlite"
            with closing(sqlite3.connect(path)) as connection:
                connection.execute("CREATE TABLE StringIds (id INTEGER, value TEXT)")
            result = analysis.analyze_nsys_sqlite(path)
            self.assertEqual(result["kernels"], [])
            self.assertTrue(result["notes"])


class NcuReportTest(unittest.TestCase):
    def test_preserves_per_launch_values_units_and_rules(self):
        def metric(value, unit):
            return SimpleNamespace(value=lambda: value, unit=lambda: unit)

        metrics = {"gpu__time_duration.sum": metric(12.5, "nsecond"),
                   "launch__registers_per_thread": metric(32, "register/thread"),
                   "large_integer": metric(2**54 + 1, "byte"),
                   "unavailable": metric(math.nan, "%")}
        rules = [{"name": "Example rule", "rule_message": {"message": "Example finding"}}]
        action = SimpleNamespace(NameBase_DEMANGLED=1,
                                 name=lambda basis=None: "shade<float>" if basis == 1 else "shade",
                                 metric_names=lambda: list(metrics),
                                 metric_by_name=metrics.__getitem__, rule_results_as_dicts=lambda: rules)
        current_range = SimpleNamespace(num_actions=lambda: 2, action_by_idx=lambda index: action)
        report = SimpleNamespace(num_ranges=lambda: 1, range_by_idx=lambda index: current_range)
        result = analysis.extract_ncu_report(report)
        self.assertEqual(len(result["actions"]), 2)
        self.assertEqual([item["action_index"] for item in result["actions"]], [0, 1])
        first = result["actions"][0]
        self.assertEqual(first["name"], "shade<float>")
        self.assertEqual(first["short_name"], "shade")
        self.assertEqual(first["metrics"]["gpu__time_duration.sum"], {"value": 12.5, "unit": "nsecond"})
        self.assertEqual(first["metrics"]["launch__registers_per_thread"]["value"], 32)
        self.assertEqual(first["metrics"]["large_integer"]["value"], 2**54 + 1)
        self.assertIsNone(first["metrics"]["unavailable"]["value"])
        self.assertIn("error", first["metrics"]["unavailable"])
        self.assertEqual(first["rules"], rules)
        json.dumps(result, allow_nan=False)

    def test_empty_capture_is_failure(self):
        with self.assertRaisesRegex(RuntimeError, "no profiled actions"):
            analysis.extract_ncu_report(SimpleNamespace(num_ranges=lambda: 0))


if __name__ == "__main__":
    unittest.main()
