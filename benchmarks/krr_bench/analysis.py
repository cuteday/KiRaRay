"""Offline, GPU-independent summaries of native profiler reports."""

from contextlib import closing
import html
import importlib
import json
import math
import os
from pathlib import Path
import sqlite3
import sys

from .profilers import find_tool, run_command


def _json_value(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return str(value)


def _load_ncu_report(tool_path=None):
    try:
        return importlib.import_module("ncu_report")
    except ImportError:
        pass
    selected = os.environ.get("KRR_NCU_REPORT_DIR")
    candidates = [Path(selected)] if selected else []
    if not selected:
        executable = find_tool("ncu", tool_path)
        candidates += [parent / "extras/python" for parent in executable.parents]
    errors = []
    for directory in candidates:
        if not (directory / "ncu_report.py").is_file():
            continue
        sys.path.insert(0, str(directory))
        try:
            return importlib.import_module("ncu_report")
        except ImportError as error:
            errors.append(f"{directory}: {error}")
        finally:
            sys.path.remove(str(directory))
    detail = "\n".join(errors)
    raise RuntimeError("Cannot import ncu_report. Use Nsight Compute's extras/python with a compatible "
                       "Python interpreter, or set KRR_NCU_REPORT_DIR to that directory. " + detail)


def extract_ncu_report(report):
    """Retain every launch and metric independently; unavailable values stay null."""
    actions = []
    notes = []
    for range_index in range(report.num_ranges()):
        current_range = report.range_by_idx(range_index)
        for action_index in range(current_range.num_actions()):
            action = current_range.action_by_idx(action_index)
            metrics = {}
            for name in action.metric_names():
                metric = action.metric_by_name(name)
                item = {"value": None, "unit": None}
                try:
                    item["unit"] = metric.unit()
                    value = metric.value()
                    item["value"] = _json_value(value)
                    if isinstance(value, float) and not math.isfinite(value):
                        item["error"] = "Non-finite metric value"
                except Exception as error:
                    item["error"] = str(error)
                metrics[name] = item
            name = action.name(action.NameBase_DEMANGLED) if hasattr(action, "NameBase_DEMANGLED") else action.name()
            record = {"range_index": range_index, "action_index": action_index,
                      "name": name, "short_name": action.name(), "metrics": metrics, "rules": [], "nvtx": []}
            if hasattr(action, "rule_results_as_dicts"):
                record["rules"] = _json_value(action.rule_results_as_dicts())
            else:
                notes.append("This ncu_report version does not expose recorded rule results.")
            state = action.nvtx_state() if hasattr(action, "nvtx_state") else None
            if state is not None:
                for domain_id in state.domains():
                    domain = state.domain_by_id(domain_id)
                    record["nvtx"].append({"domain": domain.name(),
                                           "push_pop": list(domain.push_pop_ranges()),
                                           "start_end": list(domain.start_end_ranges())})
            actions.append(record)
    if not actions:
        raise RuntimeError("The NCU report contains no profiled actions.")
    notes += ["Per-kernel profiler durations are diagnostic measurements, not benchmark scores.",
              "Metrics retain their original units; percentages and register counts are not summed."]
    return {"tool": "ncu", "actions": actions, "notes": list(dict.fromkeys(notes))}


def _quote(identifier):
    return '"' + identifier.replace('"', '""') + '"'


def _columns(connection, table):
    return {row[1] for row in connection.execute(f"PRAGMA table_info({_quote(table)})")}


def _event_summary(connection, table, strings, *, name_columns, notes):
    columns = _columns(connection, table)
    if not {"start", "end"}.issubset(columns):
        notes.append(f"Skipped {table}: no start/end columns.")
        return []
    name_column = next((name for name in name_columns if name in columns), None)
    identities = [name for name in ("deviceId", "contextId") if name in columns]
    grouping = ([name_column] if name_column else []) + identities
    fields = [_quote(name) for name in grouping]
    fields += ["COUNT(*) AS calls", "SUM(end-start) AS total_ns", "AVG(end-start) AS mean_ns",
               "MIN(end-start) AS min_ns", "MAX(end-start) AS max_ns"]
    if "bytes" in columns:
        fields.append('SUM("bytes") AS total_bytes')
    group_clause = " GROUP BY " + ", ".join(_quote(name) for name in grouping) if grouping else ""
    query = (f"SELECT {', '.join(fields)} FROM {_quote(table)} "
             f"WHERE end >= start{group_clause} ORDER BY total_ns DESC")
    records = []
    for row in connection.execute(query):
        result = dict(row)
        if result["calls"] == 0:
            continue
        identifier = result.pop(name_column, None) if name_column else None
        result["name"] = strings.get(identifier, str(identifier) if identifier is not None else table)
        for source, target in (("deviceId", "device_id"), ("contextId", "context_id")):
            if source in result:
                result[target] = result.pop(source)
        records.append(result)
    invalid = connection.execute(f"SELECT COUNT(*) FROM {_quote(table)} WHERE end < start").fetchone()[0]
    if invalid:
        notes.append(f"Ignored {invalid} events with negative durations in {table}.")
    return records


def _cuda_timeline(connection, tables):
    queries = []
    for table in tables:
        columns = _columns(connection, table)
        if {"start", "end"}.issubset(columns):
            device = '"deviceId"' if "deviceId" in columns else "NULL"
            queries.append(f"SELECT {device} AS device_id, start, end FROM {_quote(table)} WHERE end >= start")
    if not queries:
        return []
    query = " UNION ALL ".join(queries) + " ORDER BY device_id, start, end"
    result = []
    current = None
    merged_end = None
    for device, start, end in connection.execute(query):
        if current is None or device != current["device_id"]:
            if current is not None:
                result.append(current)
            current = {"device_id": device, "first_ns": start, "last_ns": end, "busy_ns": end - start}
            merged_end = end
        else:
            current["busy_ns"] += max(0, end - max(start, merged_end))
            current["last_ns"] = max(current["last_ns"], end)
            merged_end = max(merged_end, end)
    if current is not None:
        result.append(current)
    for item in result:
        item["span_ns"] = item["last_ns"] - item["first_ns"]
        item["gap_ns"] = item["span_ns"] - item["busy_ns"]
    return result


def analyze_nsys_sqlite(path):
    """Summarize CUDA events without assuming every Nsight Systems table exists."""
    path = Path(path).resolve()
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        notes = ["CUDA intervals exclude Vulkan and other processes: gaps are not total GPU idle time.",
                 "Summed kernel/API durations may overlap and are not end-to-end frame latency."]
        strings = {}
        if "StringIds" in tables and {"id", "value"}.issubset(_columns(connection, "StringIds")):
            strings = dict(connection.execute("SELECT id, value FROM StringIds"))
        kernel_table = next((name for name in ("CUPTI_ACTIVITY_KIND_KERNEL",
                                               "CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL") if name in tables), None)
        kernels = _event_summary(connection, kernel_table, strings,
                                 name_columns=("demangledName", "shortName", "name"), notes=notes) if kernel_table else []
        cuda_api = []
        for table in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"):
            if table in tables:
                events = _event_summary(connection, table, strings, name_columns=("nameId", "name"), notes=notes)
                for event in events:
                    event["source"] = table
                cuda_api.extend(events)
        transfers = []
        gpu_tables = [kernel_table] if kernel_table else []
        for kind in ("memcpy", "memset"):
            table = "CUPTI_ACTIVITY_KIND_" + kind.upper()
            if table in tables:
                gpu_tables.append(table)
                events = _event_summary(connection, table, {}, name_columns=("copyKind", "value"), notes=notes)
                for event in events:
                    event["kind"] = kind
                transfers.extend(events)
        if not kernels:
            notes.append("No CUDA kernel events were present in this export.")
        timeline = _cuda_timeline(connection, gpu_tables)
        return {"tool": "nsys", "sqlite": str(path), "tables": sorted(tables),
                "kernels": kernels, "cuda_api": sorted(cuda_api, key=lambda item: item["total_ns"], reverse=True),
                "transfers": transfers, "cuda_timeline": timeline, "notes": notes}


def _summary(result):
    lines = [f"# {result['tool'].upper()} report analysis", "", f"Report: {result['report']}", ""]
    if result["tool"] == "ncu":
        actions = result["actions"]
        lines += [f"Profiled actions: {len(actions)}", "", "| Kernel | Duration | Unit |", "| --- | ---: | --- |"]
        durations = []
        for action in actions:
            duration = action["metrics"].get("gpu__time_duration.sum", {})
            value = duration.get("value")
            durations.append((value if isinstance(value, (int, float)) else -1, action, duration))
        for _, action, duration in sorted(durations, key=lambda item: item[0], reverse=True)[:20]:
            name = html.escape(action["name"], quote=False).replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {name} | {duration.get('value', 'unavailable')} | {duration.get('unit', '')} |")
        messages = []
        for action in actions:
            for rule in action["rules"]:
                message = rule.get("rule_message", {})
                if message.get("message"):
                    messages.append(f"- {action['name']}: {message.get('title', rule.get('name', ''))}: {message['message']}")
        if messages:
            lines += ["", "## Recorded NCU findings", ""] + list(dict.fromkeys(messages))[:20]
    else:
        lines += ["| CUDA kernel | Calls | Total (ms) | Mean (ms) |", "| --- | ---: | ---: | ---: |"]
        for kernel in result["kernels"][:20]:
            name = html.escape(kernel["name"], quote=False).replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {name} | {kernel['calls']} | {kernel['total_ns'] / 1e6:.4f} | {kernel['mean_ns'] / 1e6:.4f} |")
    return "\n".join(lines + ["", "## Interpretation", ""] + [f"- {note}" for note in result["notes"]]) + "\n"


def analyze(report, output_dir, *, tool_path=None):
    """Save structured JSON and a readable summary without rerunning the renderer."""
    report = Path(report).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not report.is_file():
        raise FileNotFoundError(report)
    if report.suffix == ".ncu-rep":
        module = _load_ncu_report(tool_path)
        result = extract_ncu_report(module.load_report(str(report)))
        result["ncu_report_module"] = getattr(module, "__file__", None)
    elif report.suffix in (".nsys-rep", ".sqlite"):
        executable = None
        sqlite_path = report
        if report.suffix == ".nsys-rep":
            executable = find_tool("nsys", tool_path)
            sqlite_path = output_dir / "profile.sqlite"
            run_command([str(executable), "export", "--type=sqlite", "--force-overwrite=true",
                         "--output", str(sqlite_path), str(report)],
                        output_dir, "nsys-export")
            if not sqlite_path.is_file():
                raise RuntimeError("Nsight Systems did not produce the requested SQLite export.")
        result = analyze_nsys_sqlite(sqlite_path)
        if executable is not None:
            reports = []
            if result["kernels"]:
                reports.append("cuda_gpu_kern_sum")
            if result["cuda_api"]:
                reports.append("cuda_api_sum")
            if result["transfers"]:
                reports.append("cuda_gpu_mem_time_sum")
            if reports:
                result["stats"] = run_command([str(executable), "stats", "--report", ",".join(reports),
                                               "--format=csv", "--force-overwrite=true", "--output",
                                               str(output_dir / "stats"), str(sqlite_path)],
                                              output_dir, "nsys-stats")
    else:
        raise ValueError("Expected an .ncu-rep, .nsys-rep, or .sqlite report")
    result["report"] = str(report)
    result["analysis"] = str(output_dir / "analysis.json")
    result["summary"] = str(output_dir / "summary.md")
    Path(result["analysis"]).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    Path(result["summary"]).write_text(_summary(result), encoding="utf-8")
    return result
