"""Inspect/export an external blend file and optionally render it without saving it.

Use --background --factory-startup --disable-autoexec --python-exit-code 1.
GPU rendering requires the explicit --render flag and a built --addon-dir.
"""

import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import bpy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True, type=Path)
    parser.add_argument("--artifacts", required=True, type=Path)
    parser.add_argument("--addon-dir", type=Path, default=Path(__file__).resolve().parents[1] / "kiraray")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--resolution", type=int, nargs=2, default=(320, 180), metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--graphics-api", choices=("vulkan", "d3d12"), default="vulkan")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    if args.samples < 1 or min(args.resolution) < 1:
        parser.error("Samples and image dimensions must be positive")
    if args.render and args.preflight_only:
        parser.error("--render and --preflight-only cannot be combined")
    source = args.scene.resolve(strict=True)
    artifacts = args.artifacts.resolve()
    artifacts.mkdir(parents=True, exist_ok=True)
    original_stat = source.stat()
    sys.path.insert(0, str(args.addon_dir.resolve().parent))
    import kiraray
    from kiraray import engine, preflight

    bpy.ops.wm.open_mainfile(filepath=str(source), load_ui=False, use_scripts=False)
    scene = bpy.context.scene
    bpy.context.view_layer.update()
    reports = preflight.validate_depsgraph(bpy.context.evaluated_depsgraph_get())
    report = {
        "scene": str(source), "blender": bpy.app.version_string,
        "materials": [item.to_dict() for item in reports.values()],
        "supported": sum(item.supported for item in reports.values()),
        "unsupported": sum(not item.supported for item in reports.values()),
    }
    (artifacts / "preflight.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Material preflight: {report['supported']} supported, {report['unsupported']} rejected", flush=True)
    if args.preflight_only:
        return

    started = time.perf_counter()
    usd = artifacts / "scene.usdc"
    messages = []
    operator = SimpleNamespace(filepath=str(usd), report=lambda level, message: messages.append(message))
    result = engine.KiRaRayExportUSD.execute(operator, bpy.context)
    if result != {"FINISHED"}:
        raise RuntimeError("Validated USD export failed: " + "; ".join(messages))
    report["export_seconds"] = time.perf_counter() - started
    report["export_messages"] = messages
    report["usd"] = str(usd)
    camera = scene.camera
    if camera is not None:
        report["camera"] = {"name": camera.name, "type": camera.data.type,
                            "matrix_world": [list(row) for row in camera.matrix_world]}
    config = {
        "graphics_api": args.graphics_api, "resolution": args.resolution,
        "passes": [{"name": "WavefrontPathTracer", "params": {"nee": True, "rr": 0.8, "max_depth": 10}},
                   {"name": "AccumulatePass", "params": {"spp": 0, "mode": "accumulate"}}],
        "scene": {"model": [{"model": str(usd)}]},
    }
    (artifacts / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (artifacts / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if args.render:
        if report["unsupported"]:
            raise AssertionError("Resolve the material diagnostics in preflight.json before acceptance rendering")
        if camera is None or camera.data.type != "PERSP":
            raise RuntimeError("The acceptance scene requires its own perspective camera")
        kiraray.register()
        try:
            from render_scene import render, check_publication
            scene.render.engine = "KIRARAY"
            scene.render.resolution_x, scene.render.resolution_y = args.resolution
            scene.render.resolution_percentage = 100
            scene.kiraray.samples = args.samples
            scene.kiraray.graphics_api = args.graphics_api
            scene.kiraray.asset_root = str(source.parent)
            scene.view_layers[0].use_pass_z = True
            started = time.perf_counter()
            report["image"] = render(artifacts / "render.exr")
            report["render_seconds"] = time.perf_counter() - started
            report["status"] = dict(engine._last_status[scene.name_full])
            check_publication(report["status"], args.samples)
            if report["status"].get("diagnostics"):
                raise AssertionError("Native material conversion failed: " + repr(report["status"]["diagnostics"]))
            settings = scene.render.image_settings
            settings.media_type = "IMAGE"
            settings.file_format = "PNG"
            settings.color_mode = "RGB"
            settings.color_depth = "8"
            bpy.data.images["Render Result"].save_render(str(artifacts / "preview.png"), scene=scene)
        finally:
            (artifacts / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            kiraray.unregister()
    final_stat = source.stat()
    assert (original_stat.st_size, original_stat.st_mtime_ns) == (final_stat.st_size, final_stat.st_mtime_ns), \
        "The source blend file changed"
    print("KRR_BLENDER_MATERIAL_SCENE_SUCCESS", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main()
