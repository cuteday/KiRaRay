"""Verify Render Properties through Blender's native panel draw callbacks."""

import argparse
import json
from pathlib import Path
import sys
import time
import traceback

import bpy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--addon-dir", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    args.artifacts = args.artifacts.resolve()
    args.artifacts.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.addon_dir.resolve().parent))
    import kiraray
    from kiraray import engine

    expected = {
        "KIRARAY_PT_settings", "KIRARAY_PT_sampling", "KIRARAY_PT_light_paths",
        "KIRARAY_PT_light_paths_advanced", "KIRARAY_PT_system",
    }
    draws = {}
    results = {}
    phase = "initial"
    dismissed = False
    deadline = time.monotonic() + 20

    def track_draw(identifier, original):
        def draw(self, context):
            original(self, context)
            draws[identifier] = draws.get(identifier, 0) + 1
        return draw

    def fail():
        (args.artifacts / "failure.txt").write_text(traceback.format_exc())
        traceback.print_exc()
        bpy.ops.wm.quit_blender()

    try:
        for cls in engine._classes:
            if issubclass(cls, bpy.types.Panel):
                cls.draw = track_draw(cls.bl_idname, cls.draw)
                cls.bl_options = set(getattr(cls, "bl_options", ())) - {"DEFAULT_CLOSED"}
        kiraray.register()
        bpy.context.preferences.view.show_splash = False
        scene = bpy.context.scene
        scene.render.engine = "KIRARAY"
        area = max(bpy.context.screen.areas, key=lambda item: item.width * item.height)
        area.type = "PROPERTIES"
        area.spaces.active.context = "RENDER"
    except Exception:
        fail()
        return

    def tick():
        nonlocal phase, dismissed
        try:
            assert time.monotonic() < deadline, f"Panels did not draw in {phase}: {expected - draws.keys()}"
            if not dismissed:
                for window in bpy.context.window_manager.windows:
                    window.event_simulate(type="ESC", value="PRESS")
                    window.event_simulate(type="ESC", value="RELEASE")
                dismissed = True
                return 0.2
            if not expected.issubset(draws):
                area.tag_redraw()
                return 0.1
            results[phase] = dict(draws)
            (args.artifacts / "result.json").write_text(json.dumps(results, indent=2))
            if phase == "initial":
                scene.render.engine = "BLENDER_WORKBENCH"
                kiraray.unregister()
                draws.clear()
                kiraray.register()
                scene.render.engine = "KIRARAY"
                phase = "reregistered"
                area.tag_redraw()
                return 0.2
            bpy.ops.screen.screenshot(filepath=str(args.artifacts / "panels.png"))
            scene.render.engine = "BLENDER_WORKBENCH"
            kiraray.unregister()
            print("KRR_BLENDER_PANELS_SUCCESS", flush=True)
            bpy.ops.wm.quit_blender()
        except Exception:
            fail()
        return None

    bpy.app.timers.register(tick, first_interval=0.2)


if __name__ == "__main__":
    main()
