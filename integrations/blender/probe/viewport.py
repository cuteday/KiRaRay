"""Run in a windowed Blender host and exercise persistent viewport updates."""

import argparse
import json
from pathlib import Path
import sys
import time
import traceback

import bpy

sys.path.insert(0, str(Path(__file__).parent))
from render_scene import check_publication, cornell, render


def dismiss_popup():
    for window in bpy.context.window_manager.windows:
        window.event_simulate(type="ESC", value="PRESS")
        window.event_simulate(type="ESC", value="RELEASE")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--addon-dir", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--graphics-api", choices=("vulkan", "d3d12"), default="vulkan")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    args.artifacts = args.artifacts.resolve()
    args.artifacts.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.addon_dir.resolve().parent))
    import kiraray
    kiraray.register()
    bpy.context.preferences.view.show_splash = False
    bpy.context.preferences.view.render_display_type = "NONE"
    from bl_ui.space_statusbar import STATUSBAR_HT_header

    def draw_jobs(self, context):
        self.layout.separator_spacer()
        self.layout.template_running_jobs()

    STATUSBAR_HT_header.draw = draw_jobs
    box, white = cornell()
    scene = bpy.context.scene
    scene.render.engine = "KIRARAY"
    scene.kiraray.viewport_samples = 128
    scene.kiraray.samples = 2
    scene.kiraray.graphics_api = args.graphics_api
    scene.render.resolution_x = scene.render.resolution_y = 32
    scene.render.resolution_percentage = 100
    scene.view_layers[0].use_pass_z = True
    area = next(area for area in bpy.context.screen.areas if area.type == "VIEW_3D")
    area.spaces.active.region_3d.view_perspective = "CAMERA"
    area.spaces.active.shading.type = "RENDERED"
    results = {}
    deadline = time.monotonic() + 240
    phase = 0
    previous_version = 0
    cancellation_started = 0
    reloading = False
    setup_dismissed = False
    last_cancel_click = 0

    def tick():
        nonlocal phase, previous_version, scene, area, box, white, cancellation_started, reloading, setup_dismissed
        nonlocal last_cancel_click
        try:
            assert time.monotonic() < deadline, "Viewport test timed out"
            if not setup_dismissed:
                dismiss_popup()
                setup_dismissed = True
                return 0.2
            if reloading:
                windows = bpy.context.window_manager.windows
                if not windows:
                    return 0.1
                window = windows[0]
                scene = window.scene
                area = next(area for area in window.screen.areas if area.type == "VIEW_3D")
                box = bpy.data.objects["Box"]
                white = bpy.data.materials["White"]
                scene.render.engine = "KIRARAY"
                area.spaces.active.shading.type = "RENDERED"
                assert bpy.data.filepath == str(args.artifacts / "reload.blend")
                scene.camera.location.x += 0.015
                kiraray.engine._last_status.pop(scene.name_full, None)
                area.tag_redraw()
                reloading = False
            if phase == 9:
                elapsed = time.monotonic() - cancellation_started
                assert elapsed < 60, "Blender did not cancel the final render"
                if elapsed < 5:
                    return 0.1
                if bpy.app.is_job_running("RENDER"):
                    if not last_cancel_click:
                        bpy.ops.screen.screenshot(filepath=str(args.artifacts / "cancel.png"))
                        window = bpy.context.window_manager.windows[0]
                        for event, value in (("MOUSEMOVE", "NOTHING"), ("LEFTMOUSE", "PRESS"), ("LEFTMOUSE", "RELEASE")):
                            window.event_simulate(type=event, value=value, x=window.width - 16, y=12)
                        last_cancel_click = time.monotonic()
                    return 0.1
                results["cancellation_harness_ms"] = elapsed * 1000
                results["cancellation_response_ms"] = (time.monotonic() - last_cancel_click) * 1000
                scene.kiraray.samples = 2
                scene.camera.location.x += 0.02
                previous_version = 0
                kiraray.engine._last_status.pop(scene.name_full, None)
                phase = 10
                return 0.1
            status = kiraray.engine._last_status.get(scene.name_full, {})
            assert not status.get("error"), status
            if not status.get("frames") or status.get("version", 0) <= previous_version:
                area.tag_redraw()
                return 0.1
            previous_version = status["version"]
            check_publication(status, scene.kiraray.viewport_samples)
            if phase == 0:
                results["initial"] = dict(status)
                assert status["async_publication_count"] > 0, "Viewport did not publish an asynchronous intermediate image"
                white.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.1, 0.2, 0.8, 1)
            elif phase == 1:
                results["material"] = dict(status)
                assert status["scene_builds"] == results["initial"]["scene_builds"], "Material edit rebuilt geometry"
                assert status["material_updates"] > results["initial"]["material_updates"]
                box.location.x += 0.1
            elif phase == 2:
                results["transform"] = dict(status)
                assert status["scene_builds"] == results["initial"]["scene_builds"], "Transform edit rebuilt geometry"
                assert status["transform_updates"] > results["material"]["transform_updates"]
                box.data.vertices[0].co.z += 0.1
                box.data.update()
            elif phase == 3:
                results["geometry"] = dict(status)
                assert status["scene_builds"] > results["initial"]["scene_builds"], "Geometry edit did not rebuild"
                scene.camera.location.x += 0.15
            elif phase == 4:
                results["camera"] = dict(status)
                assert status["scene_builds"] == results["geometry"]["scene_builds"], "Camera edit rebuilt geometry"
                box.keyframe_insert(data_path="location", frame=1)
                box.location.z += 0.15
                box.keyframe_insert(data_path="location", frame=2)
                scene.frame_set(2)
            elif phase == 5:
                results["timeline"] = dict(status)
                assert status["transform_updates"] > results["camera"]["transform_updates"]
                results["final_image"] = render(args.artifacts / "final.exr")
                results["final_status"] = dict(kiraray.engine._last_status[scene.name_full])
                check_publication(results["final_status"], scene.kiraray.samples)
                kiraray.engine._last_status.pop(scene.name_full, None)
                scene.camera.location.x -= 0.15
            elif phase == 6:
                results["after_final"] = dict(status)
                ramp = white.node_tree.nodes.new("ShaderNodeValToRGB")
                white.node_tree.links.new(ramp.outputs["Color"], white.node_tree.nodes["Principled BSDF"].inputs["Base Color"])
            elif phase == 7:
                results["unsupported"] = dict(status)
                assert any("ColorRamp" in detail for detail in status["diagnostics"]), status
                bpy.ops.screen.screenshot(filepath=str(args.artifacts / "viewport.png"))
                white.node_tree.nodes.remove(white.node_tree.nodes["Color Ramp"])
                path = str(args.artifacts / "reload.blend")
                bpy.ops.wm.save_as_mainfile(filepath=path)
                kiraray.engine._last_status.pop(scene.name_full, None)
                reloading = True
                previous_version = 0
                phase = 8
                (args.artifacts / "progress.json").write_text(json.dumps(results, indent=2))

                def load_file():
                    bpy.ops.wm.open_mainfile(filepath=path)
                    return None

                bpy.app.timers.register(load_file, first_interval=0.1)
                return None
            elif phase == 8:
                results["reload"] = dict(status)
                assert not status["diagnostics"], status
                scene.kiraray.samples = 1048576
                with bpy.context.temp_override(window=bpy.context.window_manager.windows[0], area=area):
                    bpy.ops.render.render("INVOKE_DEFAULT")
                cancellation_started = time.monotonic()
            elif phase == 10:
                results["after_cancel"] = dict(status)
                (args.artifacts / "result.json").write_text(json.dumps(results, indent=2))
                print("KRR_BLENDER_VIEWPORT_SUCCESS", flush=True)
                bpy.ops.wm.quit_blender()
                return None
            phase += 1
            (args.artifacts / "progress.json").write_text(json.dumps(results, indent=2))
            return 0.1
        except Exception:
            (args.artifacts / "failure.txt").write_text(traceback.format_exc())
            traceback.print_exc()
            bpy.ops.wm.quit_blender()
            return None

    @bpy.app.handlers.persistent
    def resume_after_load(_):
        if reloading and not bpy.app.timers.is_registered(tick):
            bpy.app.timers.register(tick, first_interval=0.2)

    bpy.app.handlers.load_post.append(resume_after_load)
    bpy.app.timers.register(tick, first_interval=0.2)


if __name__ == "__main__":
    main()
