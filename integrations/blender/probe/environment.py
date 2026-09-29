"""Check Blender's bundled DWAB environment in F12 or Material Preview."""

import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys
import time
import traceback

import bpy

sys.path.insert(0, str(Path(__file__).parent))
from render_scene import material, render


def read_compression(path):
    data = path.read_bytes()
    assert struct.unpack_from("<I", data)[0] == 20000630, "Invalid EXR signature"
    offset = 8
    while data[offset]:
        end = data.index(0, offset)
        name = data[offset:end].decode()
        offset = data.index(0, end + 1) + 1
        size, = struct.unpack_from("<I", data, offset)
        offset += 4
        if name == "compression":
            assert size == 1
            return data[offset]
        offset += size
    raise AssertionError("Missing EXR compression attribute")


def setup_scene(texture):
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    bpy.ops.mesh.primitive_plane_add(size=20)
    bpy.context.object.data.materials.append(material("Diffuse", (0.7, 0.7, 0.7)))
    bpy.ops.object.camera_add(location=(0, 0, 3))
    scene = bpy.context.scene
    scene.camera = bpy.context.object
    scene.render.engine = "KIRARAY"
    scene.render.resolution_x = scene.render.resolution_y = 32
    scene.render.resolution_percentage = 100
    scene.view_layers[0].use_pass_z = True
    world = bpy.data.worlds.new("Forest")
    world.use_nodes = True
    scene.world = world
    environment = world.node_tree.nodes.new("ShaderNodeTexEnvironment")
    environment.image = bpy.data.images.load(str(texture), check_existing=False)
    background = world.node_tree.nodes.get("Background")
    world.node_tree.links.new(environment.outputs["Color"], background.inputs["Color"])
    background.inputs["Strength"].default_value = 1
    return scene


def run_viewport(args, kiraray, scene, report):
    assert not bpy.app.background, "--viewport requires a windowed Blender process"
    bpy.context.preferences.view.show_splash = False
    scene.kiraray.viewport_samples = args.samples
    # The scene world is disabled so Blender supplies its studio environment.
    scene.world = None
    area = next(area for area in bpy.context.screen.areas if area.type == "VIEW_3D")
    area.spaces.active.region_3d.view_perspective = "CAMERA"
    shading = area.spaces.active.shading
    shading.type = "MATERIAL"
    shading.use_scene_world = False
    shading.use_scene_lights = False
    shading.studiolight_intensity = 1
    shading.studio_light = "forest.exr"
    deadline = time.monotonic() + 180
    dismissed = False

    def tick():
        nonlocal dismissed
        try:
            if not dismissed:
                for window in bpy.context.window_manager.windows:
                    window.event_simulate(type="ESC", value="PRESS")
                    window.event_simulate(type="ESC", value="RELEASE")
                dismissed = True
            assert time.monotonic() < deadline, "Material Preview did not converge"
            status = kiraray.engine._last_status.get(scene.name_full, {})
            assert not status.get("error"), status
            if status.get("frames", 0) < args.samples:
                area.tag_redraw()
                return 0.2
            assert status.get("lights"), "Studio environment was not passed to Hydra"
            assert not status.get("diagnostics"), status
            report["viewport"] = dict(status)
            report["studio_light"] = shading.studio_light
            bpy.ops.screen.screenshot(filepath=str(args.artifacts / "material-preview.png"))
            (args.artifacts / "result.json").write_text(json.dumps(report, indent=2) + "\n")
            print("KRR_BLENDER_ENVIRONMENT_VIEWPORT_SUCCESS", flush=True)
            bpy.ops.wm.quit_blender()
        except Exception:
            (args.artifacts / "failure.txt").write_text(traceback.format_exc())
            traceback.print_exc()
            bpy.ops.wm.quit_blender()
        return None

    bpy.app.timers.register(tick, first_interval=0.2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--addon-dir", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--graphics-api", choices=("vulkan", "d3d12"), default="vulkan")
    parser.add_argument("--viewport", action="store_true")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    args.artifacts = args.artifacts.resolve()
    args.artifacts.mkdir(parents=True, exist_ok=True)
    for name in ("result.json", "failure.txt"):
        (args.artifacts / name).unlink(missing_ok=True)
    texture = Path(bpy.utils.system_resource("DATAFILES")) / "studiolights/world/forest.exr"
    assert texture.is_file(), f"Blender's bundled environment is missing: {texture}"
    compression = read_compression(texture)
    assert compression == 9, f"Expected Blender 5.2's DWAB fixture, got compression {compression}"
    report = {"blender": bpy.app.version_string, "source": str(texture), "compression": "DWAB",
              "sha256": hashlib.sha256(texture.read_bytes()).hexdigest()}
    sys.path.insert(0, str(args.addon_dir.resolve().parent))
    import kiraray
    kiraray.register()
    scene = setup_scene(texture)
    scene.kiraray.samples = args.samples
    scene.kiraray.graphics_api = args.graphics_api
    if args.viewport:
        run_viewport(args, kiraray, scene, report)
        return
    try:
        report["illuminated"] = render(args.artifacts / "forest.exr")
        report["status"] = dict(kiraray.engine._last_status[scene.name_full])
        assert not report["status"].get("error"), report["status"]
        assert not kiraray.engine._warnings.get(scene.name_full), kiraray.engine._warnings.get(scene.name_full)
        scene.world = None
        report["dark"] = render(args.artifacts / "dark.exr", allow_black=True)
        report["dark_status"] = dict(kiraray.engine._last_status[scene.name_full])
        assert not report["dark_status"].get("lights"), "Removed world is still present in Hydra"
        assert report["dark"]["maximum"] < 1e-7, "Scene without lights is not black"
        assert report["illuminated"]["mean"] > report["dark"]["mean"] + 0.001
        (args.artifacts / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        print("KRR_BLENDER_ENVIRONMENT_SUCCESS", report["illuminated"], flush=True)
    finally:
        scene.render.engine = "BLENDER_EEVEE"
        kiraray.unregister()


if __name__ == "__main__":
    main()
