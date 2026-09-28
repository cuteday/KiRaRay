"""Exercise the native KiRaRay engine inside Blender (run with --background)."""

import argparse
import json
import math
from pathlib import Path
import sys

import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).parent))
from run import read_channels


def material(name, color, emission=0):
    result = bpy.data.materials.new(name)
    result.use_nodes = True
    shader = result.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = (*color, 1)
    shader.inputs["Roughness"].default_value = 0.65
    shader.inputs["Emission Color"].default_value = (*color, 1)
    shader.inputs["Emission Strength"].default_value = emission
    return result


def cornell():
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    white = material("White", (0.7, 0.7, 0.7))
    red = material("Red", (0.7, 0.05, 0.05))
    green = material("Green", (0.05, 0.6, 0.05))

    def quad(name, points, surface):
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata(points, [], [(0, 1, 2, 3)])
        mesh.materials.append(surface)
        obj = bpy.data.objects.new(name, mesh)
        bpy.context.collection.objects.link(obj)
        return obj

    quad("Floor", [(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)], white)
    quad("Ceiling", [(-1, 1, 2), (1, 1, 2), (1, -1, 2), (-1, -1, 2)], white)
    quad("Back", [(-1, 1, 0), (1, 1, 0), (1, 1, 2), (-1, 1, 2)], white)
    quad("Left", [(-1, -1, 0), (-1, 1, 0), (-1, 1, 2), (-1, -1, 2)], red)
    quad("Right", [(1, 1, 0), (1, -1, 0), (1, -1, 2), (1, 1, 2)], green)
    bpy.ops.mesh.primitive_cube_add(size=1, location=(-0.35, 0.2, 0.45))
    box = bpy.context.object
    box.name = "Box"
    box.scale = (0.6, 0.6, 0.9)
    box.rotation_euler[2] = 0.2
    box.data.materials.append(white)
    light = bpy.data.lights.new("Area", "AREA")
    light.energy = 80
    light.shape = "RECTANGLE"
    light.size = light.size_y = 0.6
    light_obj = bpy.data.objects.new("Area", light)
    light_obj.location = (0, 0, 1.95)
    bpy.context.collection.objects.link(light_obj)
    camera = bpy.data.cameras.new("Camera")
    camera.lens = 40
    camera_obj = bpy.data.objects.new("Camera", camera)
    camera_obj.location = (0, -4.2, 1.0)
    camera_obj.rotation_euler = (Vector((0, 0, 1)) - camera_obj.location).to_track_quat("-Z", "Y").to_euler()
    bpy.context.collection.objects.link(camera_obj)
    bpy.context.scene.camera = camera_obj
    bpy.context.scene.world = None
    return box, white


def render(path, allow_black=False, allow_nondepth=False):
    scene = bpy.context.scene
    bpy.ops.render.render()
    settings = scene.render.image_settings
    settings.media_type = "MULTI_LAYER_IMAGE"
    settings.file_format = "OPEN_EXR_MULTILAYER"
    settings.exr_codec = "NONE"
    settings.use_exr_interleave = True
    settings.color_mode = "RGBA"
    settings.color_depth = "32"
    bpy.data.images["Render Result"].save_render(str(path), scene=scene)
    channels = read_channels(path)
    color = [values for name, values in channels.items() if ".Combined." in name and not name.endswith(".A")]
    assert len(color) == 3
    pixels = [value for channel in color for value in channel]
    assert all(math.isfinite(value) for value in pixels), "Invalid color output"
    assert allow_black or sum(pixels) / len(pixels) > 0.001, "Black color output"
    result = {"mean": sum(pixels) / len(pixels), "maximum": max(pixels)}
    depth = next((values for name, values in channels.items() if name.endswith(".Depth.Z")), None)
    if depth is None:
        assert allow_nondepth, "Missing primary depth output"
        return result
    hits = [value for value in depth if math.isfinite(value) and 0 < value < 100]
    assert len(hits) > len(depth) // 4, "Primary depth has too few surface hits"
    result.update(depth_hits=len(hits), depth_min=min(hits), depth_max=max(hits))
    return result


def check_publication(status, samples, depth=True):
    assert status["frames"] == samples, "Publication omitted the final requested sample"
    assert 1 <= status["publication_count"] <= samples, "Invalid image publication count"
    assert status["publication_count"] <= status["readback_submission_count"] <= samples, "Duplicate image readback submission"
    assert 1 <= status["max_pending_readbacks"] <= 2, "Image readback storage is not bounded"
    assert 0 <= status["async_publication_count"] < status["publication_count"], "Invalid asynchronous publication count"
    assert 75 <= status["publication_interval_ms"] <= 500, "Publication cadence is outside its bounds"
    assert status["depth_requested"] is depth, "Depth work does not match bound AOVs"
    assert status["depth_capture_count"] == int(depth), "Primary depth was not cached for this generation"
    for name in ("readback_total_ms", "render_step_ms", "wait_ms", "update_ms"):
        assert math.isfinite(status[name]) and status[name] >= 0, f"Invalid {name} timing"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--addon-dir", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--graphics-api", choices=("vulkan", "d3d12"), default="vulkan")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    args.artifacts = args.artifacts.resolve()
    args.artifacts.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.addon_dir.resolve().parent))
    import kiraray
    kiraray.register()
    box, white = cornell()
    scene = bpy.context.scene
    scene.render.engine = "KIRARAY"
    scene.render.resolution_x = scene.render.resolution_y = 32
    scene.render.resolution_percentage = 100
    scene.kiraray.samples = args.samples
    scene.kiraray.graphics_api = args.graphics_api
    scene.view_layers[0].use_pass_z = True
    baseline = render(args.artifacts / "cornell.exr")
    status = dict(kiraray.engine._last_status[scene.name_full])
    check_publication(status, args.samples)
    assert not kiraray.engine._warnings.get(scene.name_full), kiraray.engine._warnings.get(scene.name_full)
    repeat = render(args.artifacts / "repeat.exr")
    assert baseline == repeat, (baseline, repeat)
    assert read_channels(args.artifacts / "cornell.exr") == read_channels(args.artifacts / "repeat.exr"), "Repeated F12 images differ"
    assert status["wavefront"]["max_depth"] == 10 and status["wavefront"]["nee"] is True
    assert abs(status["wavefront"]["rr"] - 0.8) < 1e-6
    scene.kiraray.max_depth, scene.kiraray.nee, scene.kiraray.rr = 0, False, 1.0
    direct = render(args.artifacts / "direct_only.exr", allow_black=True)
    direct_status = dict(kiraray.engine._last_status[scene.name_full])
    assert direct_status["wavefront"] == {"max_depth": 0, "nee": False, "rr": 1.0}
    assert direct["mean"] < baseline["mean"], "Zero path depth still includes scattered illumination"
    scene.kiraray.max_depth, scene.kiraray.nee, scene.kiraray.rr = 10, True, 0.8
    scene.kiraray.samples = 256
    publication = render(args.artifacts / "publication.exr")
    publication_status = dict(kiraray.engine._last_status[scene.name_full])
    check_publication(publication_status, scene.kiraray.samples)
    assert publication_status["publication_count"] < scene.kiraray.samples, "Every sample still triggered image readback"
    assert publication_status["async_publication_count"] > 0, "Intermediate images did not use asynchronous readback"
    scene.view_layers[0].use_pass_z = False
    color_only = render(args.artifacts / "color_only.exr", allow_nondepth=True)
    color_only_status = dict(kiraray.engine._last_status[scene.name_full])
    check_publication(color_only_status, scene.kiraray.samples, depth=False)
    assert color_only_status["async_publication_count"] > 0, "Color-only rendering did not publish asynchronously"
    combined = lambda path: {name: values for name, values in read_channels(path).items() if ".Combined." in name}
    assert combined(args.artifacts / "publication.exr") == combined(args.artifacts / "color_only.exr"), "Depth capture changed the rendered color"
    scene.kiraray.samples = args.samples
    scene.view_layers[0].use_pass_z = True
    scene.camera.location.x += 0.15
    camera = render(args.artifacts / "camera.exr")
    assert camera != baseline, "Camera edit did not change the output"
    scene.camera.location.x -= 0.15
    transparent = material("Cutout", (0.7, 0.7, 0.7))
    transparent.node_tree.nodes["Principled BSDF"].inputs["Alpha"].default_value = 0
    mesh = bpy.data.meshes.new("Cutout")
    mesh.from_pydata([(-2, -1.5, -1), (2, -1.5, -1), (2, -1.5, 3), (-2, -1.5, 3)], [], [(0, 1, 2, 3)])
    mesh.materials.append(transparent)
    cutout = bpy.data.objects.new("Cutout", mesh)
    bpy.context.collection.objects.link(cutout)
    clear = render(args.artifacts / "transparent.exr")
    original_depth = next(v for n, v in read_channels(args.artifacts / "cornell.exr").items() if n.endswith(".Depth.Z"))
    clear_depth = next(v for n, v in read_channels(args.artifacts / "transparent.exr").items() if n.endswith(".Depth.Z"))
    assert original_depth == clear_depth, "Transparent material incorrectly contributes to primary depth"
    transparent.node_tree.nodes["Principled BSDF"].inputs["Alpha"].default_value = 1
    opaque = render(args.artifacts / "opaque.exr", allow_black=True)
    assert opaque["depth_max"] < baseline["depth_min"], "Opaque foreground did not occlude primary depth"
    bpy.data.objects.remove(cutout, do_unlink=True)
    ramp = white.node_tree.nodes.new("ShaderNodeValToRGB")
    white.node_tree.links.new(ramp.outputs["Color"], white.node_tree.nodes["Principled BSDF"].inputs["Base Color"])
    unsupported = render(args.artifacts / "unsupported.exr")
    warnings = kiraray.engine._last_status[scene.name_full]["diagnostics"]
    assert any("ColorRamp" in warning for warning in warnings), "Producer diagnostics did not reach the native delegate"
    white.node_tree.nodes.remove(ramp)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.artifacts / "cornell.blend"))
    (args.artifacts / "result.json").write_text(json.dumps({"baseline": baseline, "repeat": repeat, "camera": camera,
                                                          "transparent": clear, "opaque": opaque,
                                                          "unsupported": unsupported, "warnings": warnings,
                                                          "publication": publication, "publication_status": publication_status,
                                                          "color_only": color_only, "color_only_status": color_only_status,
                                                          "direct_only": direct, "direct_status": direct_status,
                                                          "status": status}, indent=2))
    scene.render.engine = "BLENDER_EEVEE"
    kiraray.unregister()
    print("KRR_BLENDER_SCENE_SUCCESS", baseline, flush=True)


if __name__ == "__main__":
    main()
