"""Compare a directly visible Blender emitter against Cycles' linear output."""

import argparse
import json
from pathlib import Path
import sys

import bpy

sys.path.insert(0, str(Path(__file__).parent))
from render_scene import material, render


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--addon-dir", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    args.artifacts = args.artifacts.resolve()
    args.artifacts.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.addon_dir.resolve().parent))
    import kiraray
    kiraray.register()
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    surface = material("Emitter", (1, 1, 1), emission=1)
    shader = surface.node_tree.nodes["Principled BSDF"]
    shader.inputs["Base Color"].default_value = (0, 0, 0, 1)
    shader.inputs["Specular IOR Level"].default_value = 0
    bpy.ops.mesh.primitive_plane_add(size=4)
    bpy.context.object.data.materials.append(surface)
    bpy.ops.object.camera_add(location=(0, 0, 3))
    scene = bpy.context.scene
    scene.camera = bpy.context.object
    scene.camera.data.lens = 40
    scene.world = None
    scene.render.resolution_x = scene.render.resolution_y = 32
    scene.render.resolution_percentage = 100
    scene.view_layers[0].use_pass_z = True
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 8
    scene.cycles.use_denoising = False
    cycles = render(args.artifacts / "cycles.exr")
    scene.render.engine = "KIRARAY"
    scene.kiraray.samples = 128
    native = render(args.artifacts / "kiraray.exr")
    ratio = native["mean"] / cycles["mean"]
    (args.artifacts / "result.json").write_text(json.dumps({"cycles": cycles, "kiraray": native, "ratio": ratio}, indent=2))
    assert abs(ratio - 1) < 0.03, f"Emission unit mismatch: KiRaRay/Cycles={ratio}"
    print("KRR_BLENDER_EMISSION_SUCCESS", ratio, flush=True)


if __name__ == "__main__":
    main()
