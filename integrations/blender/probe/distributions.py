"""Check runtime Glossy/Glass distribution edits inside Blender, using F12 renders."""

import argparse
import json
from pathlib import Path
import sys

import bpy

sys.path.insert(0, str(Path(__file__).parent))
from render_scene import cornell, render, check_publication
from run import read_channels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--addon-dir", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--graphics-api", choices=("vulkan", "d3d12"), default="vulkan")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    artifacts = args.artifacts.resolve()
    artifacts.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.addon_dir.resolve().parent))
    import kiraray
    kiraray.register()
    results = {}
    scene = bpy.context.scene
    try:
        box, _ = cornell()
        bpy.data.objects.remove(box, do_unlink=True)
        bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=0.45, location=(-0.3, 0.2, 0.5))
        sphere = bpy.context.object
        for face in sphere.data.polygons:
            face.use_smooth = True
        material = bpy.data.materials.new("Runtime distributions")
        sphere.data.materials.append(material)
        tree = material.node_tree
        tree.nodes.clear()
        output = tree.nodes.new("ShaderNodeOutputMaterial")
        scene.render.engine = "KIRARAY"
        scene.render.resolution_x = scene.render.resolution_y = 96
        scene.render.resolution_percentage = 100
        scene.kiraray.samples = args.samples
        scene.kiraray.graphics_api = args.graphics_api
        scene.view_layers[0].use_pass_z = True
        for kind in ("Glossy", "Glass", "Mixed"):
            shader = tree.nodes.new("ShaderNodeBsdf" + ("Glossy" if kind == "Mixed" else kind))
            shader.inputs["Roughness"].default_value = 0.3
            temporary = [shader]
            surface = shader
            if kind == "Mixed":
                second = tree.nodes.new("ShaderNodeBsdfGlossy")
                second.distribution = "GGX"
                second.inputs["Color"].default_value = (0.8, 0.2, 0.1, 1)
                second.inputs["Roughness"].default_value = 0.15
                surface = tree.nodes.new("ShaderNodeMixShader")
                surface.inputs[0].default_value = 0.35
                tree.links.new(shader.outputs[0], surface.inputs[1])
                tree.links.new(second.outputs[0], surface.inputs[2])
                temporary.extend((second, surface))
            tree.links.new(surface.outputs[0], output.inputs["Surface"])
            images = []
            for index, distribution in enumerate(("GGX", "BECKMANN", "GGX")):
                shader.distribution = distribution
                bpy.context.view_layer.update()
                name = f"{kind.lower()}-{index}-{distribution.lower()}"
                path = artifacts / (name + ".exr")
                image = render(path)
                status = dict(kiraray.engine._last_status[scene.name_full])
                check_publication(status, args.samples)
                assert not status["diagnostics"], status["diagnostics"]
                channels = read_channels(path)
                images.append([value for name, values in sorted(channels.items())
                               if ".Combined." in name and not name.endswith(".A") for value in values])
                results[name] = {"image": image, "status": status}
            assert images[0] == images[2], f"{kind}: returning to GGX changed the seeded image"
            mse = sum((a - b) ** 2 for a, b in zip(images[0], images[1])) / len(images[0])
            assert mse > 1e-8, f"{kind}: Beckmann edit had no measurable effect"
            results[kind + "_distribution_mse"] = mse
            for node in temporary:
                tree.nodes.remove(node)
    finally:
        (artifacts / "result.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        scene.render.engine = "BLENDER_EEVEE"
        kiraray.unregister()
    print("KRR_BLENDER_DISTRIBUTIONS_SUCCESS", flush=True)


if __name__ == "__main__":
    main()
