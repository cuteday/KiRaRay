"""Validate Blender producer diagnostics and the annotated USD export."""

import argparse
import json
from pathlib import Path
import sys

import bpy

sys.path.insert(0, str(Path(__file__).parent))
from render_scene import cornell


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
    box, white = cornell()
    reports, _, messages = kiraray.engine._diagnostics(bpy.context.evaluated_depsgraph_get())
    assert reports and not messages, messages
    kiraray.engine._validate_color_space()
    ramp = white.node_tree.nodes.new("ShaderNodeValToRGB")
    white.node_tree.links.new(ramp.outputs["Color"], white.node_tree.nodes["Principled BSDF"].inputs["Base Color"])
    white.name = "White-A"
    collision = white.copy()
    collision.name = "White A"
    box.data.materials[0] = collision
    before = (len(white.node_tree.nodes), len(white.node_tree.links))
    path = args.artifacts / "unsupported.usdc"
    assert bpy.ops.kiraray.export_usd(filepath=str(path)) == {"FINISHED"}
    from pxr import Usd, UsdShade
    stage = Usd.Stage.Open(str(path))
    diagnostics = {str(prim.GetPath()): list(prim.GetCustomDataByKey("kiraray:diagnostics"))
                   for prim in stage.Traverse() if prim.IsA(UsdShade.Material)
                   and prim.GetCustomDataByKey("kiraray:diagnostics")}
    assert diagnostics and any("ColorRamp" in detail for details in diagnostics.values() for detail in details)
    details = [detail for group in diagnostics.values() for detail in group]
    assert all(any(name in detail for detail in details) for name in ("White-A", "White A")), \
        "Sanitized material-name collisions lost producer diagnostics"
    assert any("Ambiguous source" in detail for detail in details)
    assert all(prim.GetCustomDataByKey("kiraray:emissionLuminanceScale") == 1000.0
               for prim in stage.Traverse() if prim.IsA(UsdShade.Material))
    assert before == (len(white.node_tree.nodes), len(white.node_tree.links)), "Export mutated the source graph"
    bpy.ops.wm.set_working_color_space(working_space="ACEScg", convert_colors=False)
    try:
        kiraray.engine._validate_color_space()
    except RuntimeError as error:
        assert "Linear Rec.709" in str(error)
    else:
        raise AssertionError("Unsupported working color space was accepted")
    (args.artifacts / "result.json").write_text(json.dumps(diagnostics, indent=2))
    kiraray.unregister()
    print("KRR_BLENDER_EXPORT_SUCCESS", flush=True)


if __name__ == "__main__":
    main()
