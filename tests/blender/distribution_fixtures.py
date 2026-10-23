"""Export Blender 5.2.2 microfacet fixtures and verify producer path identities.

blender --background --factory-startup --disable-autoexec --python-exit-code 1 \
    --python tests/blender/distribution_fixtures.py -- --output build/tests/distributions
"""

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "integrations/blender"))
from kiraray import engine
from kiraray.preflight import validate_depsgraph


def leaf(tree, kind="Glossy", distribution="BECKMANN"):
    shader = tree.nodes.new("ShaderNodeBsdf" + kind)
    shader.distribution = distribution
    shader.inputs["Roughness"].default_value = 0.3
    return shader


def combine(tree, first, second, kind="Mix"):
    shader = tree.nodes.new("ShaderNode" + kind + "Shader")
    tree.links.new(first.outputs[0], shader.inputs[-2])
    tree.links.new(second.outputs[0], shader.inputs[-1])
    if kind == "Mix":
        shader.inputs[0].default_value = 0.35
    return shader


def grouped(tree, source):
    group_tree = bpy.data.node_groups.new("Shared / group", "ShaderNodeTree")
    group_tree.interface.new_socket(name="Shader", in_out="INPUT", socket_type="NodeSocketShader")
    group_tree.interface.new_socket(name="Shader", in_out="OUTPUT", socket_type="NodeSocketShader")
    input = group_tree.nodes.new("NodeGroupInput")
    output = group_tree.nodes.new("NodeGroupOutput")
    group_tree.links.new(input.outputs[0], output.inputs[0])
    result = tree.nodes.new("ShaderNodeGroup")
    result.node_tree = group_tree
    tree.links.new(source.outputs[0], result.inputs[0])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    if bpy.app.version != (5, 2, 2):
        raise RuntimeError("These producer fixtures require Blender 5.2.2")
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    cases = ("Glossy", "Glass", "Mixed", "AddEmission", "MixEmission", "Grouped", "SharedGroup",
             "InsideGroup", "Transparent", "DeadBranch")
    for index, case in enumerate(cases):
        material = bpy.data.materials.new("Distribution" + case)
        tree = material.node_tree
        tree.nodes.clear()
        first = leaf(tree, "Glass" if case == "Glass" else "Glossy")
        first.name = "Same / node name"
        result = first
        if case in {"Mixed", "DeadBranch"}:
            second = leaf(tree, distribution="GGX")
            second.name = "Same _ node name"
            result = combine(tree, first, second)
            if case == "DeadBranch":
                result.inputs[0].default_value = 0
        elif case in {"AddEmission", "MixEmission"}:
            result = combine(tree, first, tree.nodes.new("ShaderNodeEmission"), case[:-8])
        elif case in {"Grouped", "SharedGroup"}:
            group = grouped(tree, first)
            if case == "Grouped":
                result = combine(tree, grouped(tree, group), leaf(tree, "Glass", "GGX"))
            else:
                second = tree.nodes.new("ShaderNodeGroup")
                second.node_tree = group.node_tree
                tree.links.new(first.outputs[0], second.inputs[0])
                result = combine(tree, group, second, "Add")
        elif case == "InsideGroup":
            group_tree = bpy.data.node_groups.new("Mixed distributions", "ShaderNodeTree")
            group_tree.interface.new_socket(name="Shader", in_out="OUTPUT", socket_type="NodeSocketShader")
            mixed = combine(group_tree, leaf(group_tree), leaf(group_tree, "Glass", "GGX"))
            group_output = group_tree.nodes.new("NodeGroupOutput")
            group_tree.links.new(mixed.outputs[0], group_output.inputs[0])
            groups = [tree.nodes.new("ShaderNodeGroup") for _ in range(2)]
            for group in groups:
                group.node_tree = group_tree
            result = combine(tree, *groups, "Add")
        elif case == "Transparent":
            result = combine(tree, first, tree.nodes.new("ShaderNodeBsdfTransparent"))
        output = tree.nodes.new("ShaderNodeOutputMaterial")
        tree.links.new(result.outputs[0], output.inputs["Surface"])
        bpy.ops.mesh.primitive_plane_add(location=(index * 3, 0, 0))
        bpy.context.object.name = material.name
        bpy.context.object.data.materials.append(material)
    bpy.context.view_layer.update()
    reports = validate_depsgraph(bpy.context.evaluated_depsgraph_get())
    for report in reports.values():
        assert report.supported, report.to_dict()
        assert report.distributions, report.to_dict()
    (directory / "preflight.json").write_text(json.dumps(
        [report.to_dict() for report in reports.values()], indent=2), encoding="utf-8")
    path = directory / "distributions.usda"
    operator = SimpleNamespace(filepath=str(path), report=lambda level, message: print(message))
    assert engine.KiRaRayExportUSD.execute(operator, bpy.context) == {"FINISHED"}
    bpy.utils.expose_bundled_modules()
    from pxr import Usd, UsdShade
    stage = Usd.Stage.Open(str(path))
    count = 0
    for prim in stage.Traverse():
        if not prim.IsA(UsdShade.Material):
            continue
        assert not prim.GetCustomDataByKey("kiraray:diagnostics"), str(prim.GetPath())
        annotations = json.loads(prim.GetCustomDataByKey("kiraray:distributions"))
        terminal = UsdShade.Material(prim).GetSurfaceOutput("mtlx")
        for annotation in annotations:
            sources = UsdShade.Utils.GetValueProducingAttributes(terminal)
            shader = UsdShade.Shader(sources[0].GetPrim())
            for edge in annotation["path"].split("/"):
                sources = UsdShade.Utils.GetValueProducingAttributes(shader.GetInput(edge))
                assert len(sources) == 1, annotation
                shader = UsdShade.Shader(sources[0].GetPrim())
            assert shader.GetIdAttr().Get() == annotation["identifier"], annotation
            # Blender itself still omits this authored choice.
            assert not shader.GetInput("distribution").Get(), annotation
        count += 1
    assert count == len(cases), count
    print(f"Blender distribution fixtures passed: {count}; {path}")


if __name__ == "__main__":
    main()
