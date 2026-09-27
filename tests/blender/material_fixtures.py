"""Validate the Blender producer and export fixtures, without rendering.

blender --background --factory-startup --python-exit-code 1 \
    --python tests/blender/material_fixtures.py -- --output build/tests/blender-materials
"""

import argparse
import json
from pathlib import Path
import sys

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "integrations/blender"))

from kiraray.preflight import validate_depsgraph, validate_material


def new_material(name):
    material = bpy.data.materials.new(name)
    return material, material.node_tree.nodes.get("Principled BSDF")


def texture(tree, image):
    node = tree.nodes.new("ShaderNodeTexImage")
    node.image = image
    return node


def make_image(directory, name, color, colorspace):
    image = bpy.data.images.new(name, width=2, height=2)
    image.colorspace_settings.name = colorspace
    image.pixels = list(color) * 4
    image.filepath_raw = str(directory / (name + ".png"))
    image.file_format = "PNG"
    image.save()
    return image


def supported_material(color_image, normal_image):
    material, bsdf = new_material("SupportedTextured")
    tree = material.node_tree
    uv = tree.nodes.new("ShaderNodeUVMap")
    uv.uv_map = "UVMap"
    mapping = tree.nodes.new("ShaderNodeMapping")
    mapping.inputs["Rotation"].default_value = (0.1, 0.2, 0.3)
    mapping.inputs["Scale"].default_value = (2, 1, 1)
    image = texture(tree, color_image)
    tree.links.new(uv.outputs["UV"], mapping.inputs["Vector"])
    tree.links.new(mapping.outputs["Vector"], image.inputs["Vector"])
    mix = tree.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.blend_type = "MIX"
    mix.inputs[0].default_value = 0.4
    mix.inputs[7].default_value = (0.8, 0.2, 0.1, 1)
    tree.links.new(image.outputs["Color"], mix.inputs[6])
    tree.links.new(mix.outputs[2], bsdf.inputs["Base Color"])
    math = tree.nodes.new("ShaderNodeMath")
    math.operation = "MULTIPLY_ADD"
    math.inputs[1].default_value = 0.5
    math.inputs[2].default_value = 0.1
    tree.links.new(image.outputs["Alpha"], math.inputs[0])
    remap = tree.nodes.new("ShaderNodeMapRange")
    remap.interpolation_type = "LINEAR"
    tree.links.new(math.outputs[0], remap.inputs["Value"])
    tree.links.new(remap.outputs["Result"], bsdf.inputs["Roughness"])
    for name in ("Normal", "Coat Normal"):
        normal = tree.nodes.new("ShaderNodeNormalMap")
        normal.name = name
        normal.inputs["Strength"].default_value = 0.5 if name == "Normal" else 0.25
        tree.links.new(texture(tree, normal_image).outputs["Color"], normal.inputs["Color"])
        tree.links.new(normal.outputs["Normal"], bsdf.inputs[name])
    bsdf.inputs["Coat Weight"].default_value = 0.8
    bsdf.inputs["Sheen Weight"].default_value = 0.1
    bsdf.inputs["Anisotropic"].default_value = 0.4
    tree.links.new(image.outputs["Alpha"], bsdf.inputs["Anisotropic Rotation"])
    tangent = tree.nodes.new("ShaderNodeTangent")
    tangent.direction_type = "UV_MAP"
    tree.links.new(tangent.outputs["Tangent"], bsdf.inputs["Tangent"])
    return material


def grouped_material():
    material, bsdf = new_material("SupportedGrouped")
    tree = bpy.data.node_groups.new("ColorGroup", "ShaderNodeTree")
    tree.interface.new_socket(name="Tint", in_out="INPUT", socket_type="NodeSocketColor")
    tree.interface.new_socket(name="Color", in_out="OUTPUT", socket_type="NodeSocketColor")
    input = tree.nodes.new("NodeGroupInput")
    output = tree.nodes.new("NodeGroupOutput")
    tree.links.new(input.outputs["Tint"], output.inputs["Color"])
    group = material.node_tree.nodes.new("ShaderNodeGroup")
    group.node_tree = tree
    group.inputs["Tint"].default_value = (0.2, 0.4, 0.8, 1)
    material.node_tree.links.new(group.outputs["Color"], bsdf.inputs["Base Color"])
    return material


def expression_materials(image):
    result = []
    for operation in ("ADD", "SUBTRACT", "MULTIPLY", "DIVIDE", "MULTIPLY_ADD",
                      "MINIMUM", "MAXIMUM", "LESS_THAN", "GREATER_THAN"):
        material, bsdf = new_material("SupportedMath" + operation.title().replace("_", ""))
        tree = material.node_tree
        node = tree.nodes.new("ShaderNodeMath")
        node.operation = operation
        node.use_clamp = True
        node.inputs[1].default_value = 0.3
        node.inputs[2].default_value = 0.1
        tree.links.new(texture(tree, image).outputs["Alpha"], node.inputs[0])
        tree.links.new(node.outputs[0], bsdf.inputs["Roughness"])
        result.append(material)
    for mode in ("POINT", "TEXTURE", "VECTOR", "NORMAL"):
        material, bsdf = new_material("SupportedMapping" + mode.title())
        tree = material.node_tree
        uv = tree.nodes.new("ShaderNodeTexCoord")
        mapping = tree.nodes.new("ShaderNodeMapping")
        mapping.vector_type = mode
        if "Location" in mapping.inputs:
            mapping.inputs["Location"].default_value = (0.1, 0.2, 0.3)
        mapping.inputs["Rotation"].default_value = (0.2, 0.3, 0.4)
        mapping.inputs["Scale"].default_value = (2, 3, 1)
        tex = texture(tree, image)
        tree.links.new(uv.outputs["UV"], mapping.inputs["Vector"])
        tree.links.new(mapping.outputs["Vector"], tex.inputs["Vector"])
        tree.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
        result.append(material)
    for data_type in ("FLOAT", "VECTOR"):
        material, bsdf = new_material("SupportedMix" + data_type.title())
        tree = material.node_tree
        tex = texture(tree, image)
        mix = tree.nodes.new("ShaderNodeMix")
        mix.data_type = data_type
        tree.links.new(tex.outputs["Alpha"], mix.inputs[0])
        input_index, output_index = (2, 0) if data_type == "FLOAT" else (4, 1)
        tree.links.new(tex.outputs["Alpha" if data_type == "FLOAT" else "Color"], mix.inputs[input_index])
        tree.links.new(mix.outputs[output_index], bsdf.inputs["Base Color"])
        result.append(material)
    material, bsdf = new_material("SupportedVectorRemap")
    tree = material.node_tree
    remap = tree.nodes.new("ShaderNodeMapRange")
    remap.data_type = "FLOAT_VECTOR"
    remap.interpolation_type = "LINEAR"
    tree.links.new(texture(tree, image).outputs["Color"], remap.inputs["Vector"])
    tree.links.new(remap.outputs["Vector"], bsdf.inputs["Base Color"])
    result.append(material)
    material, bsdf = new_material("SupportedChannels")
    tree = material.node_tree
    tex = texture(tree, image)
    separate = tree.nodes.new("ShaderNodeSeparateColor")
    combine = tree.nodes.new("ShaderNodeCombineColor")
    tree.links.new(tex.outputs["Color"], separate.inputs[0])
    for source, destination in (("Red", "Blue"), ("Green", "Red"), ("Blue", "Green")):
        tree.links.new(separate.outputs[source], combine.inputs[destination])
    separate_xyz = tree.nodes.new("ShaderNodeSeparateXYZ")
    combine_xyz = tree.nodes.new("ShaderNodeCombineXYZ")
    tree.links.new(combine.outputs[0], separate_xyz.inputs[0])
    for source, destination in (("X", "Z"), ("Y", "X"), ("Z", "Y")):
        tree.links.new(separate_xyz.outputs[source], combine_xyz.inputs[destination])
    tree.links.new(combine_xyz.outputs[0], bsdf.inputs["Base Color"])
    clamp = tree.nodes.new("ShaderNodeClamp")
    tree.links.new(separate.outputs["Red"], clamp.inputs["Value"])
    tree.links.new(clamp.outputs[0], bsdf.inputs["Roughness"])
    result.append(material)
    return result


def unsupported_materials():
    result = []
    for name, node_type, output_name, destination, settings in (
        ("UnsupportedRamp", "ShaderNodeValToRGB", "Color", "Base Color", {}),
        ("UnsupportedGenerated", "ShaderNodeTexCoord", "Generated", "Base Color", {}),
        ("UnsupportedBlend", "ShaderNodeMix", "Result", "Base Color",
         {"data_type": "RGBA", "blend_type": "MULTIPLY"}),
        ("UnsupportedNormal", "ShaderNodeNormalMap", "Normal", "Normal", {"space": "OBJECT"}),
    ):
        material, bsdf = new_material(name)
        node = material.node_tree.nodes.new(node_type)
        for key, value in settings.items():
            setattr(node, key, value)
        output = node.outputs[2] if node_type == "ShaderNodeMix" else node.outputs[output_name]
        material.node_tree.links.new(output, bsdf.inputs[destination])
        result.append(material)
    for name, input in (("UnsupportedSubsurface", "Subsurface Weight"),
                        ("UnsupportedThinFilm", "Thin Film Thickness")):
        material, bsdf = new_material(name)
        bsdf.inputs[input].default_value = 1
        result.append(material)
    material, bsdf = new_material("UnsupportedImplicitColorToFloat")
    rgb = material.node_tree.nodes.new("ShaderNodeRGB")
    rgb.outputs[0].default_value = (0.1, 0.4, 0.7, 1)
    material.node_tree.links.new(rgb.outputs[0], bsdf.inputs["Roughness"])
    result.append(material)
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
    color = make_image(directory, "color", (0.2, 0.4, 0.8, 1), "sRGB")
    normal = make_image(directory, "normal", (0.5, 0.5, 1, 1), "Non-Color")
    supported = [supported_material(color, normal), grouped_material(), *expression_materials(color)]
    unsupported = unsupported_materials()
    materials = supported + unsupported
    for index, material in enumerate(materials):
        bpy.ops.mesh.primitive_plane_add(location=(index * 3, 0, 0))
        bpy.context.object.name = material.name
        bpy.context.object.data.materials.append(material)
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    reports = validate_depsgraph(depsgraph)
    for material in materials:
        evaluated = material.evaluated_get(depsgraph)
        report = reports[evaluated.as_pointer()]
        expected = material in supported
        if report.supported != expected:
            raise AssertionError(json.dumps(report.to_dict(), indent=2))
        if validate_material(evaluated, active_uv_names={"UVMap"}) != report:
            raise AssertionError("Direct and depsgraph preflight disagree")
    report_json = {"blender": bpy.app.version_string, "materials": [report.to_dict() for report in reports.values()]}
    (directory / "preflight.json").write_text(json.dumps(report_json, indent=2) + "\n", encoding="utf-8")
    result = bpy.ops.wm.usd_export(filepath=str(directory / "materials.usda"), check_existing=False,
                                   generate_materialx_network=True, generate_preview_surface=False,
                                   export_animation=False, export_materials=True,
                                   export_lights=False, export_cameras=False, relative_paths=True)
    if result != {"FINISHED"}:
        raise RuntimeError("USD fixture export failed: " + repr(result))
    usd = (directory / "materials.usda").read_text(encoding="utf-8")
    if "ND_open_pbr_surface_surfaceshader" not in usd:
        raise AssertionError("Blender did not emit the expected OpenPBR MaterialX network")
    print("Blender producer fixtures passed: {} supported, {} rejected; {}".format(
        len(supported), len(unsupported), directory))


if __name__ == "__main__":
    main()
