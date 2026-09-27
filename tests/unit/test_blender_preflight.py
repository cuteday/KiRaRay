import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace as Struct
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "integrations/blender"))

from kiraray.preflight import validate_depsgraph, validate_material


class Socket:
    def __init__(self, name, value=0.0, identifier=None, **kwargs):
        self.name = name
        self.identifier = identifier or name
        self.default_value = value
        self.links = []
        self.enabled = True
        self.is_unavailable = False
        self.__dict__.update(kwargs)


class Node:
    def __init__(self, kind, inputs=(), outputs=("Value",), name=None, **kwargs):
        self.type = kind
        self.name = name or kind
        self.inputs = list(inputs)
        self.outputs = [value if isinstance(value, Socket) else Socket(value) for value in outputs]
        self.mute = False
        self.internal_links = []
        self.__dict__.update(kwargs)


class Material:
    def __init__(self, nodes, pointer=123, name="Material"):
        self.node_tree = Struct(nodes=list(nodes))
        self.name_full = name
        self.pointer = pointer

    def as_pointer(self):
        return self.pointer

    def evaluated_get(self, depsgraph):
        return getattr(self, "evaluated", self)


def socket(collection, name):
    return collection[name] if isinstance(name, int) else next(item for item in collection if item.name == name)


def connect(source, target, input_name, output_name=0):
    output = socket(source.outputs, output_name)
    input = socket(target.inputs, input_name)
    link = Struct(from_node=source, from_socket=output, to_socket=input, is_valid=True)
    input.links.append(link)
    return link


def material():
    defaults = {
        "Base Color": (0.8, 0.8, 0.8, 1), "Metallic": 0, "Roughness": 0.5,
        "IOR": 1.5, "Alpha": 1, "Thin Wall": False, "Normal": (0, 0, 0),
        "Diffuse Roughness": 0, "Specular IOR Level": 0.5, "Specular Tint": (1, 1, 1, 1),
        "Anisotropic": 0, "Anisotropic Rotation": 0, "Tangent": (0, 0, 0),
        "Transmission Weight": 0, "Coat Weight": 0, "Coat Roughness": 0.03,
        "Coat IOR": 1.5, "Coat Tint": (1, 1, 1, 1), "Coat Normal": (0, 0, 0),
        "Sheen Weight": 0, "Sheen Roughness": 0.5, "Sheen Tint": (1, 1, 1, 1),
        "Emission Color": (1, 1, 1, 1), "Emission Strength": 0,
        "Subsurface Weight": 0, "Subsurface Radius": (1, 0.2, 0.1),
        "Subsurface Scale": 0.005, "Subsurface IOR": 1.4, "Subsurface Anisotropy": 0,
        "Thin Film Thickness": 0, "Thin Film IOR": 1.33,
    }
    bsdf = Node("BSDF_PRINCIPLED", [Socket(key, value) for key, value in defaults.items()], ("BSDF",))
    bsdf.inputs.append(Socket("Weight", is_unavailable=True))
    output = Node("OUTPUT_MATERIAL", [Socket("Surface", None), Socket("Volume", None),
                                     Socket("Displacement", (0, 0, 0))], (), is_active_output=True)
    connect(bsdf, output, "Surface")
    return Material([bsdf, output]), bsdf, output


def image_node(**kwargs):
    defaults = {"image": Struct(source="FILE", colorspace_settings=Struct(name="sRGB")),
                "projection": "FLAT", "interpolation": "Linear", "extension": "REPEAT"}
    defaults.update(kwargs)
    return Node("TEX_IMAGE", [Socket("Vector", (0, 0, 0))], ("Color", "Alpha"), **defaults)


def math_node(operation="MULTIPLY", values=(0.0, 1.0, 0.0)):
    return Node("MATH", [Socket("Value", value, "Value_" + str(index))
                         for index, value in enumerate(values)], operation=operation, use_clamp=False)


def passthrough_group(name="Group"):
    input = Node("GROUP_INPUT", outputs=(Socket("Inner input", identifier="Input_1"),))
    output = Node("GROUP_OUTPUT", [Socket("Inner output", identifier="Output_1")], (),
                  is_active_output=True)
    connect(input, output, 0)
    group = Node("GROUP", [Socket("Outer input", identifier="Input_1")],
                 (Socket("Outer output", identifier="Output_1"),), name=name,
                 node_tree=Struct(nodes=[input, output]))
    return group


class MaterialPreflightTest(unittest.TestCase):
    def test_lossy_implicit_color_and_vector_scalar_conversions(self):
        for kind in ("RGBA", "VECTOR"):
            material_value, bsdf, output = material()
            scalar = socket(bsdf.inputs, "Roughness")
            scalar.type = "VALUE"
            rgb = Node("RGB", outputs=(Socket("Color", (0.1, 0.4, 0.7, 1), type=kind),))
            material_value.node_tree.nodes.append(rgb)
            connect(rgb, bsdf, "Roughness")
            self.assertIn("lossy_conversion", {item.code for item in validate_material(material_value).diagnostics})

    def assertSupported(self, value, **context):
        report = validate_material(value, **context)
        self.assertTrue(report.supported, report.to_dict())
        return report

    def assertRejected(self, value, code, text=""):
        report = validate_material(value)
        self.assertFalse(report.supported)
        self.assertTrue(any(item.code == code and text in item.message for item in report.diagnostics),
                        report.to_dict())
        return report

    def test_default_principled_and_report_serialization(self):
        value, bsdf, output = material()
        original_inputs = [(item.default_value, list(item.links)) for item in bsdf.inputs]
        report = self.assertSupported(value)
        self.assertEqual(json.loads(json.dumps(report.to_dict())), {
            "material": "Material", "pointer": "0x7b", "supported": True, "diagnostics": []})
        self.assertEqual(original_inputs, [(item.default_value, item.links) for item in bsdf.inputs])

    def test_textured_coat_normals_anisotropy_and_uv_mapping(self):
        value, bsdf, output = material()
        uv = Node("UVMAP", outputs=("UV",), uv_map="UVMap", from_instancer=False)
        mapping = Node("MAPPING", [Socket("Vector", (0, 0, 0)), Socket("Location", (0.2, 0, 0)),
                                   Socket("Rotation", (0, 0, 0.5)), Socket("Scale", (2, 2, 1))],
                       ("Vector",), vector_type="POINT")
        image = image_node()
        connect(uv, mapping, "Vector")
        connect(mapping, image, "Vector")
        connect(image, bsdf, "Base Color", "Color")
        socket(bsdf.inputs, "Coat Weight").default_value = 1
        socket(bsdf.inputs, "Anisotropic").default_value = 0.5
        connect(image, bsdf, "Anisotropic Rotation", "Alpha")
        for name in ("Normal", "Coat Normal"):
            normal = Node("NORMAL_MAP", [Socket("Strength", 1), Socket("Color", (0.5, 0.5, 1, 1))],
                          ("Normal",), name=name, space="TANGENT", uv_map="", convention="OPENGL")
            connect(image_node(image=Struct(source="FILE", colorspace_settings=Struct(name="Non-Color"))),
                    normal, "Color")
            connect(normal, bsdf, name)
        connect(Node("TANGENT", direction_type="UV_MAP", uv_map=""), bsdf, "Tangent")
        self.assertSupported(value, active_uv_names={"UVMap"})

    def test_named_uv_map_requires_matching_binding_context(self):
        value, bsdf, output = material()
        uv = Node("UVMAP", outputs=("UV",), uv_map="UVMap", from_instancer=False)
        connect(uv, bsdf, "Base Color", "UV")
        self.assertSupported(value, active_uv_names={"UVMap"})
        self.assertRejected(value, "unsupported_node", "bindings")
        for names in ({"Other"}, {"UVMap", "Other"}, {""}, {"UVMap", ""}, set()):
            with self.subTest(names=names):
                report = validate_material(value, active_uv_names=names)
                self.assertFalse(report.supported)
                self.assertTrue(any("every bound mesh" in item.message for item in report.diagnostics))
        uv.uv_map = ""
        self.assertSupported(value)
        self.assertSupported(value, active_uv_names={""})

    def test_unreachable_nodes_and_inactive_outputs_are_ignored(self):
        value, bsdf, output = material()
        ramp = Node("VALTORGB", interpolation="LINEAR")
        inactive = Node("OUTPUT_MATERIAL", [Socket("Surface", None)], (), is_active_output=False)
        connect(ramp, inactive, "Surface")
        value.node_tree.nodes.extend([ramp, inactive])
        self.assertSupported(value)
        connect(ramp, bsdf, "Base Color")
        report = self.assertRejected(value, "unsupported_node", "ColorRamp")
        self.assertIn("materials['Material'].node_tree.nodes['VALTORGB']", report.diagnostics[0].source)

    def test_only_reachable_coordinate_output_is_checked(self):
        for name in ("UV", "Generated", "Object", "Normal"):
            with self.subTest(output=name):
                value, bsdf, output = material()
                coords = Node("TEX_COORD", outputs=("UV", "Generated", "Object", "Normal"))
                connect(coords, bsdf, "Base Color", name)
                if name == "UV":
                    self.assertSupported(value)
                else:
                    self.assertRejected(value, "unsupported_node", "Only UV")

    def test_disabled_features_do_not_validate_inactive_parameters(self):
        pairs = (("Coat Weight", "Coat Normal"), ("Sheen Weight", "Sheen Tint"),
                 ("Emission Strength", "Emission Color"), ("Anisotropic", "Anisotropic Rotation"),
                 ("Subsurface Weight", "Subsurface Radius"), ("Thin Film Thickness", "Thin Film IOR"))
        for weight, parameter in pairs:
            with self.subTest(weight=weight):
                value, bsdf, output = material()
                connect(Node("VALTORGB"), bsdf, parameter)
                self.assertSupported(value)
                socket(bsdf.inputs, weight).default_value = 0.3
                self.assertFalse(validate_material(value).supported)

    def test_unsupported_features_reject_nonzero_and_dynamic_weights(self):
        for name in ("Subsurface Weight", "Thin Film Thickness"):
            value, bsdf, output = material()
            socket(bsdf.inputs, name).default_value = 1
            self.assertRejected(value, "unsupported_feature")
            socket(bsdf.inputs, name).default_value = 0
            connect(image_node(), bsdf, name, "Alpha")
            self.assertRejected(value, "unsupported_feature")

    def test_constant_zero_math_and_group_defaults_disable_features(self):
        value, bsdf, output = material()
        group = passthrough_group()
        socket(group.inputs, 0).default_value = 0
        multiply = math_node(values=(2, 0, 0))
        connect(group, multiply, 1)
        connect(multiply, bsdf, "Subsurface Weight")
        connect(Node("VALTORGB"), bsdf, "Subsurface Radius")
        self.assertSupported(value)
        socket(group.inputs, 0).default_value = 1
        self.assertRejected(value, "unsupported_feature")

    def test_groups_resolve_identifiers_and_each_instance_context(self):
        value, bsdf, output = material()
        first = passthrough_group("First")
        second = passthrough_group("Second")
        second.node_tree = first.node_tree
        connect(Node("RGB", outputs=(Socket("Color", (1, 0, 0, 1)),)), first, 0)
        connect(Node("VALTORGB", name="Bad ramp"), second, 0)
        connect(first, bsdf, "Base Color")
        connect(second, bsdf, "Roughness")
        self.assertRejected(value, "unsupported_node", "ColorRamp")
        socket(second.inputs, 0).links.clear()
        self.assertSupported(value)

    def test_nested_group_path_and_unused_group_inputs(self):
        value, bsdf, output = material()
        outer = passthrough_group("Outer")
        inner = passthrough_group("Inner")
        inner.node_tree.nodes[-1].inputs[0].links.clear()
        connect(Node("VALTORGB", name="Ramp"), inner.node_tree.nodes[-1], 0)
        outer.node_tree.nodes[-1].inputs[0].links.clear()
        connect(inner, outer.node_tree.nodes[-1], 0)
        connect(Node("TEX_NOISE"), outer, 0)
        connect(outer, bsdf, "Base Color")
        report = self.assertRejected(value, "unsupported_node", "ColorRamp")
        self.assertEqual(report.diagnostics[0].source,
                         "materials['Material'].node_tree.nodes['Outer'].node_tree.nodes['Inner']"
                         ".node_tree.nodes['Ramp']")
        inner.node_tree.nodes[-1].inputs[0].links.clear()
        self.assertSupported(value)

    def test_group_surface_and_muted_nodes_follow_bypass(self):
        value, bsdf, output = material()
        group = passthrough_group()
        socket(output.inputs, "Surface").links.clear()
        connect(bsdf, group, 0)
        connect(group, output, "Surface")
        muted = Node("VALTORGB", [Socket("Fac", 0.2)], mute=True)
        muted.internal_links = [Struct(from_socket=muted.inputs[0], to_socket=muted.outputs[0])]
        reroute = Node("REROUTE", [Socket("Input")])
        connect(muted, reroute, 0)
        connect(reroute, bsdf, "Roughness")
        self.assertSupported(value)

    def test_cycles_and_recursive_groups_fail_without_recursing_forever(self):
        value, bsdf, output = material()
        reroute = Node("REROUTE", [Socket("Input")])
        connect(reroute, reroute, 0)
        connect(reroute, bsdf, "Base Color")
        self.assertRejected(value, "cyclic_graph")
        socket(bsdf.inputs, "Base Color").links.clear()
        group = passthrough_group()
        group.node_tree.nodes[-1].inputs[0].links.clear()
        connect(group, group.node_tree.nodes[-1], 0)
        connect(group, bsdf, "Base Color")
        self.assertRejected(value, "invalid_group")

    def test_image_settings_are_validated_before_conversion(self):
        invalid = [dict(image=None), dict(projection="BOX"), dict(interpolation="Cubic"),
                   dict(image=Struct(source="TILED", colorspace_settings=Struct(name="sRGB"))),
                   dict(image=Struct(source="MOVIE", colorspace_settings=Struct(name="sRGB"))),
                   dict(image=Struct(source="FILE", colorspace_settings=Struct(name="ACEScg")))]
        for settings in invalid:
            with self.subTest(settings=settings):
                value, bsdf, output = material()
                connect(image_node(**settings), bsdf, "Base Color")
                self.assertRejected(value, "unsupported_node")

    def test_normal_and_tangent_settings_that_exporter_drops_are_rejected(self):
        for settings in (dict(space="OBJECT"), dict(space="WORLD"), dict(uv_map="Named UV"),
                         dict(convention="DIRECTX")):
            value, bsdf, output = material()
            attrs = dict(space="TANGENT", uv_map="", convention="OPENGL")
            attrs.update(settings)
            connect(Node("NORMAL_MAP", [Socket("Strength", 1), Socket("Color", (0.5, 0.5, 1, 1))],
                         **attrs), bsdf, "Normal")
            self.assertRejected(value, "unsupported_node")
        value, bsdf, output = material()
        socket(bsdf.inputs, "Anisotropic").default_value = 1
        connect(Node("TANGENT", direction_type="RADIAL", uv_map=""), bsdf, "Tangent")
        self.assertRejected(value, "unsupported_node")

    def test_mix_checks_only_selected_inputs_and_rejects_other_blends(self):
        for data_type, active_index in (("FLOAT", 2), ("VECTOR", 4), ("RGBA", 6)):
            with self.subTest(data_type=data_type):
                value, bsdf, output = material()
                mix = Node("MIX", [Socket(str(index)) for index in range(8)],
                           data_type=data_type, factor_mode="UNIFORM", blend_type="MIX")
                unused = 6 if data_type != "RGBA" else 2
                connect(Node("VALTORGB"), mix, unused)
                connect(mix, bsdf, "Base Color")
                self.assertSupported(value)
                connect(Node("VALTORGB"), mix, active_index)
                self.assertRejected(value, "unsupported_node", "ColorRamp")
                socket(mix.inputs, active_index).links.clear()
                if data_type == "RGBA":
                    mix.blend_type = "MULTIPLY"
                    self.assertRejected(value, "unsupported_node", "blend modes")

    def test_linear_remap_and_math_ignore_unused_sockets(self):
        value, bsdf, output = material()
        remap = Node("MAP_RANGE", [Socket(str(index), float(index)) for index in range(12)],
                     interpolation_type="LINEAR", data_type="FLOAT")
        connect(Node("VALTORGB"), remap, 5)
        multiply = math_node()
        connect(Node("VALTORGB"), multiply, 2)
        connect(multiply, remap, 0)
        connect(remap, bsdf, "Roughness")
        self.assertSupported(value)
        remap.interpolation_type = "STEPPED"
        self.assertRejected(value, "unsupported_node", "linear")
        remap.interpolation_type = "LINEAR"
        multiply.operation = "POWER"
        self.assertRejected(value, "unsupported_node", "POWER")

    def test_mapping_requires_constant_transform(self):
        value, bsdf, output = material()
        mapping = Node("MAPPING", [Socket("Vector", (0, 0, 0)), Socket("Rotation", (0, 0, 0))],
                       vector_type="POINT")
        connect(image_node(), mapping, "Rotation")
        connect(mapping, bsdf, "Base Color")
        self.assertRejected(value, "dynamic_mapping")

    def test_missing_output_invalid_links_and_nonfinite_values(self):
        value, bsdf, output = material()
        value.node_tree.nodes.remove(output)
        self.assertRejected(value, "missing_output")
        value.node_tree.nodes.append(output)
        socket(output.inputs, "Surface").links.clear()
        self.assertRejected(value, "missing_surface")
        connect(bsdf, output, "Surface").is_valid = False
        self.assertRejected(value, "invalid_link")
        socket(output.inputs, "Surface").links[0].is_valid = True
        for number in (math.nan, math.inf, -math.inf):
            socket(bsdf.inputs, "Roughness").default_value = number
            self.assertRejected(value, "nonfinite_value")

    def test_volume_displacement_and_other_surface_closures_are_rejected(self):
        for name in ("Volume", "Displacement"):
            value, bsdf, output = material()
            connect(Node("TEX_NOISE"), output, name)
            self.assertRejected(value, "unsupported_output", name)
        value, bsdf, output = material()
        socket(output.inputs, "Surface").links.clear()
        connect(Node("MIX_SHADER"), output, "Surface")
        self.assertRejected(value, "unsupported_surface")


class DepsgraphPreflightTest(unittest.TestCase):
    def test_shared_material_checks_all_bound_active_and_render_uv_maps(self):
        class UVLayers(list):
            def __init__(self, active, render):
                super().__init__(Struct(name=name, active_render=name == render)
                                 for name in dict.fromkeys((active, render)) if name)
                self.active = next((layer for layer in self if layer.name == active), None)

        value, bsdf, output = material()
        connect(Node("UVMAP", outputs=("UV",), uv_map="UVMap", from_instancer=False),
                bsdf, "Base Color", "UV")

        def instance(active="UVMap", render="UVMap"):
            return Struct(object=Struct(material_slots=[Struct(material=value)],
                data=Struct(polygons=[Struct(material_index=0)], uv_layers=UVLayers(active, render))))

        for active, render, supported in (("UVMap", "UVMap", True),
                                          ("Other", "Other", False),
                                          ("UVMap", "Other", False),
                                          ("", "", False)):
            with self.subTest(active=active, render=render):
                graph = Struct(object_instances=[instance(), instance(active, render)])
                reports = validate_depsgraph(graph)
                self.assertEqual(set(reports), {value.pointer})
                self.assertEqual(reports[value.pointer].supported, supported)

    def test_evaluated_identity_used_slots_and_instance_deduplication(self):
        good, bsdf, output = material()
        original, unused, output = material()
        original.pointer = 55
        original.evaluated = good
        bad, bsdf, output = material()
        bad.pointer = 456
        connect(Node("VALTORGB"), bsdf, "Base Color")
        obj = Struct(material_slots=[Struct(material=original), Struct(material=bad), Struct(material=None)],
                     data=Struct(polygons=[Struct(material_index=0)]))
        instance = Struct(object=obj)
        depsgraph = Struct(object_instances=[instance, instance])
        reports = validate_depsgraph(depsgraph)
        self.assertEqual(set(reports), {123})
        self.assertTrue(reports[123].supported)
        obj.data.polygons.extend([Struct(material_index=1), Struct(material_index=2)])
        reports = validate_depsgraph(depsgraph)
        self.assertEqual(set(reports), {123, 456})
        self.assertFalse(reports[456].supported)
        obj.data.polygons.clear()
        self.assertEqual(validate_depsgraph(depsgraph), {})


if __name__ == "__main__":
    unittest.main()
