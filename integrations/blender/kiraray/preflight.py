"""Validate the Blender 5.2 material subset before MaterialX conversion."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Diagnostic:
    code: str
    source: str
    message: str

    def to_dict(self):
        return {"code": self.code, "source": self.source, "message": self.message}


@dataclass(frozen=True)
class MaterialReport:
    name: str
    pointer: int
    diagnostics: tuple

    @property
    def supported(self):
        return not self.diagnostics

    def to_dict(self):
        return {"material": self.name, "pointer": hex(self.pointer),
                "supported": self.supported,
                "diagnostics": [item.to_dict() for item in self.diagnostics]}


_MATH_OPERATIONS = {
    "ADD", "SUBTRACT", "MULTIPLY", "DIVIDE", "MULTIPLY_ADD",
    "MINIMUM", "MAXIMUM", "LESS_THAN", "GREATER_THAN",
}
_PRINCIPLED_INPUTS = {
    "Base Color", "Metallic", "Roughness", "IOR", "Alpha", "Thin Wall", "Normal",
    "Diffuse Roughness", "Specular IOR Level", "Specular Tint", "Anisotropic",
    "Anisotropic Rotation", "Tangent", "Transmission Weight", "Coat Weight",
    "Coat Roughness", "Coat IOR", "Coat Tint", "Coat Normal", "Sheen Weight",
    "Sheen Roughness", "Sheen Tint", "Emission Color", "Emission Strength",
    "Subsurface Weight", "Subsurface Radius", "Subsurface Scale", "Subsurface IOR",
    "Subsurface Anisotropy", "Thin Film Thickness", "Thin Film IOR",
}
_NODE_TYPES = {
    "ShaderNodeSeparateXYZ": "SEPXYZ", "ShaderNodeCombineXYZ": "COMBXYZ",
    "ShaderNodeSeparateRGB": "SEPRGB", "ShaderNodeCombineRGB": "COMBRGB",
}
_UNKNOWN = object()


def _node_type(node):
    return _NODE_TYPES.get(getattr(node, "bl_idname", ""), node.type)


def _identity(value):
    return value.as_pointer() if hasattr(value, "as_pointer") else id(value)


def _is_zero(value):
    return isinstance(value, (int, float)) and value == 0


def _socket(collection, name):
    for item in collection:
        if item.identifier == name or item.name == name:
            return item
    return None


def _matching_socket(collection, other):
    return next((item for item in collection if item.identifier == other.identifier), None)


def _active_inputs(node):
    return [item for item in node.inputs
            if getattr(item, "enabled", True) and not getattr(item, "is_unavailable", False)]


def _output(tree, node_type):
    if node_type == "OUTPUT_MATERIAL" and hasattr(tree, "get_output_node"):
        return tree.get_output_node("ALL")
    nodes = [node for node in tree.nodes if _node_type(node) == node_type]
    active = [node for node in nodes if getattr(node, "is_active_output", False)]
    if len(active) == 1:
        return active[0]
    return nodes[0] if len(nodes) == 1 else None


class _Validator:
    def __init__(self, material, active_uv_names=None):
        self.material = material
        self.active_uv_names = None if active_uv_names is None else frozenset(active_uv_names)
        self.name = getattr(material, "name_full", getattr(material, "name", "<material>"))
        self.diagnostics = []
        self.visiting = set()
        self.visited = set()

    def path(self, node=None, socket=None, groups=()):
        path = "materials[" + repr(self.name) + "]"
        for group in groups:
            path += ".node_tree.nodes[" + repr(group.name) + "]"
        if node is not None:
            path += ".node_tree.nodes[" + repr(node.name) + "]"
        if socket is not None:
            path += ".inputs[" + repr(socket.identifier) + "]"
        return path

    def error(self, code, message, node=None, socket=None, groups=()):
        diagnostic = Diagnostic(code, self.path(node, socket, groups), message)
        if diagnostic not in self.diagnostics:
            self.diagnostics.append(diagnostic)

    def input(self, node, socket, groups=(), surface=False):
        if socket is None:
            self.error("missing_socket", "The material output has no Surface input.", node)
            return
        links = [link for link in socket.links if getattr(link, "is_valid", True)]
        if len(links) > 1:
            self.error("multiple_links", "Multiple links to one input are unsupported.",
                       node, socket, groups)
        elif links:
            link = links[0]
            source_type = getattr(link.from_socket, "type", None)
            target_type = getattr(socket, "type", None)
            if source_type in {"RGBA", "VECTOR"} and target_type in {"VALUE", "INT", "BOOLEAN"}:
                self.error("lossy_conversion", "Blender's exporter takes the first component for this "
                           "implicit conversion; use Separate Color or Separate XYZ explicitly.",
                           node, socket, groups)
            self.output(link.from_node, link.from_socket, groups, surface)
        elif socket.links:
            self.error("invalid_link", "The input has an invalid node link.", node, socket, groups)
        elif surface:
            self.error("missing_surface", "Connect a supported Principled BSDF to Surface.",
                       node, socket, groups)
        else:
            value = getattr(socket, "default_value", None)
            if value is not None and not isinstance(value, (str, bool)):
                values = [value] if isinstance(value, (float, int)) else value
                if any(not math.isfinite(float(item)) for item in values):
                    self.error("nonfinite_value", "Material values must be finite.",
                               node, socket, groups)

    def output(self, node, socket, groups=(), surface=False):
        if getattr(socket, "is_unavailable", False):
            self.error("invalid_output", "The link uses an unavailable node output.", node, groups=groups)
            return
        key = (_identity(node), socket.identifier, tuple(_identity(item) for item in groups), surface)
        if key in self.visiting or len(groups) > 32:
            self.error("cyclic_graph", "The material graph contains a cycle or recursive group.",
                       node, groups=groups)
            return
        if key in self.visited:
            return
        self.visiting.add(key)
        try:
            self.visit(node, socket, groups, surface)
        finally:
            self.visiting.remove(key)
            self.visited.add(key)

    def passthrough(self, node, socket, groups):
        kind = _node_type(node)
        if getattr(node, "mute", False):
            link = next((link for link in node.internal_links
                         if link.to_socket.identifier == socket.identifier), None)
            if link is not None:
                return node, link.from_socket, groups
            self.error("muted_output", "The muted output has no bypass link.", node, groups=groups)
            return False
        if kind == "REROUTE":
            return node, node.inputs[0], groups
        if kind == "GROUP":
            tree = getattr(node, "node_tree", None)
            if tree is None or any(_identity(item.node_tree) == _identity(tree) for item in groups):
                self.error("invalid_group", "The node group is missing or recursive.", node,
                           groups=groups)
                return False
            output = _output(tree, "GROUP_OUTPUT")
            inner = _matching_socket(output.inputs, socket) if output is not None else None
            if inner is None:
                self.error("invalid_group_output", "Cannot resolve the active group output.",
                           node, groups=groups)
                return False
            return output, inner, groups + (node,)
        if kind == "GROUP_INPUT":
            outer = _matching_socket(groups[-1].inputs, socket) if groups else None
            if outer is None:
                self.error("invalid_group_input", "Cannot resolve the group input.", node,
                           groups=groups)
                return False
            return groups[-1], outer, groups[:-1]
        return None

    def constant(self, socket, groups=(), seen=None):
        if socket is None:
            return _UNKNOWN
        if not socket.links:
            return getattr(socket, "default_value", _UNKNOWN)
        if len(socket.links) != 1 or not getattr(socket.links[0], "is_valid", True):
            return _UNKNOWN
        seen = set() if seen is None else seen
        link = socket.links[0]
        node, output = link.from_node, link.from_socket
        key = (_identity(node), output.identifier, tuple(_identity(item) for item in groups))
        if key in seen or len(seen) > 64:
            return _UNKNOWN
        seen = seen | {key}
        bypass = self.passthrough(node, output, groups)
        if bypass is False:
            return _UNKNOWN
        if bypass is not None:
            _, inner, inner_groups = bypass
            return self.constant(inner, inner_groups, seen)
        kind = _node_type(node)
        if kind in {"VALUE", "RGB"}:
            return getattr(output, "default_value", _UNKNOWN)
        if kind == "MATH" and node.operation in _MATH_OPERATIONS:
            inputs = _active_inputs(node)
            count = 3 if node.operation == "MULTIPLY_ADD" else 2
            values = [self.constant(item, groups, seen) for item in inputs[:count]]
            if len(values) < count or not all(isinstance(value, (float, int)) and math.isfinite(value)
                                             for value in values):
                return _UNKNOWN
            a, b = values[:2]
            if node.operation == "DIVIDE" and b == 0:
                return _UNKNOWN
            operations = {
                "ADD": lambda: a + b, "SUBTRACT": lambda: a - b,
                "MULTIPLY": lambda: a * b, "DIVIDE": lambda: a / b,
                "MULTIPLY_ADD": lambda: a * b + values[2],
                "MINIMUM": lambda: min(a, b), "MAXIMUM": lambda: max(a, b),
                "LESS_THAN": lambda: float(a < b), "GREATER_THAN": lambda: float(a > b),
            }
            value = operations[node.operation]()
            return min(max(value, 0.0), 1.0) if getattr(node, "use_clamp", False) else value
        return _UNKNOWN

    def principled(self, node, groups):
        inactive = set()
        for weight, fields in (
            ("Coat Weight", ("Coat Roughness", "Coat IOR", "Coat Tint", "Coat Normal")),
            ("Sheen Weight", ("Sheen Roughness", "Sheen Tint")),
            ("Emission Strength", ("Emission Color",)),
            ("Anisotropic", ("Anisotropic Rotation", "Tangent")),
        ):
            if _is_zero(self.constant(_socket(node.inputs, weight), groups)):
                inactive.update(fields)
        for weight, fields, label in (
            ("Subsurface Weight", ("Subsurface Radius", "Subsurface Scale", "Subsurface IOR",
                                    "Subsurface Anisotropy"), "Subsurface scattering"),
            ("Thin Film Thickness", ("Thin Film IOR",), "Thin film"),
        ):
            socket = _socket(node.inputs, weight)
            if socket is None:
                continue
            if _is_zero(self.constant(socket, groups)):
                inactive.update(fields)
            else:
                self.error("unsupported_feature", label + " is outside the supported subset.",
                           node, socket, groups)
                inactive.update(fields)
        for socket in _active_inputs(node):
            if socket.name in inactive:
                continue
            if socket.name not in _PRINCIPLED_INPUTS:
                self.error("unsupported_input", "Unknown Principled input.", node, socket, groups)
            else:
                self.input(node, socket, groups)

    def visit(self, node, output, groups, surface):
        bypass = self.passthrough(node, output, groups)
        if bypass is False:
            return
        if bypass is not None:
            inner_node, inner, inner_groups = bypass
            self.input(inner_node, inner, inner_groups, surface)
            return
        kind = _node_type(node)
        if kind == "BSDF_PRINCIPLED" and surface:
            self.principled(node, groups)
            return
        if surface:
            self.error("unsupported_surface", "Only a single Principled BSDF is supported.",
                       node, groups=groups)
            return
        inputs = _active_inputs(node)
        reason = None
        if kind in {"VALUE", "RGB"}:
            value = getattr(output, "default_value", None)
            values = [value] if isinstance(value, (float, int)) else value
            if values is None or any(not math.isfinite(float(item)) for item in values):
                reason = "Constant values must be finite."
        elif kind == "TEX_IMAGE":
            image = getattr(node, "image", None)
            if image is None:
                reason = "The Image Texture has no image."
            elif getattr(image, "source", "FILE") not in {"FILE", "GENERATED"}:
                reason = "Only static images are supported; sequences, movies and UDIMs are not."
            elif image.colorspace_settings.name not in {"sRGB", "Linear Rec.709", "Non-Color", "Raw"}:
                reason = "The image color space cannot be preserved by the supported exporter profile."
            elif node.projection != "FLAT":
                reason = "Only flat image projection is supported."
            elif node.interpolation not in {"Linear", "Closest"}:
                reason = "Only Linear and Closest image filtering are supported."
            elif node.extension not in {"REPEAT", "EXTEND", "CLIP", "MIRROR"}:
                reason = "Unsupported image extension mode."
        elif kind == "TEX_COORD":
            if output.name != "UV":
                reason = "Only UV coordinates are supported; Generated is exported incorrectly as UV."
            elif getattr(node, "from_instancer", False):
                reason = "Texture Coordinate From Instancer is not preserved by the exporter."
        elif kind == "UVMAP":
            if getattr(node, "from_instancer", False):
                reason = "UV Map From Instancer is not preserved by the exporter."
            elif getattr(node, "uv_map", ""):
                if self.active_uv_names is None:
                    reason = "Named UV maps require evaluated mesh bindings for validation."
                elif self.active_uv_names != {node.uv_map}:
                    reason = ("Named UV map " + repr(node.uv_map) +
                              " must be the active and render UV map on every bound mesh.")
        elif kind == "MAPPING":
            if node.vector_type not in {"POINT", "TEXTURE", "VECTOR", "NORMAL"}:
                reason = "Unsupported Mapping mode."
            for name in ("Location", "Rotation", "Scale"):
                socket = _socket(node.inputs, name)
                if socket is not None and self.constant(socket, groups) is _UNKNOWN:
                    self.error("dynamic_mapping", "Mapping transforms must be constant.",
                               node, socket, groups)
        elif kind == "NORMAL_MAP":
            if node.space != "TANGENT":
                reason = "Only tangent-space normal maps are preserved by the exporter."
            elif getattr(node, "convention", "OPENGL") != "OPENGL":
                reason = "Only the OpenGL normal-map convention is supported."
            elif getattr(node, "uv_map", ""):
                reason = "The exporter does not preserve a Normal Map's named tangent UV map."
        elif kind == "TANGENT":
            if node.direction_type != "UV_MAP" or getattr(node, "uv_map", ""):
                reason = "Only the default UV tangent is preserved by the exporter."
        elif kind == "MATH":
            if node.operation not in _MATH_OPERATIONS:
                reason = "Unsupported Math operation: " + node.operation + "."
            inputs = inputs[:3 if node.operation == "MULTIPLY_ADD" else 2]
        elif kind == "MIX":
            if node.data_type == "FLOAT":
                indices = (0, 2, 3)
            elif node.data_type == "VECTOR":
                indices = (0 if node.factor_mode == "UNIFORM" else 1, 4, 5)
            elif node.data_type == "RGBA":
                indices = (0, 6, 7)
                if node.blend_type != "MIX":
                    reason = "Only ordinary Mix is preserved; other blend modes are unsupported."
            else:
                indices = ()
                reason = "Unsupported Mix data type: " + node.data_type + "."
            inputs = [node.inputs[index] for index in indices]
        elif kind == "MIX_RGB":
            if node.blend_type != "MIX":
                reason = "Only ordinary Mix is supported."
        elif kind == "MAP_RANGE":
            if node.interpolation_type != "LINEAR" or node.data_type not in {"FLOAT", "FLOAT_VECTOR"}:
                reason = "Only linear scalar/vector Map Range is supported."
            indices = (0, 1, 2, 3, 4) if node.data_type == "FLOAT" else (6, 7, 8, 9, 10)
            inputs = [node.inputs[index] for index in indices]
        elif kind == "CLAMP":
            if node.clamp_type != "MINMAX":
                reason = "Only Min Max Clamp is supported."
        elif kind in {"SEPARATE_COLOR", "COMBINE_COLOR"}:
            if node.mode != "RGB":
                reason = "Only RGB color separation/combination is supported."
        elif kind in {"SEPXYZ", "COMBXYZ", "SEPRGB", "COMBRGB"}:
            pass
        elif kind == "VALTORGB":
            reason = "ColorRamp is not exported by Blender 5.2.2; use supported textures instead."
        else:
            reason = "Unsupported node type: " + kind + "."
        if reason:
            self.error("unsupported_node", reason, node, groups=groups)
            return
        for socket in inputs:
            self.input(node, socket, groups)

    def run(self):
        tree = getattr(self.material, "node_tree", None)
        output = _output(tree, "OUTPUT_MATERIAL") if tree is not None else None
        if output is None:
            self.error("missing_output", "The material has no unambiguous active Material Output.")
        else:
            self.input(output, _socket(output.inputs, "Surface"), surface=True)
            for name in ("Volume", "Displacement"):
                socket = _socket(output.inputs, name)
                if socket is not None and socket.links and not _is_zero(self.constant(socket)):
                    self.error("unsupported_output", name + " is outside the supported subset.",
                               output, socket)
        return MaterialReport(self.name, int(self.material.as_pointer()), tuple(self.diagnostics))


def validate_material(material, *, active_uv_names=None):
    """Validate a material; named UV maps require the names from all bound meshes."""
    return _Validator(material, active_uv_names).run()


def _active_uv_names(mesh):
    layers = getattr(mesh, "uv_layers", ())
    active = getattr(layers, "active", None)
    names = {getattr(active, "name", "")}
    names.update(layer.name for layer in layers if getattr(layer, "active_render", False))
    return names


def validate_depsgraph(depsgraph):
    """Return reports keyed by evaluated pointers, valid only for this depsgraph state."""
    materials = {}
    bindings = {}
    for instance in depsgraph.object_instances:
        obj = instance.object
        slots = getattr(obj, "material_slots", ())
        polygons = getattr(getattr(obj, "data", None), "polygons", None)
        indices = range(len(slots)) if polygons is None else {
            min(max(face.material_index, 0), len(slots) - 1) for face in polygons
        }
        for index in indices:
            if not slots:
                continue
            material = slots[index].material
            if material is None:
                continue
            material = material.evaluated_get(depsgraph)
            pointer = int(material.as_pointer())
            materials[pointer] = material
            bindings.setdefault(pointer, set()).update(_active_uv_names(getattr(obj, "data", None)))
    return {pointer: validate_material(material, active_uv_names=bindings[pointer])
            for pointer, material in materials.items()}
