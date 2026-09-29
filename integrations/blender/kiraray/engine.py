"""KiRaRay's Blender 5.2 Hydra integration."""

import json
import os
import re
from pathlib import Path
import tempfile

import bpy
from bpy_extras.io_utils import ExportHelper

from .preflight import validate_depsgraph

_warnings = {}
_last_status = {}
_dll_directory = None


def _validate_color_space():
    color = bpy.data.colorspace
    if color.is_missing_opencolorio_config or color.working_space_interop_id != "lin_rec709_scene":
        raise RuntimeError("KiRaRay requires the blend file working space Linear Rec.709; "
                           f"current working space is {color.working_space}. Display transforms such as AgX are supported.")


def _diagnostics(depsgraph):
    reports = validate_depsgraph(depsgraph)
    result = {}
    messages = []
    for pointer, report in reports.items():
        if report.supported:
            continue
        details = [f"{report.name}: {item.source}: {item.message}" for item in report.diagnostics]
        result[f"/scene/M_{pointer:016X}"] = details
        messages.extend(details)
    return reports, result, messages


def _scene_warnings(depsgraph):
    messages = []
    for obj in depsgraph.objects:
        if obj.type == "LIGHT":
            light = obj.data
            if light.type == "AREA" and light.shape == "ELLIPSE":
                messages.append(f"{obj.name}: Blender's Hydra bridge approximates ellipse lights as disks")
            if light.diffuse_factor != 1 or light.specular_factor != 1:
                messages.append(f"{obj.name}: separate diffuse/specular light factors are unsupported")
        elif obj.type in {"CURVE", "SURFACE", "FONT", "VOLUME", "CURVES", "META"}:
            messages.append(f"{obj.name}: {obj.type} geometry is outside the supported mesh profile; convert to a mesh")
    return messages


def _settings_changed(settings, context):
    settings.id_data.update_tag()
    for window in context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


class KiRaRaySettings(bpy.types.PropertyGroup):
    samples: bpy.props.IntProperty(name="Render Samples", default=128, min=1, max=1048576)
    viewport_samples: bpy.props.IntProperty(name="Viewport Samples", default=32, min=1, max=1048576)
    seed: bpy.props.IntProperty(name="Seed", default=0, min=0)
    max_depth: bpy.props.IntProperty(name="Maximum Path Depth", default=10, min=0, soft_max=64,
        description="Maximum scattering depth; zero shows only directly visible emission and the environment",
        update=_settings_changed)
    nee: bpy.props.BoolProperty(name="Next Event Estimation", default=True,
        description="Sample lights explicitly at each scattering event to reduce noise",
        update=_settings_changed)
    rr: bpy.props.FloatProperty(name="Russian Roulette Survival", default=0.8, min=0.001, max=1.0,
        precision=3, description="Path survival probability at each scattering event; one disables Russian roulette",
        update=_settings_changed)
    graphics_api: bpy.props.EnumProperty(name="Graphics API", items=(
        ("vulkan", "Vulkan", "Vulkan offscreen interop"),
        ("d3d12", "D3D12", "Direct3D 12 offscreen interop")), default="vulkan")
    asset_root: bpy.props.StringProperty(name="Asset Root", subtype="DIR_PATH")


class KiRaRayEngine(bpy.types.HydraRenderEngine):
    bl_idname = "KIRARAY"
    bl_label = "KiRaRay"
    bl_delegate_id = "HdKiRaRayRendererPlugin"
    bl_use_materialx = True
    bl_use_gpu_context = False
    bl_use_preview = False

    @classmethod
    def register(cls):
        global _dll_directory
        if bpy.app.version[:2] != (5, 2):
            raise RuntimeError("This KiRaRay build requires Blender 5.2 LTS")
        bpy.utils.expose_bundled_modules()
        from pxr import Plug, Usd
        if Usd.GetVersion() != (0, 26, 3):
            raise RuntimeError("KiRaRay requires Blender's matching USD 26.03 build")
        plugin = Path(os.environ.get("KRR_HYDRA_PLUGIN_DIR", Path(__file__).parent / "native"))
        if not (plugin / "plugInfo.json").is_file():
            raise RuntimeError(f"KiRaRay native plugin is missing: {plugin}")
        _dll_directory = os.add_dll_directory(str(plugin.resolve()))
        Plug.Registry().RegisterPlugins(str(plugin.resolve()))

    def _prepare(self, depsgraph):
        _validate_color_space()
        self._scene = depsgraph.scene
        reports, self._invalid, messages = _diagnostics(depsgraph)
        self._opaque_mix_branches = {
            f"/scene/M_{pointer:016X}": report.opaque_mix_branch
            for pointer, report in reports.items() if report.supported and report.opaque_mix_branch}
        messages.extend(_scene_warnings(depsgraph))
        previous = _warnings.get(self._scene.name_full, [])
        if messages != previous:
            for message in messages:
                self.report({"WARNING"}, message)
        _warnings[self._scene.name_full] = messages
        self._producer_warnings = messages
        if not hasattr(self, "_status_path"):
            descriptor, path = tempfile.mkstemp(prefix=f"kiraray-{os.getpid()}-", suffix=".json")
            os.close(descriptor)
            self._status_path = path
        Path(self._status_path).write_text("{}", encoding="utf-8")

    def get_render_settings(self, engine_type):
        scene = getattr(self, "_scene", bpy.context.scene)
        settings = scene.kiraray
        root = bpy.path.abspath(settings.asset_root) if settings.asset_root else (
            str(Path(bpy.data.filepath).parent) if bpy.data.filepath else str(Path.cwd()))
        return {
            "krr:paused": False,
            "krr:samples": settings.viewport_samples if engine_type == "VIEWPORT" else settings.samples,
            "krr:seed": settings.seed,
            "krr:maxDepth": settings.max_depth,
            "krr:nee": settings.nee,
            "krr:rr": settings.rr,
            "krr:priority": 2 if engine_type == "FINAL" else 1,
            "krr:assetRoot": root,
            "krr:graphicsApi": settings.graphics_api,
            "krr:blenderScene": True,
            "krr:emissionLuminanceScale": 1000.0,
            "krr:diagnostics": json.dumps(getattr(self, "_invalid", {})),
            "krr:opaqueMixBranches": json.dumps(getattr(self, "_opaque_mix_branches", {})),
            "krr:statusPath": getattr(self, "_status_path", ""),
            "aovToken:Combined": "color",
            "aovToken:Depth": "linearDepth",
        }

    def update(self, data, depsgraph):
        self._prepare(depsgraph)
        if depsgraph.scene.camera and depsgraph.scene.camera.data.type != "PERSP":
            raise RuntimeError("KiRaRay currently requires a perspective camera")
        super().update(data, depsgraph)

    def view_update(self, context, depsgraph):
        try:
            self._prepare(depsgraph)
            self._viewport_error = ""
        except RuntimeError as error:
            self._viewport_error = str(error)
            if getattr(self, "engine_ptr", None):
                import _bpy_hydra
                _bpy_hydra.engine_set_render_setting(self.engine_ptr, "krr:paused", True)
            self.report({"ERROR"}, self._viewport_error)
            return
        super().view_update(context, depsgraph)

    def render(self, depsgraph):
        try:
            super().render(depsgraph)
        finally:
            if getattr(self, "engine_ptr", None):
                import _bpy_hydra
                _bpy_hydra.engine_set_render_setting(self.engine_ptr, "krr:paused", True)
                _bpy_hydra.engine_free(self.engine_ptr)
                self.engine_ptr = None
        status = json.loads(Path(self._status_path).read_text(encoding="utf-8"))
        self._show_diagnostics(status)
        if status.get("error"):
            self.report({"ERROR"}, status["error"])
            raise RuntimeError(status["error"])

    def view_draw(self, context, depsgraph):
        if getattr(self, "_viewport_error", ""):
            self.update_stats("KiRaRay", self._viewport_error)
            return
        view = context.region_data.view_perspective if context.region_data else "PERSP"
        unsupported_camera = view == "CAMERA" and context.scene.camera and context.scene.camera.data.type != "PERSP"
        if view == "ORTHO" or unsupported_camera:
            if getattr(self, "engine_ptr", None):
                import _bpy_hydra
                _bpy_hydra.engine_set_render_setting(self.engine_ptr, "krr:paused", True)
            self.update_stats("KiRaRay", "Orthographic preview is not supported; use perspective or a perspective camera")
            return
        if getattr(self, "engine_ptr", None):
            import _bpy_hydra
            _bpy_hydra.engine_set_render_setting(self.engine_ptr, "krr:paused", False)
        super().view_draw(context, depsgraph)
        if hasattr(self, "_status_path"):
            try:
                status = json.loads(Path(self._status_path).read_text(encoding="utf-8") or "{}")
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                return
            if status.get("error"):
                self.update_stats("KiRaRay", status["error"])
            self._show_diagnostics(status)

    def _show_diagnostics(self, status):
        _last_status[self._scene.name_full] = status
        messages = list(dict.fromkeys(getattr(self, "_producer_warnings", []) + status.get("diagnostics", [])))
        previous = _warnings.get(self._scene.name_full, [])
        for message in messages:
            if message not in previous:
                self.report({"WARNING"}, message)
        _warnings[self._scene.name_full] = messages

    def update_render_passes(self, scene, render_layer):
        self.register_pass(scene, render_layer, "Combined", 4, "RGBA", "COLOR")
        if render_layer.use_pass_z:
            self.register_pass(scene, render_layer, "Depth", 1, "Z", "VALUE")

    def __del__(self):
        attributes = object.__getattribute__(self, "__dict__")
        pointer = attributes.get("engine_ptr")
        if pointer:
            import _bpy_hydra
            _bpy_hydra.engine_free(pointer)
            attributes["engine_ptr"] = None
        path = attributes.get("_status_path")
        if path:
            Path(path).unlink(missing_ok=True)


class KiRaRayRenderPanel(bpy.types.Panel):
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "render"

    @classmethod
    def poll(cls, context):
        return context.scene.render.engine == "KIRARAY"


class KiRaRayPanel(KiRaRayRenderPanel):
    bl_label = "KiRaRay"
    bl_idname = "KIRARAY_PT_settings"

    def draw(self, context):
        messages = _warnings.get(context.scene.name_full, [])
        if messages:
            box = self.layout.box()
            box.label(text="Unsupported materials use magenta", icon="ERROR")
            for message in messages[:8]:
                box.label(text=message)
        self.layout.operator("kiraray.export_usd")


class KiRaRaySamplingPanel(KiRaRayRenderPanel):
    bl_label = "Sampling"
    bl_idname = "KIRARAY_PT_sampling"
    bl_parent_id = "KIRARAY_PT_settings"

    def draw(self, context):
        for name in ("samples", "viewport_samples", "seed"):
            self.layout.prop(context.scene.kiraray, name)


class KiRaRayLightPathsPanel(KiRaRayRenderPanel):
    bl_label = "Light Paths"
    bl_idname = "KIRARAY_PT_light_paths"
    bl_parent_id = "KIRARAY_PT_settings"

    def draw(self, context):
        self.layout.prop(context.scene.kiraray, "max_depth")


class KiRaRayAdvancedPanel(KiRaRayRenderPanel):
    bl_label = "Advanced"
    bl_idname = "KIRARAY_PT_light_paths_advanced"
    bl_parent_id = "KIRARAY_PT_light_paths"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        for name in ("nee", "rr"):
            self.layout.prop(context.scene.kiraray, name)


class KiRaRaySystemPanel(KiRaRayRenderPanel):
    bl_label = "System"
    bl_idname = "KIRARAY_PT_system"
    bl_parent_id = "KIRARAY_PT_settings"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        for name in ("graphics_api", "asset_root"):
            self.layout.prop(context.scene.kiraray, name)


class KiRaRayExportUSD(bpy.types.Operator, ExportHelper):
    bl_idname = "kiraray.export_usd"
    bl_label = "Export Validated USD"
    filename_ext = ".usdc"
    filter_glob: bpy.props.StringProperty(default="*.usd;*.usda;*.usdc", options={"HIDDEN"})

    def execute(self, context):
        try:
            _validate_color_space()
        except RuntimeError as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        reports, _, messages = _diagnostics(context.evaluated_depsgraph_get())
        result = bpy.ops.wm.usd_export(filepath=self.filepath, export_materials=True,
                                      generate_materialx_network=True, use_instancing=True,
                                      convert_world_material=True)
        if "FINISHED" not in result:
            return {"CANCELLED"}
        bpy.utils.expose_bundled_modules()
        from pxr import Tf, Usd, UsdShade, Vt
        by_name = {}
        by_source_name = {}
        for report in reports.values():
            by_name.setdefault(Tf.MakeValidIdentifier(report.name), []).append(report)
            by_source_name.setdefault(report.name, []).append(report)
        stage = Usd.Stage.Open(self.filepath)
        for prim in stage.Traverse():
            if not prim.IsA(UsdShade.Material):
                continue
            prim.SetCustomDataByKey("kiraray:emissionLuminanceScale", 1000.0)
            name = prim.GetName()
            source = prim.GetAttribute("userProperties:blender:data_name")
            candidates = list(by_source_name.get(source.Get() if source else "", []))
            for report in by_name.get(name, []):
                if report not in candidates:
                    candidates.append(report)
            for base, reports_for_name in by_name.items():
                if len(reports_for_name) > 1 and re.fullmatch(re.escape(base) + r"_\d+", name):
                    candidates.extend(report for report in reports_for_name if report not in candidates)
            details = [f"{report.name}: {item.source}: {item.message}"
                       for report in candidates for item in report.diagnostics]
            if len(candidates) > 1:
                details.append("Ambiguous source material names in validated export")
            if not candidates:
                details.append("Exported material could not be matched to a validated Blender material")
            if details:
                prim.SetCustomDataByKey("kiraray:diagnostics", Vt.StringArray(details))
            elif candidates[0].opaque_mix_branch:
                prim.SetCustomDataByKey("kiraray:opaqueMixBranch", candidates[0].opaque_mix_branch)
        data = dict(stage.GetRootLayer().customLayerData)
        data["kiraray:producer"] = "Blender " + bpy.app.version_string
        stage.GetRootLayer().customLayerData = data
        stage.GetRootLayer().Save()
        for message in messages:
            self.report({"WARNING"}, message)
        self.report({"INFO"}, "Exported USD with KiRaRay material diagnostics")
        return {"FINISHED"}


_classes = (KiRaRaySettings, KiRaRayEngine, KiRaRayPanel, KiRaRaySamplingPanel,
            KiRaRayLightPathsPanel, KiRaRayAdvancedPanel, KiRaRaySystemPanel, KiRaRayExportUSD)


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.kiraray = bpy.props.PointerProperty(type=KiRaRaySettings)


def unregister():
    del bpy.types.Scene.kiraray
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
