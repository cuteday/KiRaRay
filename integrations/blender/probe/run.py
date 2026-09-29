"""Run inside the pinned Blender host; validate native Hydra AOV readback."""

import argparse
import json
from pathlib import Path
import struct
import sys


def read_channels(path):
    """Read this probe's uncompressed, single-part float32 EXR channels."""
    def string(stream):
        value = bytearray()
        while (byte := stream.read(1)) != b"\0":
            if not byte:
                raise ValueError("Truncated EXR string")
            value.extend(byte)
        return value.decode()

    with path.open("rb") as stream:
        assert struct.unpack("<I", stream.read(4))[0] == 20000630
        version, = struct.unpack("<I", stream.read(4))
        assert version & ~1024 == 2, version
        attributes = {}
        while name := string(stream):
            kind = string(stream)
            length, = struct.unpack("<I", stream.read(4))
            attributes[name] = (kind, stream.read(length))
        assert attributes["compression"][1] == b"\0"
        import io
        channel_data = io.BytesIO(attributes["channels"][1])
        names = []
        while name := string(channel_data):
            pixel_type, _, x_sampling, y_sampling = struct.unpack("<iB3xii", channel_data.read(16))
            assert (pixel_type, x_sampling, y_sampling) == (2, 1, 1)
            names.append(name)
        x0, y0, x1, y1 = struct.unpack("<4i", attributes["dataWindow"][1])
        width, height = x1 - x0 + 1, y1 - y0 + 1
        offsets = struct.unpack(f"<{height}Q", stream.read(height * 8))
        channels = {name: [0.0] * (width * height) for name in names}
        for offset in offsets:
            stream.seek(offset)
            y, size = struct.unpack("<ii", stream.read(8))
            assert size == width * len(names) * 4
            for name in names:
                channels[name][(y - y0) * width:(y - y0 + 1) * width] = struct.unpack(
                    f"<{width}f", stream.read(width * 4))
        return channels


def main():
    import bpy
    parser = argparse.ArgumentParser()
    parser.add_argument("--plugin-dir", required=True, type=Path)
    parser.add_argument("--artifacts", required=True, type=Path)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    args.artifacts = args.artifacts.resolve()
    args.artifacts.mkdir(parents=True, exist_ok=True)
    bpy.utils.expose_bundled_modules()
    from pxr import Plug, Usd

    plugins = Plug.Registry().RegisterPlugins(str(args.plugin_dir.resolve()))
    assert any(plugin.name == "hdKrrProbe" for plugin in plugins), "Probe plugin was not discovered"

    class ProbeEngine(bpy.types.HydraRenderEngine):
        bl_idname = "KRR_HYDRA_PROBE"
        bl_label = "KiRaRay ABI probe"
        bl_delegate_id = "HdKrrProbeRendererPlugin"
        bl_use_gpu_context = False
        bl_use_materialx = False

        def get_render_settings(self, engine_type):
            return {"aovToken:Combined": "color", "aovToken:Depth": "depth"}

        def update_render_passes(self, scene, render_layer):
            self.register_pass(scene, render_layer, "Combined", 4, "RGBA", "COLOR")
            self.register_pass(scene, render_layer, "Depth", 1, "Z", "VALUE")

    bpy.utils.register_class(ProbeEngine)
    scene = bpy.context.scene
    previous_engine = scene.render.engine
    for obj in list(scene.objects):
        if obj.type != "CAMERA":
            bpy.data.objects.remove(obj, do_unlink=True)
    scene.world = None
    scene.render.engine = ProbeEngine.bl_idname
    scene.render.resolution_x = 8
    scene.render.resolution_y = 6
    scene.render.resolution_percentage = 100
    scene.render.image_settings.media_type = "MULTI_LAYER_IMAGE"
    scene.render.image_settings.file_format = "OPEN_EXR_MULTILAYER"
    scene.render.image_settings.exr_codec = "NONE"
    scene.render.image_settings.use_exr_interleave = True
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "32"
    scene.view_settings.view_transform = "Standard"
    scene.view_layers[0].use_pass_z = True

    results = []
    for index in range(3):
        bpy.ops.render.render()
        path = args.artifacts / f"probe-{index}.exr"
        bpy.data.images["Render Result"].save_render(str(path), scene=scene)
        channels = read_channels(path)
        color = [next(values for name, values in channels.items() if name.endswith(f".Combined.{component}"))
                 for component in "RGBA"]
        assert all(len(values) == 48 for values in color)
        maximum_error = 0.0
        for y in range(6):
            for x in range(8):
                expected = (x / 7, (5 - y) / 5, 0.25, 1)
                offset = y * 8 + x
                maximum_error = max(maximum_error, *(abs(color[i][offset] - expected[i]) for i in range(4)))
        assert maximum_error < 1e-6, maximum_error
        depth = next(values for name, values in channels.items() if name.endswith(".Depth.Z"))
        depth_error = max(abs(value - 0.5) for value in depth)
        assert depth_error < 1e-6, depth_error
        results.append({"file": path.name, "maximum_error": maximum_error, "depth_error": depth_error})
    scene.render.engine = previous_engine
    bpy.utils.unregister_class(ProbeEngine)
    report = {"blender": bpy.app.version_string, "blender_revision": bpy.app.build_hash.decode(),
              "python": sys.version, "usd": Usd.GetVersion(), "renders": results,
              "color": "float32 RGBA, Blender bottom-to-top rows; gradient verified",
              "depth": "float32 Depth.Z read back from Blender's saved multilayer EXR"}
    (args.artifacts / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print("KRR_HYDRA_PROBE_SUCCESS", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
