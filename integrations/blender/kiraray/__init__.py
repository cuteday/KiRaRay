"""KiRaRay Blender extension; preflight remains importable without Blender."""

bl_info = {"name": "KiRaRay", "author": "KiRaRay contributors", "version": (0, 1, 0),
           "blender": (5, 2, 0), "location": "Render Properties", "category": "Render"}


def register():
    from .engine import register as register_engine
    register_engine()


def unregister():
    from .engine import unregister as unregister_engine
    unregister_engine()
