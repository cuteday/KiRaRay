"""Package the built Blender extension without host SDK DLLs or Python caches."""

import argparse
from pathlib import Path
import shutil
import zipfile


def stage_licenses(destination):
    root = Path(__file__).resolve().parents[2]
    notices = {
        "KiRaRay-LICENSE.txt": root / "LICENSE",
        "OpenPBR-LICENSE.txt": root / "src/render/materials/openpbr/LICENSE",
        "OpenPBR-NOTICE.txt": root / "src/render/materials/openpbr/NOTICE",
        "MaterialX-LICENSE.txt": Path(__file__).parent / "licenses/MaterialX-LICENSE.txt",
        "OpenEXR-LICENSE.txt": Path(__file__).parent / "licenses/OpenEXR-LICENSE.txt",
        "Imath-LICENSE.txt": Path(__file__).parent / "licenses/Imath-LICENSE.txt",
        "libdeflate-LICENSE.txt": Path(__file__).parent / "licenses/libdeflate-LICENSE.txt",
        "OpenJPH-LICENSE.txt": Path(__file__).parent / "licenses/OpenJPH-LICENSE.txt",
    }
    destination.mkdir(parents=True, exist_ok=True)
    for name, source in notices.items():
        shutil.copy2(source, destination / name)
    header = (root / "src/ext/mikktspace/mikktspace.h").read_text()
    license_start = header.index("/**", header.index("*/") + 2)
    license_end = header.index("*/", license_start) + 2
    (destination / "MikkTSpace-LICENSE.txt").write_text(header[license_start:license_end] + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--libraries", type=Path)
    parser.add_argument("--stage-only", action="store_true")
    args = parser.parse_args()
    stage_licenses(args.source / "native/licenses")
    if args.libraries:
        definitions = list(args.libraries.rglob("*.mtlx"))
        if not definitions:
            raise FileNotFoundError(f"MaterialX definitions are missing: {args.libraries}")
        for source in definitions:
            target = args.source / "native/libraries" / source.relative_to(args.libraries)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    if args.stage_only:
        return
    if not args.output:
        parser.error("--output is required unless --stage-only is used")
    for name in ("blender_manifest.toml", "__init__.py", "engine.py", "preflight.py",
                 "native/hdKiRaRay.dll", "native/plugInfo.json", "native/dxcompiler.dll"):
        if not (args.source / name).is_file():
            raise FileNotFoundError(args.source / name)
    allowed_dlls = {"hdKiRaRay.dll", "dxcompiler.dll", "dxil.dll"}
    temporary = args.output.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
        for source in sorted(args.source.rglob("*")):
            if not source.is_file() or "__pycache__" in source.parts:
                continue
            if source.suffix.lower() == ".dll" and source.name not in allowed_dlls:
                raise ValueError(f"Private SDK DLL must not be packaged: {source}")
            if source.suffix.lower() in {".py", ".toml", ".json", ".dll", ".mtlx", ".txt"}:
                archive.write(source, source.relative_to(args.source).as_posix())
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
