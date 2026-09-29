"""Build a private OpenUSD SDK without Python bindings for standalone KiRaRay."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request
import zipfile


SOURCES = {
    "tbb": ("2022.3.0", "https://github.com/uxlfoundation/oneTBB/archive/refs/tags/v2022.3.0.zip", "oneTBB-2022.3.0"),
    "materialx": ("1.39.4", "https://github.com/AcademySoftwareFoundation/MaterialX/archive/refs/tags/v1.39.4.zip", "MaterialX-1.39.4"),
    "usd": ("26.03", "https://github.com/PixarAnimationStudios/OpenUSD/archive/refs/tags/v26.03.zip", "OpenUSD-26.03"),
}
CHECKSUMS = {
    "tbb": "4f47379064f99cc50da8dde85e27651d3609ac6c3e0941b1c728a1b2dd1e4b68",
    "materialx": "3910b2498d7da6c8e5ac150b9e6a74e530626044309edb077829c024ed691e0e",
    "usd": "bdb4a051e3453a557fecacaf9d1731c79c6f5a1eaf8e4ffc707bebca580704a9",
}


def source(directory, name):
    version, url, folder = SOURCES[name]
    archive = directory / (folder + ".zip")
    if not archive.is_file():
        temporary = archive.with_suffix(".download")
        print(f"Downloading {name} {version}", flush=True)
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        temporary.replace(archive)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != CHECKSUMS[name]:
        raise RuntimeError(f"Source checksum mismatch: {archive}")
    root = directory / folder
    marker = directory / (folder + ".extracted")
    if not marker.is_file() or marker.read_text(encoding="utf-8").strip() != digest:
        with zipfile.ZipFile(archive) as package:
            for item in package.infolist():
                (directory / item.filename).resolve().relative_to(directory.resolve())
            package.extractall(directory)
        marker.write_text(digest + "\n", encoding="utf-8")
    return root, {"version": version, "url": url, "sha256": digest}


def command(arguments, log):
    print(subprocess.list2cmdline([str(arg) for arg in arguments]), flush=True)
    with log.open("w", encoding="utf-8") as output:
        result = subprocess.run(arguments, stdout=output, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Command exited with {result.returncode}; see {log}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("build/usd-sdk/standalone"))
    parser.add_argument("--cmake", default="cmake")
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--generator", default="Ninja")
    args = parser.parse_args()
    if args.parallel < 1:
        parser.error("--parallel must be positive")
    root = args.output_dir.resolve()
    downloads, install = root / "sources", root / "install"
    downloads.mkdir(parents=True, exist_ok=True)
    common = ["-G", args.generator, "-DCMAKE_BUILD_TYPE=Release",
              f"-DCMAKE_INSTALL_PREFIX={install}", f"-DCMAKE_PREFIX_PATH={install}",
              "-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL"]
    options = {
        "tbb": ["-DTBB_TEST=OFF", "-DTBB_STRICT=OFF", "-DTBBMALLOC_BUILD=OFF"],
        "materialx": ["-DMATERIALX_BUILD_SHARED_LIBS=ON", "-DMATERIALX_BUILD_TESTS=OFF",
                      "-DMATERIALX_BUILD_RENDER=OFF", "-DMATERIALX_BUILD_PYTHON=OFF",
                      "-DMATERIALX_BUILD_GEN_GLSL=ON", "-DMATERIALX_BUILD_GEN_OSL=OFF",
                      "-DMATERIALX_BUILD_GEN_MDL=OFF", "-DMATERIALX_BUILD_GEN_MSL=OFF"],
        "usd": ["-DPXR_ENABLE_PYTHON_SUPPORT=OFF", "-DPXR_ENABLE_MATERIALX_SUPPORT=ON",
                "-DPXR_BUILD_IMAGING=OFF", "-DPXR_BUILD_USD_IMAGING=OFF",
                "-DPXR_ENABLE_OPENVDB_SUPPORT=OFF", "-DPXR_ENABLE_GL_SUPPORT=OFF",
                "-DPXR_BUILD_TESTS=OFF", "-DPXR_BUILD_EXAMPLES=OFF",
                "-DPXR_BUILD_TUTORIALS=OFF", "-DPXR_BUILD_USD_TOOLS=OFF",
                "-DPXR_BUILD_DOCUMENTATION=OFF", "-DPXR_BUILD_MONOLITHIC=ON",
                "-DBUILD_SHARED_LIBS=ON", "-DPXR_ENABLE_PRECOMPILED_HEADERS=OFF"],
    }
    manifest = {"profile": "standalone", "python_support": False,
                "generator": args.generator, "configuration": "Release", "sources": {}}
    for name in SOURCES:
        tree, metadata = source(downloads, name)
        manifest["sources"][name] = metadata
        build = root / (name + "-build-" + args.generator.replace(" ", "-"))
        command([args.cmake, "-S", str(tree), "-B", str(build), *common, *options[name]],
                root / (name + "-configure.log"))
        command([args.cmake, "--build", str(build), "--config", "Release", "--target",
                 "install", "--parallel", str(args.parallel)], root / (name + "-build.log"))
    (install / "krr-sdk.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Standalone SDK: {install}", flush=True)


if __name__ == "__main__":
    main()
