"""Fetch the release libraries after the documented sparse SDK checkout."""

import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sdk", type=Path)
    args = parser.parse_args()
    root = args.sdk.resolve()
    directories = ("usd/lib", "tbb/lib", "tbb/bin", "MaterialX/lib", "MaterialX/bin",
                   "imath/lib", "imath/bin", "python/313/libs")
    pending = []
    for directory in directories:
        for path in (root / directory).rglob("*"):
            if not path.is_file() or path.stat().st_size > 200:
                continue
            if path.suffix not in (".lib", ".dll") or "_d." in path.name or "_debug." in path.name:
                continue
            if directory == "usd/lib" and path.name not in ("usd_ms.lib", "usd_ms.dll"):
                continue
            pointer = path.read_text()
            if pointer.startswith("version https://git-lfs.github.com/spec/v1"):
                lines = pointer.splitlines()
                pending.append((path, lines[1].split(":")[1], int(lines[2].split()[1])))
    if not pending:
        print("Selected release libraries are already materialized.")
        return
    body = json.dumps({"operation": "download", "transfers": ["basic"],
                       "objects": [{"oid": oid, "size": size} for _, oid, size in pending]}).encode()
    request = urllib.request.Request(
        "https://projects.blender.org/blender/lib-windows_x64.git/info/lfs/objects/batch",
        data=body, headers={"Content-Type": "application/json", "Accept": "application/vnd.git-lfs+json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        objects = {obj["oid"]: obj for obj in json.load(response)["objects"]}

    def download(item):
        path, oid, size = item
        action = objects[oid]["actions"]["download"]
        request = urllib.request.Request(action["href"], headers=action.get("header", {}))
        temporary = path.with_name(path.name + ".download")
        digest = hashlib.sha256()
        with urllib.request.urlopen(request, timeout=120) as source, temporary.open("wb") as target:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
                target.write(chunk)
        if temporary.stat().st_size != size or digest.hexdigest() != oid:
            raise RuntimeError(f"Checksum mismatch: {path}")
        temporary.replace(path)
        print(f"Verified {path.relative_to(root)} ({size} bytes)", flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(download, pending))


if __name__ == "__main__":
    main()
