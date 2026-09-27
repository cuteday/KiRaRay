# Blender host compatibility probe

This synthetic delegate verifies native plugin loading and color/depth AOV readback before linking KiRaRay. It does not render scene geometry. The probe is independent of the normal KiRaRay build and its Python interpreter.

Validated host: Blender **5.2.2 LTS**, revision `d13f752e3b9c`, Windows x64, Python 3.13.13, USD 26.03. Use the matching official [Windows development libraries](https://projects.blender.org/blender/lib-windows_x64/src/tag/v5.2.2), commit `60d6e96b917568278d400a4024c98da0fb777338`. Vanilla USD does not have Blender's namespace or build ABI.

Download [portable Blender 5.2.2](https://download.blender.org/release/Blender5.2/blender-5.2.2-windows-x64.zip) beneath `build/blender-5.2/` and extract there. Its SHA256 is `3849d17a682cba006075aaa3f3597ecb5c9c30ec31035b2e092c53e40679b535`, as recorded in the [official checksum list](https://download.blender.org/release/Blender5.2/blender-5.2.2.sha256).

Acquire only the relevant SDK directories, from the repository root in PowerShell:

```powershell
$env:GIT_LFS_SKIP_SMUDGE = '1'
git clone --filter=blob:none --depth 1 --branch v5.2.2 --no-checkout https://projects.blender.org/blender/lib-windows_x64.git build/blender-5.2/sdk
git -C build/blender-5.2/sdk sparse-checkout init --cone
git -C build/blender-5.2/sdk sparse-checkout set usd/include usd/cmake usd/lib tbb/include tbb/lib tbb/bin MaterialX/include MaterialX/lib MaterialX/bin MaterialX/libraries imath/include imath/lib imath/bin python/313/include python/313/libs
git -C build/blender-5.2/sdk checkout
Remove-Item Env:GIT_LFS_SKIP_SMUDGE
python integrations/hydra/probe/fetch_sdk.py build/blender-5.2/sdk
```

The helper uses Blender's public LFS endpoint and verifies each downloaded library against its Git LFS SHA256. It avoids asking Git LFS to enumerate the entire partially cloned repository. Debug libraries are excluded: the probe uses the release CRT even with debug symbols.

Build in a VS2022 x64 developer shell with toolset 14.44:

```powershell
cmake -S integrations/hydra/probe -B build/blender-5.2/probe -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DKRR_BLENDER_SDK="$PWD/build/blender-5.2/sdk"
cmake --build build/blender-5.2/probe
& build/blender-5.2/blender-5.2.2-windows-x64/blender.exe --background --factory-startup --python-exit-code 1 --python integrations/blender/probe/run.py -- --plugin-dir build/blender-5.2/probe/plugin --artifacts build/blender-5.2/probe/artifacts
```

The harness renders three times, verifies the RGB gradient and constant depth from Blender's float32 multilayer EXRs, and writes `result.json`. It tests row orientation, channels, AOV allocation, repeated render lifecycle and teardown. Failures exit nonzero. Blender owns its USD/Python/TBB runtime DLLs; do not copy the standalone KiRaRay or standalone USD DLLs beside the probe.
