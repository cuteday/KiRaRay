# Blender interoperability validation

Local acceptance on Windows x64, 2026-09-27, with an NVIDIA GeForce RTX 5080.
Baseline source revision: `84090368cd40cca9f5ae0a38758786dbd2da1262`.
Results below include the uncommitted implementation; reference metadata records
that dirty working tree explicitly. GPU tests have not run in GitHub Actions.

## Build and runtime profiles

| Profile | Compiler / CUDA | Runtime |
| --- | --- | --- |
| Ordinary spectral | MSVC 14.44 / CUDA 13.2 | OptiX 8, Vulkan and D3D12, Python 3.9 |
| Ordinary RGB | MSVC 14.29 / CUDA 12.9 | OptiX 8, Vulkan and D3D12, Python 3.9 |
| Standalone USD spectral | MSVC 14.44 / CUDA 13.2 | Python-free USD 26.03, MaterialX 1.39.4, TBB 2022.3.0; Python 3.9 bindings |
| Blender spectral | MSVC 14.44 / CUDA 13.2 | Blender 5.2.2, revision `d13f752e3b9c`, Python 3.13.13, patched USD 26.03 |

The Blender SDK is pinned to the official development bundle's `v5.2.2` tag,
commit `60d6e96b917568278d400a4024c98da0fb777338`. Its USD namespace is
`pxrBlender_v26_03__pxrReserved__`, distinct from standalone USD's
`pxrInternal_v0_26_3__pxrReserved__`. The SDK acquisition helper checks LFS hashes;
the standalone builder records source archive hashes in `install/krr-sdk.json`.
See the [ABI probe](hydra/probe/README.md) for acquisition and reproduction.

## Compatibility and correctness

- The independent synthetic Hydra delegate loaded and rendered three times in
  Blender. RGB row/channel error was below `2.6e-8`; the depth value was exact.
- Ordinary spectral CTest passed all 20 cases: CPU, material GPU checks,
  persistent sessions, readback, smoke, benchmark smoke, and image regression
  across both graphics APIs. The legacy Cornell NRMSE was `0.075994`, below its
  unchanged `0.113991` threshold.
- Ordinary CUDA 12/RGB passed all nine CPU and nine GPU cases, including both
  graphics backends and the same material/session checks. GPU cases took 320.43
  seconds; spectral-only image regression is intentionally absent in this build.
- Standalone USD passed all 11 CPU cases, including 19 actual Blender-exported
  supported graph fixtures, producer diagnostics, selected-time scene import,
  camera precedence, binary USD, material bindings, UV/default semantics,
  triangulation, and tangent continuity across material boundaries.
- The standalone USD Cornell smoke and spectral regression passed on both
  graphics APIs. Its deliberately generated 16,384-sample reference is stored
  with the case. Five 128-sample calibration seeds produced NRMSE
  `0.12035–0.12157`; the threshold is `0.182352`. Black output (`1.0`) and half
  exposure (`0.5`) fail that threshold.
- Wavefront with and without next-event estimation and the config-selected
  megakernel all rendered the USD Cornell scene. At 32×32, 128 samples their
  mean linear values were `0.427876`, `0.427940`, and `0.428865`, respectively.
  These are consistency observations, not a requirement for pixel identity.
- Material GPU checks cover 16 BSDF configurations, sampled PDFs and finite
  values, suitable energy tests, separate coat/base normals, opacity and light
  sampling, expression/simple-path agreement, texture color space, and repeated
  resource uploads. The nonabsorbing rough thin-sheet furnace result was
  `1.00061` within its numerical tolerance.
- Native Blender F12 checks passed on Vulkan and D3D12, including exact repeated
  images, camera edits, opacity/depth, and unsupported-material feedback. The
  direct emitter comparison gives KiRaRay/Cycles mean radiance `0.998437`.
  Convergence is latched only after the current generation reaches Hydra's AOV
  buffers. The last delegate joins its worker and releases the native context
  before DLL teardown; both F12 processes exit successfully.
- The Blender SDK profile passed all 11 CPU cases, including validated export
  with material-name collisions and unavailable volume-file loading. Its test
  environment includes Blender's root and `blender.shared` DLL directories.
  The packaged extension passed Blender's extension CLI validation.
- Both ordinary interactive backends completed eight-frame rendering runs and
  exited successfully. Vulkan still reports the existing presentation validation
  message `VUID-VkPresentInfoKHR-pWaitSemaphores-03268`. The same message is in
  pre-change logs from 2026-09-26; this integration does not change the graphics
  synchronization/presentation implementation.

CPU tests are selected by the existing CI `ctest -L cpu` command. SDK tests and
GPU tests are separately opt-in. The local CPU suite also runs with CUDA devices
hidden. CUDA 13 CPU executables have no NVIDIA driver DLL dependency. CUDA 12
test targets delay-load the driver and install a test-only guard that fails
before any driver load; the nine CPU cases pass with that guard. A deliberate
`cuInit` probe fails with the expected diagnostic, verifying the guard itself.

## Legacy performance gate

Matched warmed Cornell wavefront measurements: 512×512, seed 0, 32 warmup
frames, 128 measured frames, seven repetitions, validation disabled, idle GPU.
Values are median wall time per frame. Baseline and final binaries use the same
compiler, build variant, scene, settings, and hardware.

| Build / API | Baseline ms | Final ms | Baseline throughput retained |
| --- | ---: | ---: | ---: |
| Spectral / Vulkan | 1.93551 | 2.08395 | 92.88% |
| Spectral / D3D12 | 1.90168 | 2.00927 | 94.65% |
| CUDA 12 RGB / Vulkan | 1.91211 | 1.72539 | 110.82% |

All measured variants pass the required 90% throughput floor. New materials initially inflated
the legacy shading kernel to 254 registers and 2,088 stack bytes. Separate
legacy CUDA shading/light paths and an OptiX module-bound material category
restore the legacy scatter kernel to 128 registers and 24 stack bytes, matching
the retained original probe. Changing a persistent scene between legacy and
authored material categories rebuilds its OptiX pipeline while retaining scene
geometry. New graph materials are measured separately from this compatibility
gate.

An additional spectral Vulkan benchmark uses the USD Cornell geometry with the
actual Blender `SupportedTextured` graph on its surfaces: mapped UVs, three
textures, color mixing, roughness remapping, anisotropy, separate base/coat normal
maps, coat, and fuzz. At the same 512×512 and warmup/repetition settings it takes
**5.557 ms/frame** (seven-run range 5.545–5.561 ms). First batch setup takes
5.077 seconds; subsequent fresh batches take 2.286–2.336 seconds. Final RGB
readback takes 1.494–1.550 ms. This is a different shading workload, so its cost
is not a measured regression percentage or an isolated interpreter overhead.
The generated case and composition script remain in the build's artifacts.

## Persistent sessions and viewport behavior

The native session tests cover split stepping versus a fresh batch, accumulation
resets, camera/size changes, scene replacement, live material category and
emission edits, instance updates, depth, cancellation, owned snapshots, explicit
completion, worker-thread ownership, borrowed CUDA contexts, and repeated cleanup.
Tracked CUDA allocations stayed at 37,803,100 bytes through repeated native
sessions and returned to zero after context finalization. After exercising the
additional session paths, the two sequential-worker idle samples both reported
14,840,496,128 bytes of device-wide free memory. This observation is not a
device-wide leak assertion.

The windowed Blender acceptance harness exercised progressive rendering,
material/transform/geometry changes, perspective navigation, timeline changes,
diagnostic materials, F12/viewport handoff, file reload, real render-job
cancellation, preview resumption, and extension shutdown. Material and transform
edits retained the geometry build; a topology edit increased the build count.
The final D3D12 run completed every lifecycle stage. A separate active-viewport
disable, register, and unregister check also exited successfully.

Runs at 520×344 and two samples recorded:

| Native operation | Vulkan | D3D12 |
| --- | ---: | ---: |
| Initial initialization / first snapshot | 5,662 / 6,040 ms | 5,360 / 5,755 ms |
| Material update through first snapshot | 9.71 ms | 8.90 ms |
| Instance transform update through first snapshot | 5.86 ms | 5.49 ms |
| Camera update through first snapshot | 7.02 ms | 6.93 ms |
| Timeline transform update through first snapshot | 7.96 ms | 6.20 ms |
| Geometry rebuild through first snapshot | 2,168 ms | 2,172 ms |
| Snapshot including synchronization, depth and CPU readback | 3.46–6.75 ms | 2.74–42.60 ms |

These functional measurements were collected while other CPU builds were
running; they are not a controlled latency benchmark. Update times start when
the native worker begins applying the received update and end at its first
snapshot. They exclude Blender dependency-graph synchronization, worker scheduling
delay, and display presentation, so they are not end-to-end UI edit latency.
Snapshot time is not a pure PCIe copy measurement. D3D12 snapshots normally took
2.74–4.38 ms, with a 42.60 ms sample after the geometry rebuild.

The final D3D12 stop-button-to-render-job-completion measurement was 555 ms,
observed with 100 ms polling. The complete 5.65-second harness scenario includes
a deliberate five-second wait before cancellation. Rendering is bounded to one
frame between cancellation/update checks. F12 suspends the viewport's GPU session;
resuming it reconstructs its resources and starts fresh accumulation.

At the same viewport resolution, tracked native allocations returned to
204,529,500 bytes after reload and F12/cancellation handoffs. A temporary error
material used an additional 224 bytes. This checks stable renderer-owned
allocations, not all allocations made by Blender and its drivers. Worker counters
restart after the last delegate closes, including file reload.

## Deliberate limits

This is a documented material subset, not full Blender/OpenPBR conformance.
The [material notes](../src/render/materials/openpbr/NOTICE) describe the finite
glossy roughness floor, IOR handling, coat/fuzz approximation, and thin-sheet
energy compensation. Preview Surface's nonopaque `transparent` opacity mode is
diagnosed; presence and threshold cutouts are supported. Missing textures and
unsupported active nodes remain visible diagnostic materials.

Blender 5.2.2 writes scene-linear emission strength directly into its OpenPBR
network. The Blender adapter and validated-export material metadata explicitly
normalize that input to nits once. Generic unannotated USD retains standard
OpenPBR units. Use the validated exporter for Blender scenes; a raw export
cannot carry the preflight diagnostics and emission-unit contract automatically.

CPU Hydra transfer supplies genuine linear primary-hit depth and normalized
projected depth. Blender 5.2's CPU viewport display uploads only color, so this
path cannot occlude Blender overlays with the rendered depth. Film alpha remains
opaque. One perspective viewport is supported. Other planned exclusions are
listed in the [integration overview](README.md).

## Local evidence

Generated results are beneath the selected build directories:

- `build/blender-interop/final-ctest.log`: ordinary spectral acceptance.
- `build/blender-interop/legacy-performance.json`: matched spectral measurements
  and RGB measurements and compiler resource counts, with paths to raw results.
- `build/graphics-cuda12/benchmarks/Release/blender-final-vulkan/`: matched RGB
  timing and startup/readback measurements.
- `build/blender-interop/standalone/benchmarks/RelWithDebInfo/blender-graph-vulkan/`:
  the textured expression benchmark. Its generated USD/config are in
  `build/blender-interop/graph-benchmark-input/`.
- `build/blender-interop/standalone/cpu-acceptance.log`: USD CPU tests.
- `build/blender-interop/standalone/tests/artifacts/RelWithDebInfo/`: USD images,
  metrics, configs, reference generation, and native logs.
- `build/blender-interop/standalone/usd-probe/`: wavefront/megakernel observations.
- `build/blender-interop/host/tests/artifacts/RelWithDebInfo/`: Blender F12,
  emission-unit, and validated-export tests.
- `build/blender-interop/host-cpu-runtime-fixed.log`: final host CPU suite.
- `build/blender-interop/viewport-stop-job/`: windowed lifecycle counters,
  images, and logs for Vulkan.
- `build/blender-interop/viewport-final-d3d12/`: final D3D12 lifecycle counters,
  images, cancellation measurements, and logs.
- `build/blender-interop/viewport-disable/`: active extension shutdown and
  re-registration, with successful process exit.
- `build/blender-interop/package-validation-final.log`: Blender extension CLI
  validation of the packaged ZIP.

See the [test guide](../tests/README.md), [standalone guide](usd/README.md), and
[Blender guide](blender/README.md) for commands and deliberate reference updates.
