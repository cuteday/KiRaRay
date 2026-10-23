# Kitchen material milestone

Local validation on Windows x64, 2026-09-28, with an NVIDIA GeForce RTX 5080
(driver 616.52). The implementation is based on
`e32a51cba99acad5f20c2621f5edfc82c6dc6de9`; results include the uncommitted changes.
The supplied Blender scenes and existing image references were not modified.

## Scene acceptance

`D:/Projects/scenes_blender-main/kitchen/kitchen.blend` passes Blender preflight
with **90 supported materials and no rejected materials**. Its annotated USD
export renders without native material diagnostics.

- Blender 5.2.2 LTS F12, Vulkan, 640×360, 256 samples: finite scene-linear color,
  genuine primary depth, successful completion, no material diagnostics.
- Blender F12, D3D12, 320×180, 128 samples: the same acceptance checks pass,
  including all 90 materials and an empty native diagnostic list.
- Standalone USD, Vulkan wavefront, 320×180, 128 samples: finite, nonblack output.
- Standalone USD, D3D12 wavefront and Vulkan megakernel, 64×36, 32 samples:
  repeated batches match, arrays are independent, and tracked CUDA allocations
  return to the same value after each batch.
- Optional producer checks: Glass of Water passes 7/7 materials and Cornell Box
  passes 8/8. Veach MIS passes 4/8; unimplemented glossy distributions remain
  explicit diagnostics. These producer checks do not establish rendering parity.

F12's initial run took 11.24 seconds including scene initialization and rendering.
That is an acceptance observation, not a steady-state throughput benchmark.
It published 23 images for 256 samples and captured depth once.

The Kitchen capture and annotated export are under
`build/blender-interop/host/tests/artifacts/RelWithDebInfo/kitchen/`.
Standalone captures are under
`build/blender-interop/standalone/tests/artifacts/RelWithDebInfo/kitchen-wavefront/`
and `kitchen-extra/`. Each retains the config, output, and relevant diagnostics.
The external-scene harness and reproduction commands are documented in
[`../blender/README.md`](../blender/README.md).

## Automated coverage

- Standalone CUDA 13.2 / MSVC 14.44 spectral: all 14 CPU and 16 GPU cases pass,
  including both graphics APIs, persistent sessions, readback, smoke, benchmark
  smoke, and unchanged legacy/USD image regressions.
- Blender SDK spectral profile: all 15 CPU/SDK and four GPU host tests pass,
  including repeated rendering on both APIs, emission, and environment maps.
- Ordinary CUDA 12.9 / MSVC 14.29 RGB, with USD and Hydra disabled: all 12 CPU
  tests and the material/composite and Vulkan/D3D12 render smoke tests pass.
- The new composite GPU test covers 13 cases, including analytic delta energy,
  Add weights, PDF integration, IOR transport from both sides, independent
  normals, numeric expression weights, and Cubic texture reconstruction.
- Thirty Python preflight tests cover producer semantics and traversal limits.
  SDK tests use 15 actual Blender-exported surface fixtures and synthetic tests
  for lowering, inactive branches, weights, emission, cycles, and graph limits.

Legacy Cornell's spectral NRMSE remains `0.0759938`, below the existing
`0.113991` threshold. No reference was regenerated.

## Legacy performance

The existing `benchmarks/cases/cornell_wavefront.json` ran through the same
standalone spectral build, Vulkan, 512×512, seed 0, 32 warmup frames, and five
128-frame repetitions. No other validation or build process ran during the
post-change measurements.

| Run | Median ms/frame | Throughput relative to baseline |
| --- | ---: | ---: |
| Before implementation | 2.155 | 100% |
| After implementation | 2.304 | 93.5% |
| Independent repeat after implementation | 2.359 | 91.4% |

Both comparisons stay above the agreed 90% throughput gate, but they do show a
6–9% lower median. These short desktop runs have appreciable variation
(0.148–0.162 ms/frame standard deviation); this is not evidence of zero overhead.
The minimum timings are 2.081 ms before and 2.091–2.094 ms after. This comparison
does not establish performance at other resolutions or for the new materials.

Full requests, build metadata, timings, and logs are retained beneath
`build/blender-interop/standalone/benchmarks/RelWithDebInfo/` in `kitchen-before`,
`kitchen-after`, and `kitchen-after-repeat`. Reproduce the post-change run with:

```powershell
python benchmarks/run.py run --build-dir build/blender-interop/standalone `
  --config benchmarks/cases/cornell_wavefront.json --resolution 512 512 `
  --frames 128 --warmup 32 --repeats 5 --seed 0 `
  --output-dir build/blender-interop/standalone/benchmarks/RelWithDebInfo/kitchen-check
```

## Behavior corrections

The leaf reuse exposed two existing dielectric defects. The parameterized
constructor assigned its IOR argument to itself; it now sets the member. Rough
dielectric PDF evaluation used the outgoing surface cosine for Fresnel instead
of the signed microfacet cosine; it now agrees with the sampler. The latter can
change legacy rough-glass MIS results and is an intentional correctness fix.

`BSDFSample::isDelta()` now includes the existing null-event flag, which is
required for index-matched dielectric transmission. It continues to identify
specular reflection and transmission as delta events.

## Scope

The supported approximations are listed in [README.md](README.md). In particular,
Principled inside Mix/Add, arbitrary layering, and nested or colored transparency
remain unsupported. White transparency uses the existing opacity traversal;
this milestone does not revise its treatment of correlated coverage across
multiple surfaces. GPU tests remain local, and no claim of Cycles pixel identity
or full MaterialX closure support is made.

# Runtime GGX and Beckmann validation

Local validation on Windows x64, 2026-10-23, with an RTX 5080, driver 616.52,
CUDA 13.2, OptiX 8, MSVC 14.44, and spectral RelWithDebInfo builds. The baseline is
`89635664e9e154f504f3c9dc0aeaec1429eff2ec`, including the preceding performance
fixes. Frozen baseline libraries match the final `exp19-resize-host` libraries
from that work. The following measurements supersede the older performance
observations above for this change.

## Unchanged scenes

Matched native helpers use Vulkan, seed 0, 64 warmup frames, and 256 measured
frames in each of three paired rounds with alternating run order. Timings include
frame submission and completion waits, but exclude import, setup, and readback.
No other GPU tests or builds ran during measurement.

| Scene | Integrator | Resolution | Baseline ms/frame | Updated ms/frame | Time change |
| --- | --- | --- | ---: | ---: | ---: |
| Original OBJ Cornell | Wavefront | 750×750 | 2.9615 | 2.9611 | −0.01% |
| Original OBJ Cornell | Megakernel | 750×750 | 2.2533 | 2.2511 | −0.10% |
| Kitchen USD | Wavefront | 640×360 | 8.1016 | 8.1071 | +0.07% |
| Kitchen USD | Megakernel | 640×360 | 11.9352 | 11.9155 | −0.17% |

These are medians across the three runs. All 12 baseline/updated image pairs are
bitwise identical. There is no measurable steady-state regression in these cases;
the measurements do not establish performance at other resolutions or on other GPUs.
One Cornell megakernel run took 3.679 ms/frame; that outlier remains in the raw
results. Three additional alternating pairs did not reproduce it and had a
0.56% lower updated median, with all images again bitwise identical.

Requests, frozen binaries and library hashes, configs, images, and timings are
under `build/beckmann-20261023/`. The main report is `compare/summary.json`;
the outlier confirmation is `compare-cornell-confirm/summary.json`.

## Mixed-distribution Kitchen

A separate USD sublayer changes 12 of Kitchen's 23 native conductor/dielectric
leaves, selected in stable path order, to Beckmann. The original scene and its
asset paths remain unchanged. The same updated binary renders the original and
mixed scenes using the settings above in three alternating paired rounds.

| Integrator | All-GGX ms/frame | Mixed ms/frame | Time change |
| --- | ---: | ---: | ---: |
| Wavefront | 8.1052 | 8.1365 | +0.39% |
| Megakernel | 11.9296 | 11.7718 | −1.32% |

Both images are finite and nonblack, differ from the all-GGX image (NRMSE about
0.049), and repeat bitwise for the same configuration. Different distributions
change the light paths, so these results do not isolate the cost of one BSDF
operation or establish that Beckmann is faster. The overlay, selection manifest,
generator, and results are retained in `build/beckmann-20261023/`, with the report
in `compare-mixed/summary.json`.

## Correctness and integration

- Both standalone USD and Blender/Hydra profiles build, including the Blender
  extension package. Their CPU suites pass 14/14 and 15/15 tests respectively.
- Numerical microfacet tests cover host and CUDA sampler boundaries, anisotropic
  NDF/PDF normalization, sampled moments, conductor/dielectric sampling and
  evaluation, both sides of glass, grazing angles, delta events, TIR, and matched IOR.
  All 16 numerical cases have zero invalid samples. Beckmann PDF integrals differ
  from one by at most 0.00016; sampled/integrated energies agree within 0.00048.
- All 27 composite GPU cases pass, covering GGX, Beckmann, mixed components,
  material upload, delta behavior, and repeated replacement. The existing Python
  render smoke tests pass on Vulkan and D3D12; legacy Cornell and USD spectral
  image regressions also pass against their unchanged references.
- The Blender preflight suite passes 33 tests. Ten actual Blender-exported
  fixtures cover mixed distributions, reused groups, emission-only branches,
  white transparency, and inactive branches. USD tests validate their identities
  and reject stale/conflicting metadata.
- Blender F12 renders Glossy, Glass, and mixed materials through GGX → Beckmann
  → GGX in one process. All nine renders have valid color/depth and no material
  diagnostics. Beckmann changes each image; returning to GGX reproduces its
  original image exactly.

GPU test artifacts remain under each build's
`tests/artifacts/RelWithDebInfo/`; the Blender runtime check is in
`build/blender-interop/host/tests/artifacts/RelWithDebInfo/blender_distributions/`.
No existing image reference or user scene was changed. These runs cover the
spectral CUDA 13.2 builds; RGB and older CUDA builds were not revalidated in this
pass. Existing leaf roughness mapping, OpenPBR/Preview Surface behavior, and
legacy JSON GGX defaults are unchanged.
