# Blender and USD integration

The optional integrations share conversion code, while the renderer owns its
scene, material, and rendering-session data. Neither OpenUSD nor Blender types
enter the public core interfaces.

| Profile | Entry point | Dependencies |
| --- | --- | --- |
| Ordinary | Existing executable and Python APIs | No USD; legacy volume-file import enabled |
| Standalone USD | Existing config `model` and import nodes | Python-free OpenUSD SDK; KiRaRay's selected Python |
| Blender | Native Hydra delegate and Python extension | Blender 5.2.2's patched USD/MaterialX/Python SDK |

Use separate build directories. See the [standalone guide](usd/README.md),
[Blender guide](blender/README.md), and [host ABI probe](hydra/probe/README.md).
The initial package is for personal Windows x64 use. The normal configuration
has both optional integrations disabled.

## Source boundaries

- `src/core/material/` defines typed expressions, validation, optimization, and
  the bounded GPU program. Constants, direct image bindings, and interpreted
  programs have separate execution paths. Surface, opacity, and emission each
  have a dependency slice. A later JIT can consume the same optimized semantics.
- `src/render/materials/openpbr/` implements a native spectral model subset.
  Preview Surface has its own identity and parameters. Existing materials retain
  their evaluator and identifiers. Read the [numerical and source notes](../src/render/materials/openpbr/NOTICE).
- `src/scene/interop.*` converts renderer-owned mesh/light descriptions and
  builds shared instances. Imported UV meshes get MikkTSpace tangents; existing
  importers retain their original behavior.
- `integrations/usd/` reads selected-time USD snapshots and lowers standard
  material networks. Asset resolution uses USD's resolver context.
- `integrations/hydra/` receives incremental scene changes and serializes GPU
  work on a native worker. It transfers owned color/depth snapshots through
  Hydra render buffers.
- `integrations/blender/kiraray/` registers the engine and validated exporter.
  Its producer preflight runs before Blender can discard unsupported nodes.

## Rendering lifetime

The internal C++ `RenderSession` separates initialization, bounded frame steps,
owned snapshots, updates, accumulation reset, cancellation, successful completion,
and cleanup. Camera/material/instance edits retain geometry where possible.
Topology edits may rebuild. Scene-update generations do not reuse reset sample
indices. Snapshots never save files; `finish()` performs configured completion
actions once. `close()` only releases resources.

Python `krr.render()` and `HeadlessRenderer.render()` still start a fresh batch
from their config snapshot on every call. Blender uses persistent sessions, one
rendering at a time; F12 takes priority over the single supported preview.

## Supported subset and diagnostics

The initial material model includes diffuse, metallic/dielectric reflection,
anisotropy, surface transmission, thin sheets, coat with an independent normal,
fuzz, opacity, and emission. Expressions include image/UV inputs, arithmetic,
explicit conversions/channels, ordinary mix, remapping, comparisons, selection,
normalization, rotation, and tangent-space normal maps.

The pinned exporter is part of the supported contract. Preflight rejects active
ColorRamp, Generated coordinates, arbitrary shader closure graphs, unsupported
blend/filter/normal-map modes, and lossy implicit color/vector-to-scalar links.
Use explicit Separate Color/XYZ nodes for scalar channels. The validated exporter
accepts a named UV Map only when it matches the active and render UV map on every
mesh using the material; empty/default UV selection uses the active set. It
stores diagnostics on affected USD materials; the importer validates the
resulting standard graph independently. Unsupported materials appear magenta and
report the offending node/input. An ordinary USD export cannot reveal information
that its producer already discarded.

The working space is linear Rec.709. OpenPBR luminance uses 1,000 nits per renderer
radiance unit; Preview Surface and legacy emission retain scene-linear units.
Blender owns display color management. Film alpha is opaque; material opacity
and primary-hit depth are separate features.

Subdivision, native skeletal evaluation, motion blur, imported volumes/hair,
displacement, bump derivatives, subsurface, thin films, UDIMs, arbitrary procedural
shading, material thumbnails, orthographic preview, and simultaneous rendered
viewports are deferred. Blender supplies evaluated geometry and animation state.

## Validation

CPU tests join the ordinary CTest/CI suite. GPU tests require
`KRR_ENABLE_GPU_TESTS=ON`; SDK integration tests additionally require
`KRR_ENABLE_SDK_TESTS=ON`. The Blender profile supplies its own background and
windowed acceptance harnesses. See [the test guide](../tests/README.md).
Recorded compatibility, performance, and lifecycle results are in
[the validation report](VALIDATION.md).

Artifacts remain beneath the selected build directory. References are generated
only by an explicit command and calibrated across several seeds. Legacy-scene
throughput is measured separately from graph materials; a confirmed result below
90% of matched baseline throughput blocks acceptance.
