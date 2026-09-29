# Material translation

This layer lowers material networks from USD import and Hydra into KiRaRay's
renderer-owned material descriptions. Blender-specific validation stays in
`integrations/blender`; CUDA shading has no USD, MaterialX, or Blender dependency.

The existing OpenPBR and USD Preview Surface paths retain their behavior. A
MaterialX `surface` can additionally contain the following scattering and emission
nodes:

| MaterialX input | KiRaRay representation |
| --- | --- |
| Oren–Nayar / Burley diffuse | Native Lambertian diffuse |
| Conductor | Native GGX conductor, matching RGB normal-incidence reflectance |
| Dielectric with `scatter_mode=RT` | Native GGX dielectric |
| BSDF Mix / Add / scalar multiply | Flat weighted native components |
| Uniform EDF; EDF Mix / Add / scalar or color multiply | A separate radiance expression |

Texture and arithmetic inputs use the existing typed expression program, including
Linear, Closest, and Cubic image filtering. Material graphs are validated and
optimized before upload. Unknown active scattering, emission, or expression nodes
produce the diagnostic material rather than disappearing from the graph.

## Composition

`Mix(A, B, t)` contributes `(1-t) A + t B`, with the factor clamped to `[0, 1]`.
`Add(A, B)` contributes `A + B`: its physical weights are **not** normalized.
Nested graphs flatten into native components referencing the same numeric graph.
Repeated references to the same leaf can merge; zero-weight branches are removed.
At most eight components survive compilation. A unit-weight native leaf collapses
to its direct path. Closure traversal is also bounded to 256 nested nodes and
4096 expanded visits; deeply nested or repeatedly shared Mix/Add branches receive
an actionable diagnostic before compilation can consume unbounded work.

Rendering evaluates the complete mixture value and PDF. The probability of
selecting a component to generate a direction is distinct from its physical
weight. Delta events retain discrete-event handling. Emission is compiled
separately, so emission plus diffuse needs only one scattering component. Shader
composition does not select one material for the entire scattering vertex.

## Deliberate approximations and limits

- Rough diffuse uses Lambertian scattering, without Oren–Nayar or Burley roughness
  behavior or their energy compensation.
- Conductors preserve normal-incidence reflectance. Artistic IOR inputs reuse
  their reflectivity; complex IOR inputs convert to RGB reflectance. KiRaRay's
  native conductor reconstructs an approximate spectral conductor, so the edge
  color and angular Fresnel curve can differ.
- MaterialX microfacet roughness is alpha. Its two axes are averaged and converted
  to the native perceptual roughness. This initial adapter uses isotropic GGX.
- Dielectric tint, IOR, zero roughness, and index matching are retained. Separate
  reflection-only or transmission-only dielectric nodes are not yet supported.
- Thin films, MaterialX thin-walled surfaces, vertical closure layering, arbitrary
  colored BSDF multipliers, and additional scattering families remain unsupported.
- Principled works as the existing OpenPBR surface. Blender expands Principled
  inside Mix/Add into a larger closure graph; that exported representation is not
  currently supported. Compositions of the native leaf shaders above are supported.

These are approximations to the exported MaterialX representation, not a promise
of matching Cycles. Blender's exporter can itself change shader behavior.

## White transparency correction

Blender 5.2 can export a white Transparent shader as white diffuse and set the
surface opacity to zero. Preflight recognizes a restricted root Mix with exactly
one constant-white Transparent branch and one supported opaque branch. It records
which branch is opaque without modifying the Blender material.

Hydra passes the map in `krr:opaqueMixBranches`; validated USD export writes
`kiraray:opaqueMixBranch` (`bg` or `fg`) on the material. Translation selects the
opaque branch's scattering and uses the original mix factor as coverage. Emission
is unweighted by that coverage because traversal applies it. A pure emissive
opaque branch is handled as an absorbing surface with emission. Both visible and
sampled emission must retain the same coverage semantics.

This correction is deliberately limited to validated root mixes. Colored
transparency, nested transparency, and standalone Transparent shaders remain
diagnostic. An ordinary USD file cannot recover Blender information lost before
export, so unannotated files use their authored MaterialX values.

MaterialX EDF color is renderer radiance. Blender's existing
`emissionLuminanceScale` correction applies only to the OpenPBR surface, whose
emission input is luminance; it must not rescale EDF color a second time.

## Validation

`tests/fixtures/usd/blender52/surfaces.usda` is an actual Blender export covering
native leaves, Mix/Add, emission, white transparency, and Cubic textures.
`tests/blender/material_fixtures.py --surface-only` recreates that fixture.
`tests/unit/test_usd.cpp` checks those networks and the lowerer's weight, unit,
cycle, and inactive-branch behavior.

After building the standalone SDK test target:

```powershell
ctest --test-dir build/blender-interop/standalone -R krr_usd --output-on-failure
```

GPU mixture tests and the Kitchen scene separately validate rendering. Normal
tests never rewrite Blender files or references.
See [VALIDATION.md](VALIDATION.md) for local acceptance results, performance,
and the related dielectric correctness fixes.
