This stage and its two tiny textures were generated with Blender 5.2.2 LTS by
`tests/blender/material_fixtures.py`. They preserve the actual exporter graph
structure, including MaterialX node-graph forwarding and conversion nodes.
The 19 supported materials exercise image inputs, normal mapping, grouped
nodes, arithmetic, Mapping modes, Mix, Map Range, and explicit channels.
Seven producer-side rejection cases exercise features lost by the exporter.

`emission.usda` is a separate Blender 5.2.2 export of a white Principled emitter
with strength 1 and no reflective lobes. Its raw OpenPBR luminance is 1; tests
add validated-export unit metadata to check the explicit conversion to 1000 nits.

The ordinary USD export deliberately has no KiRaRay diagnostics. The tests
distinguish graph validation from Blender preflight: unsupported Blender nodes
already discarded by this export cannot be recovered by a USD reader.

Regenerate deliberately with the pinned Blender executable, review the graph
diff, and copy `materials.usda` and `textures/` from the selected build's test
artifact directory. Do not regenerate during ordinary tests.

`surfaces.usda` contains 15 actual Blender exports of native leaf BSDFs, Mix/Add
graphs, weighted emission, Cubic image filtering, and the bounded white
Transparent mixture. It reuses the same `textures/color.png`. The transparent
fixtures include `kiraray:opaqueMixBranch` producer metadata; emission units
are also explicit. Generate it separately with
`tests/blender/material_fixtures.py -- --surface-only --output <build-artifacts>`
and deliberately copy only `surfaces.usda` after reviewing the graph changes.
