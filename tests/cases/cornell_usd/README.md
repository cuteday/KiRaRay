# Blender USD Cornell case

`config.json` loads `tests/fixtures/usd/cornell.usda` through the ordinary scene
config. The stage was exported by Blender 5.2.2 from the Cornell scene built in
`integrations/blender/probe/render_scene.py`. It contains OpenPBR materials, a
perspective camera, and a normalized rectangular light. No tone mapping or
denoising is applied to the comparison image.

The spectral reference uses 16,384 frames. Five independent 128-frame seeds
measured normalized RMSE between 0.12035 and 0.12157; the threshold is 0.18235,
including a 50% margin. Black output (1.0) and half exposure (0.5) fail. The PNG
is a display preview; comparisons use the unclipped float32 NumPy array.

Regenerate deliberately:

```powershell
python tests/generate_reference.py --case tests/cases/cornell_usd `
  --artifacts <build>/tests/artifacts/<configuration>/usd_reference_generation
```

Set `KRR_BUILD_DIR` to a spectral USD-enabled build. Existing references require
`--force`; review the preview, metadata, and calibration before replacing them.
