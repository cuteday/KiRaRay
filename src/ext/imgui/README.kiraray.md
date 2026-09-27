# Dear ImGui in KiRaRay

Vendored from the upstream [docking branch](https://github.com/ocornut/imgui/tree/3bae66c735670619baf51391eba7f3d90a25d125):

- Revision: `3bae66c735670619baf51391eba7f3d90a25d125` (2026-09-21).
- Version: `1.93.0 WIP` (`IMGUI_VERSION_NUM=19297`).
- Updated: 2026-09-27.
- License: MIT; see [LICENSE](LICENSE).

The root `imgui*.cpp`, `imgui*.h`, `imconfig.h` and `imstb_*.h` files are
unmodified upstream sources. `LICENSE` is upstream `LICENSE.txt`.
`CMakeLists.txt` and this file belong to KiRaRay. No build-time download is needed.

KiRaRay owns window callbacks and input routing. Its UI adapter forwards selected
events to ImGui's current event API; no upstream platform backend is included.
`src/core/graphics/uirender.*` implements the shared Vulkan/D3D12 renderer through
NVRHI, including dynamic font texture updates. Docking is enabled inside the
main window; detached platform windows remain disabled.

To update, copy the same core files and license from one exact upstream docking
revision and record that revision here.
Review upstream `docs/CHANGELOG.txt` and `docs/BACKENDS.md`, rebuild all targets,
and validate text/input, DPI changes, font texture updates and cleanup on both
graphics APIs.
