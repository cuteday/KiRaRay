# Graphics backends

KiRaRay uses the official NVRHI submodule without local patches. Vulkan is the
default graphics API; Windows builds also enable D3D12 unless configured with
`-DKRR_ENABLE_D3D12=OFF`. Select an API with the top-level config field:

```json
"graphics_api": "d3d12"
```

The same setting works with the interactive executable, Python headless API,
and benchmark configs. CUDA/OptiX still perform ray tracing on the same NVIDIA
GPU. Changing graphics API requires a new renderer instance.

## Responsibilities

- `device`: common device lifecycle, optional GLFW window, input and frame loop.
  A private `GraphicsBackend` owns native device/queues and presentation.
- `rendercontext`: shared frame resources and graphics/CUDA handoff.
- `rendertarget`: NVRHI render textures and their CUDA surface mappings.
- `interop`: move-only texture/buffer mappings and GPU handoff sequencing.
- `binding`, `descriptor`, `textureloader`, `helperpass`, `scene`, `uirender`:
  helpers implemented with generic NVRHI APIs.
- `shader`: runtime DXC compilation of shared HLSL to SPIR-V or DXIL.
- `vulkan/`, `d3d12/`: native device, swapchain and interop implementations.

Render passes use `nvrhi::IDevice`, `RenderContext`, and CUDA render targets.
Native API headers and handle operations belong in the backend implementations.
There is no additional general graphics abstraction above NVRHI.

Presentation synchronization belongs to each native backend. D3D12 signals its
frame fence after `Present` and waits before reusing or releasing swapchain
images; NVRHI's submission tracking alone does not cover DXGI presentation.
Vulkan uses FIFO with VSync enabled and prefers Immediate when disabled,
falling back to Mailbox or FIFO if Immediate is unavailable.

## CUDA sharing

NVRHI creates exportable allocations using `SharedResourceFlags::Shared`.
KiRaRay imports the public shared handles into CUDA, using actual allocation
sizes and dedicated-allocation flags. Mappings retain their NVRHI resources.
CUDA views/mappings/imports and owned OS handles are released before the
backing graphics resource. Callers complete pending GPU work before destroying
or replacing resources; session shutdown and resize perform this synchronization.

Graphics-to-CUDA and CUDA-to-graphics transfers use separate application-owned
timeline semaphores on Vulkan and shared fences on D3D12. Counters persist
throughout a session. Vulkan additionally transfers queue-family ownership with
images in GENERAL layout; D3D12 resources return to Common at the handoff.
KiRaRay never replaces NVRHI's internal queue tracking objects.

The pass executor calls `beginCuda()`/`endCuda()` around consecutive CUDA passes,
returning ownership before the next graphics pass or the end of the frame. A graphics
pass that temporarily writes shared resources from CUDA can use
`RenderContext::CudaScope`; additional shared buffers must be registered with
the context for the duration of their use. Keep registration and mapping
lifetimes within the owning renderer session.

## Build and validation

NVRHI fetches its pinned Vulkan-Headers and DirectX-Headers. The installed
Vulkan SDK still supplies the loader and DXC; its headers do not override the
newer NVRHI header dependency. D3D12 sources are compiled only when enabled.

For Vulkan validation with CUDA interop, use the layer from SDK 1.4.321 or newer.
SDK 1.4.313 can report `VUID-vkQueuePresentKHR-pWaitSemaphores-03268` for valid
CUDA signals on exported timeline semaphores. Khronos corrected this external
signal tracking in [June 2025](https://github.com/KhronosGroup/Vulkan-ValidationLayers/commit/8f3509a0aee2533c249374989d5dbcfd09723aca).
The installed validation layer is separate from NVRHI's fetched Vulkan headers.

With `KRR_ENABLE_GPU_TESTS=ON`, CTest registers the same readback, wavefront,
benchmark and spectral regression cases for each enabled graphics API. Use
`ctest --test-dir <build> -L vulkan --output-on-failure` or `-L d3d12` to select
a backend. GPU tests remain local; CI builds both backends and runs CPU tests.
See [test documentation](../../../tests/README.md) for artifacts and references.
