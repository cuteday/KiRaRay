#include "common.h"
#if KRR_ENABLE_PYTORCH
#define TORCH_API_INCLUDE_EXTENSION_H
#include <torch/torch.h>
#endif

#include "main/renderer.h"
#include "scene/importer.h"
#include "render/passes/denoise/denoise.h"
// put these headers before pybind or it causes a _DEBUG definition contradict

#include <optional>
#include "py.h"
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11_json.hpp>

NAMESPACE_BEGIN(krr)

namespace {
	std::unique_ptr<DenoiseBackend> sDenoiser;
	DenoiseBackend &getDenoiser() {
		Context::ensureInitialized();
		if (!sDenoiser) sDenoiser = std::make_unique<DenoiseBackend>();
		return *sDenoiser;
	}

	struct DenoiseBuffers {
		CUDABuffer rgb, normals, albedo, result;
		~DenoiseBuffers() {
			cudaDeviceSynchronize();
			for (auto *buffer : {&rgb, &normals, &albedo, &result})
				if (buffer->data()) cudaFree(reinterpret_cast<void *>(buffer->data()));
		}
	};
}

void run(const json& config) {
	RenderApp app;
	app.loadConfig(config);
	app.run();
}

py::array_t<float> imageArray(const std::vector<float> &pixels, const Vector2i &size) {
	py::array_t<float> image({size[1], size[0], 3});
	std::memcpy(image.mutable_data(), pixels.data(), pixels.size() * sizeof(float));
	return image;
}

py::array_t<float> renderImage(HeadlessRenderer &renderer, int64_t frames, uint64_t seed) {
	auto pixels = renderer.render(frames, seed);
	return imageArray(pixels, renderer.getFrameSize());
}

py::dict benchmark(HeadlessRenderer &renderer, int64_t frames, int64_t warmup, uint64_t seed,
	bool capture, py::object onCaptureBegin, py::object onCaptureEnd) {
	for (const auto *callback : {&onCaptureBegin, &onCaptureEnd})
		if (!callback->is_none() && !PyCallable_Check(callback->ptr())) {
			renderer.close();
			throw std::invalid_argument("capture callbacks must be callable or None");
		}
	auto batch = renderer.benchmark(frames, warmup, seed, capture,
		[&] { if (!onCaptureBegin.is_none()) onCaptureBegin(); },
		[&] { if (!onCaptureEnd.is_none()) onCaptureEnd(); });
	py::dict result;
	result["image"] = imageArray(batch.image, renderer.getFrameSize());
	result["timings"] = py::cast(batch.timings);
	result["frames"] = frames;
	result["warmup_frames"] = warmup;
	return result;
}

json getBuildInfo() {
	json info = {{"spectral", bool(KRR_RENDER_SPECTRAL)}, {"cuda_version", CUDART_VERSION},
		{"optix_version", OPTIX_VERSION}, {"build_type", KRR_BUILD_TYPE},
		{"project_root", KRR_PROJECT_DIR}, {"optix_profiling", bool(KRR_PROFILE_OPTIX)},
		{"debug_build", KRR_DEBUG_SELECT(true, false)}};
#ifdef _MSC_FULL_VER
	info["msvc_version"] = _MSC_FULL_VER;
#endif
	if (gpContext) {
		int driver = 0;
		cudaDriverGetVersion(&driver);
		info["device"] = gpContext->deviceProps.name;
		info["driver_version"] = driver;
		info["tracked_bytes"] = CUDATrackedMemory::singleton.BytesAllocated();
	}
	return info;
}

py::array_t<float> denoise(py::array_t<float, py::array::c_style | py::array::forcecast> rgb,
			std::optional<py::array_t<float, py::array::c_style | py::array::forcecast>> normals,
			std::optional<py::array_t<float, py::array::c_style | py::array::forcecast>> albedo) {
	if (rgb.ndim() != 3 || rgb.shape(0) <= 0 || rgb.shape(1) <= 0 ||
		(rgb.shape(2) != 3 && rgb.shape(2) != 4))
		throw std::invalid_argument("rgb must have shape [height, width, 3 or 4]");
	if (normals.has_value() != albedo.has_value())
		throw std::invalid_argument("normals and albedo must be supplied together");
	for (const auto *guide : {&normals, &albedo}) {
		if (guide->has_value() && (guide->value().ndim() != 3 ||
			guide->value().shape(0) != rgb.shape(0) || guide->value().shape(1) != rgb.shape(1) ||
			guide->value().shape(2) != rgb.shape(2)))
			throw std::invalid_argument("guides must match the rgb shape");
	}
	auto &denoiser = getDenoiser();
	Vector2i size = { (int)rgb.shape()[1], (int)rgb.shape()[0] };
	Log(Info, "Processing image with %lld channels...", rgb.shape()[2]);
	if(rgb.shape()[2] != 3 && rgb.shape()[2] != 4)
		logError("Incorrect image color channels (not 3)!"); // ensure 3 channels for hdr images.
	DenoiseBackend::PixelFormat pixelFormat = rgb.shape()[2] == 3
												  ? DenoiseBackend::PixelFormat::FLOAT3
												  : DenoiseBackend::PixelFormat::FLOAT4;
	
	bool hasGeometry = normals.has_value() && albedo.has_value();
	
	denoiser.resize(size);
	denoiser.setPixelFormat(pixelFormat);
	denoiser.setHaveGeometry(hasGeometry);
	
	py::buffer_info buf_rgb = rgb.request(), buf_albedo, buf_normals;
	py::array_t<float> result = py::array_t<float>(buf_rgb.size);
	py::buffer_info buf_result = result.request();

	DenoiseBuffers buffers;
	auto &gpumem_rgb = buffers.rgb, &gpumem_albedo = buffers.albedo;
	auto &gpumem_normals = buffers.normals, &gpumem_result = buffers.result;
	gpumem_rgb.resize(buf_rgb.size * sizeof(float));
	gpumem_result.resize(buf_result.size * sizeof(float));
	gpumem_rgb.copy_from_host((float *) buf_rgb.ptr, buf_rgb.size);

	if (hasGeometry) {
		buf_albedo = albedo.value().request();
		buf_normals = normals.value().request();
		gpumem_albedo.resize(buf_albedo.size * sizeof(float));
		gpumem_normals.resize(buf_normals.size * sizeof(float));
		gpumem_albedo.copy_from_host((float *) buf_albedo.ptr, buf_albedo.size);
		gpumem_normals.copy_from_host((float *) buf_normals.ptr, buf_normals.size);
	}

	denoiser.denoise((CUstream)0, (float *) gpumem_rgb.data(), (float *) gpumem_normals.data(),
					 (float *) gpumem_albedo.data(), (float *) gpumem_result.data());
	cudaDeviceSynchronize();
	gpumem_result.copy_to_host((float *) buf_result.ptr, buf_result.size);
	result = result.reshape({ rgb.shape()[0], rgb.shape()[1], rgb.shape()[2] });
	CUDA_SYNC_CHECK();
	return result;
}

#if KRR_ENABLE_PYTORCH
torch::Tensor denoise_torch_tensor(torch::Tensor rgb, 
	std::optional<torch::Tensor> normals, 
	std::optional<torch::Tensor> albedo) {
	auto &denoiser = getDenoiser();

	Vector2i size = {(int) rgb.size(1), (int) rgb.size(0)};
	Log(Info, "Processing image with %lld channels...", rgb.size(2));
	if (rgb.size(2) != 3 && rgb.size(2) != 4) logError("Incorrect image color channels (not 3)!");

	DenoiseBackend::PixelFormat pixelFormat = rgb.size(2) == 3
												  ? DenoiseBackend::PixelFormat::FLOAT3
												  : DenoiseBackend::PixelFormat::FLOAT4;

	bool hasGeometry = normals.has_value() && albedo.has_value();
	auto result = torch::empty_like(rgb);

	denoiser.resize(size);
	denoiser.setPixelFormat(pixelFormat);
	denoiser.setHaveGeometry(hasGeometry);

	denoiser.denoise((CUstream) 0, (float *) rgb.data_ptr(), 
			hasGeometry ? (float *) normals.value().data_ptr() : nullptr,
			hasGeometry ? (float *) albedo.value().data_ptr() : nullptr, 
			(float *) result.data_ptr());
	return result;
}	
#endif

PYBIND11_MODULE(pykrr, m) { 
	m.doc() = "KiRaRay python binding!";
	// Driver resources must be released before Windows starts unloading DLLs.
	py::module_::import("atexit").attr("register")(py::cpp_function([] {
		Renderer::closeActive();
		sDenoiser.reset();
		gpContext.reset();
	}));

	py::class_<HeadlessRenderer>(m, "HeadlessRenderer")
		.def(py::init([](const json &config, const string &assetRoot, bool validation) {
			return std::make_unique<HeadlessRenderer>(config, fs::path(assetRoot), validation);
		}), "config"_a, "asset_root"_a = "", "validation"_a = true)
		.def("render", &renderImage, "frames"_a, "seed"_a = 0)
		.def("benchmark", &benchmark, py::kw_only(), "frames"_a, "warmup"_a = 8,
			"seed"_a = 0, "capture"_a = false, "on_capture_begin"_a = py::none(),
			"on_capture_end"_a = py::none())
		.def("close", &HeadlessRenderer::close)
		.def_property_readonly("closed", &HeadlessRenderer::isClosed)
		.def("__enter__", [](HeadlessRenderer &renderer) -> HeadlessRenderer & {
			if (renderer.isClosed()) throw std::runtime_error("Renderer is closed");
			return renderer;
		}, py::return_value_policy::reference_internal)
		.def("__exit__", [](HeadlessRenderer &renderer, py::object, py::object, py::object) {
			renderer.close();
		});
	m.def("render", [](const json &config, int64_t frames, uint64_t seed,
		const string &assetRoot) {
		HeadlessRenderer renderer(config, fs::path(assetRoot));
		return renderImage(renderer, frames, seed);
	}, "config"_a, "frames"_a, "seed"_a = 0, "asset_root"_a = "");
	m.def("get_build_info", &getBuildInfo);

	m.def("run", &run,
		"Run KiRaRay renderer with specified configuration file",
		"config"_a);

	m.def("denoise", &denoise, 
		"Denoise the hdr image using optix's builtin denoiser", "rgb"_a,
		  "normals"_a = py::none(), "albedo"_a = py::none());
#if KRR_ENABLE_PYTORCH
	m.def("denoise_torch_tensor", &denoise_torch_tensor, 
		"Denoise the hdr image in tensor using optix's builtin denoiser", "rgb"_a,
		  "normals"_a = py::none(), "albedo"_a = py::none());
#endif
}

NAMESPACE_END(krr)
