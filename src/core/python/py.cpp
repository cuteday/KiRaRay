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

KRR_NAMESPACE_BEGIN

void run(const json& config) {
	if (!gpContext) gpContext = std::make_unique<Context>(); 
	{
		RenderApp app;
		app.loadConfig(config);
		app.run();
	}
}

py::array_t<float> denoise(py::array_t<float, py::array::c_style | py::array::forcecast> rgb,
			std::optional<py::array_t<float, py::array::c_style | py::array::forcecast>> normals,
			std::optional<py::array_t<float, py::array::c_style | py::array::forcecast>> albedo) {
	static bool initialized{};
	static DenoiseBackend denoiser;
	if (!initialized) {
		if (!gpContext) gpContext = std::make_unique<Context>();
		initialized = true;
	}
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

	CUDABuffer gpumem_rgb(buf_rgb.size * sizeof(float)), gpumem_albedo, gpumem_normals;
	CUDABuffer gpumem_result(buf_result.size * sizeof(float));
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
	auto result = torch::empty_like(rgb);
	return result;
}	
#endif

PYBIND11_MODULE(pykrr, m) { 
	m.doc() = "KiRaRay python binding!";

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

KRR_NAMESPACE_END