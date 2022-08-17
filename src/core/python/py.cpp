#include <optional>
#include "py.h"
#include "common.h"

#include "render/wavefront/integrator.h"
#include "render/passes/accumulate.h"
#include "render/passes/tonemapping.h"
#include "render/passes/denoise.h"
#include "render/path/pathtracer.h"
#include "main/renderer.h"
#include "scene/importer.h"

KRR_NAMESPACE_BEGIN

void run(const char scene_file[], const char env_file[] = nullptr) {
	if (!gpContext)
		gpContext = Context::SharedPtr(new Context());
	RenderApp app(KRR_PROJECT_NAME, { 1920, 1080 },
				  { RenderPass::SharedPtr(new WavefrontPathTracer()),
					RenderPass::SharedPtr(new AccumulatePass()),
					RenderPass::SharedPtr(new ToneMappingPass()),
					RenderPass::SharedPtr(new DenoisePass(false)) });
	Scene::SharedPtr scene = Scene::SharedPtr(new Scene());
	if (env_file) scene->addInfiniteLight(InfiniteLight(env_file));
	AssimpImporter importer;
	importer.import(scene_file, scene);
	app.setScene(scene);
	app.run();
}

py::array_t<float> denoise(py::array_t<float, py::array::c_style | py::array::forcecast> rgb,
			std::optional<py::array_t<float, py::array::c_style | py::array::forcecast>> normals,
			std::optional<py::array_t<float, py::array::c_style | py::array::forcecast>> albedo) {
	static bool initialized{};
	static DenoiseBackend denoiser;
	if (!initialized) {
		if (!gpContext)
			gpContext = Context::SharedPtr(new Context());
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
	result.reshape({ rgb.shape()[0], rgb.shape()[1], rgb.shape()[2] });
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

	denoiser.denoise((float *) gpumem_rgb.data(), (float *) gpumem_normals.data(),
					 (float *) gpumem_albedo.data(), (float *) gpumem_result.data());
	cudaDeviceSynchronize();
	gpumem_result.copy_to_host((float *) buf_result.ptr, buf_result.size);
	CUDA_SYNC_CHECK();
	return result;
}

PYBIND11_MODULE(pykrr, m) { 
	m.doc() = "KiRaRay python binding!";

	m.def("run", &run, 
		"Run KiRaRay renderer with default configuration", 
		"scene"_a, "env"_a=nullptr);

	m.def("denoise", &denoise, 
		"Denoise the hdr image using optix's builtin denoiser", "rgb"_a,
		  "normals"_a = py::none(), "albedo"_a = py::none());
}

KRR_NAMESPACE_END