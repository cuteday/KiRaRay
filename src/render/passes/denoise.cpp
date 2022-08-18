#include "denoise.h"
#include "util/check.h"
#include "render/profiler/profiler.h"
#include "device/context.h"

KRR_NAMESPACE_BEGIN

void DenoiseBackend::initialize() {
	CUcontext cudaContext;
	cuCtxGetCurrent(&cudaContext);
	CHECK(cudaContext != nullptr);
	const OptixDeviceContext &optixContext = gpContext->optixContext;
	if (denoiserHandle) optixDenoiserDestroy(denoiserHandle);

	OptixDenoiserOptions options = {};
#if (OPTIX_VERSION >= 70300)
	if (haveGeometryBuffer)
		options.guideAlbedo = options.guideNormal = 1;

	OPTIX_CHECK(optixDenoiserCreate(optixContext, OPTIX_DENOISER_MODEL_KIND_HDR, &options,
									&denoiserHandle));
#else
	options.inputKind = haveGeometryBuffer ? 
		(pixelFormat == PixelFormat::FLOAT3 ? OPTIX_DENOISER_INPUT_Color3f_ALBEDO_NORMAL : OPTIX_DENOISER_INPUT_Color4f_ALBEDO_NORMAL) : 
		(pixelFormat == PixelFormat::FLOAT3 ? OPTIX_DENOISER_INPUT_Color3f : OPTIX_DENOISER_INPUT_Color4f);

	OPTIX_CHECK(optixDenoiserCreate(optixContext, &options, &denoiserHandle));

	OPTIX_CHECK(optixDenoiserSetModel(denoiserHandle, OPTIX_DENOISER_MODEL_KIND_HDR, nullptr, 0));
#endif
	// re-create compute memory resources.
	OPTIX_CHECK(optixDenoiserComputeMemoryResources(denoiserHandle, resolution[0], resolution[1],
													&memorySizes));

	denoiserState.resize(memorySizes.stateSizeInBytes);
	scratchBuffer.resize(memorySizes.withoutOverlapScratchSizeInBytes);
	intensity.resize(sizeof(float));

	OPTIX_CHECK(optixDenoiserSetup(denoiserHandle, 0 /* stream */, resolution[0], resolution[1],
								   CUdeviceptr(denoiserState.data()), memorySizes.stateSizeInBytes,
								   CUdeviceptr(scratchBuffer.data()),
								   memorySizes.withoutOverlapScratchSizeInBytes));
}

void DenoiseBackend::denoise(float *rgb, float *normal, float *albedo, float *result) {
	std::array<OptixImage2D, 3> inputLayers;
	int inputPixelStride = pixelFormat == PixelFormat::FLOAT3 ? sizeof(Color3f) : sizeof(Color4f);
	int outputPixelStride = pixelFormat == PixelFormat::FLOAT3 ? sizeof(Color3f) : sizeof(Color4f);
	int nLayers = haveGeometryBuffer ? 3 : 1;
	
	for (int i = 0; i < nLayers; ++i) {
		inputLayers[i].width			  = resolution[0];
		inputLayers[i].height			  = resolution[1];
		inputLayers[i].rowStrideInBytes	  = resolution[0] * inputPixelStride;
		inputLayers[i].pixelStrideInBytes = inputPixelStride;
		inputLayers[i].format = pixelFormat == PixelFormat::FLOAT3 ? 
			OPTIX_PIXEL_FORMAT_FLOAT3 : OPTIX_PIXEL_FORMAT_FLOAT4;
	}
	inputLayers[0].data = CUdeviceptr(rgb);
	if (haveGeometryBuffer) {
		CHECK(normal != nullptr && albedo != nullptr);
		inputLayers[1].data = CUdeviceptr(albedo);
		inputLayers[2].data = CUdeviceptr(normal);
	} else
		CHECK(normal == nullptr && albedo == nullptr);

	OptixImage2D outputImage;
	outputImage.width			   = resolution[0];
	outputImage.height			   = resolution[1];
	outputImage.rowStrideInBytes   = resolution[0] * outputPixelStride;
	outputImage.pixelStrideInBytes = outputPixelStride;
	outputImage.format = pixelFormat == PixelFormat::FLOAT3 ? 
		OPTIX_PIXEL_FORMAT_FLOAT3 : OPTIX_PIXEL_FORMAT_FLOAT4;
	outputImage.data			   = CUdeviceptr(result);

	OPTIX_CHECK(optixDenoiserComputeIntensity(denoiserHandle, 0 /* stream */, &inputLayers[0],
											  CUdeviceptr(intensity.data()), CUdeviceptr(scratchBuffer.data()),
											  memorySizes.withoutOverlapScratchSizeInBytes));

	OptixDenoiserParams params = {};
	params.denoiseAlpha		   = 1;
	params.hdrIntensity		   = CUdeviceptr(intensity.data());
	params.blendFactor		   = 0; 

#if (OPTIX_VERSION >= 70300)
	OptixDenoiserGuideLayer guideLayer;
	if (haveGeometryBuffer) {
		guideLayer.albedo = inputLayers[1];
		guideLayer.normal = inputLayers[2];
	}

	OptixDenoiserLayer layers;
	layers.input  = inputLayers[0];
	layers.output = outputImage;

	OPTIX_CHECK(optixDenoiserInvoke(denoiserHandle, 0 /* stream */, &params,
									CUdeviceptr(denoiserState.data()), memorySizes.stateSizeInBytes,
									&guideLayer, &layers, 1 /* # layers to denoise */,
									0 /* offset x */, 0 /* offset y */, CUdeviceptr(scratchBuffer.data()),
									memorySizes.withoutOverlapScratchSizeInBytes));
#else
	OPTIX_CHECK(optixDenoiserInvoke(denoiserHandle, 0 /* stream */, &params,
									CUdeviceptr(denoiserState.data()), memorySizes.stateSizeInBytes,
									inputLayers.data(), nLayers, 0 /* offset x */, 0 /* offset y */,
									&outputImage, CUdeviceptr(scratchBuffer.data()),
									memorySizes.withoutOverlapScratchSizeInBytes));
#endif
}

void DenoiseBackend::resize(Vector2i size) { 
	if (resolution == size) return;
	resolution = size; 
	initialize();
}

void DenoiseBackend::setHaveGeometry(bool haveGeometry) {
	if (haveGeometryBuffer == haveGeometry) return;
	haveGeometryBuffer	= haveGeometry;
	initialize();
}

void DenoiseBackend::setPixelFormat(PixelFormat format) {
	if (pixelFormat == format) return;
	pixelFormat = format;
	initialize();
}

void DenoisePass::render(CUDABuffer &frame) {
	if (!mEnable) return;
	PROFILE("Denoise");
	mBackend.denoise((float *) frame.data(), nullptr, nullptr, (float *) frame.data());
}

void DenoisePass::renderUI() { 
	ui::Checkbox("Enabled", &mEnable);
	if (!mEnable) return;
	if (ui::Checkbox("Use geometry buffer", &mUseGeometry))
		mBackend.setHaveGeometry(mUseGeometry);
}

void DenoisePass::resize(const Vector2i& size) { mBackend.resize(size); }

KRR_NAMESPACE_END