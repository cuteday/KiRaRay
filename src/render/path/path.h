#pragma once
#include <device/cuda.h>

#include "sampler.h"
#include "device/scene.h"
#include "render/bsdf.h"
#include "render/spectrum.h"
#include "render/lightsampler.h"

KRR_NAMESPACE_BEGIN

using namespace shader;

enum {
	RADIANCE_RAY_TYPE			  = 0,
	SHADOW_RAY_TYPE				  = 1,
	SHADOW_TRANSMITTANCE_RAY_TYPE = 2,
	RAY_TYPE_COUNT
};

const string shaderProgramNames[RAY_TYPE_COUNT] = {
	"Radiance",
	"ShadowRay"
	"ShadowTransmittanceRay"
};

class MegakernelPathTracer;

template <>
struct LaunchParameters<MegakernelPathTracer> {
	uint frameID{0};
	Vector2i fbSize = Vector2i::Zero();
	// per pixel debugging output
	bool debugOutput	 = false;
	bool NEE			 = false; // enable next event estimation (and MIS)
	Vector2i debugPixel	 = {960, 540};
	int maxDepth		 = 10;
	float probRR		 = 0.8;
	float clampThreshold = 1e4f; // clamp max radiance contrib per frame
	int spp				 = 1;
	int lightSamples	 = 1;
	// scene
	Camera::CameraData camera;
	LightSampler lightSampler;
	rt::SceneData sceneData;
	const RGBColorSpace *colorSpace;
	CudaRenderTarget colorBuffer;
	OptixTraversableHandle traversable{0};
};

struct PathData {
	Spectrum L{};			   // total contribution to the current pixel
	Spectrum throughput;	   // maintain the throughput of path
	float pdf;				   // BxDF sampling pdf from last scatter
	int depth;				   // number of vertices along the path
	BSDFType bsdfType;		   // the sampled type of the last scatter event
	Sampler sampler;		   // rng
	SampledWavelengths lambda; // sampled wavelength
	LightSampler lightSampler; // randomly choosing a light source
	LightSampleContext ctx;	   // *last* context used for direct light sampling
	SurfaceInteraction intr;   // surface interaction of the *current* hit
	Ray ray;				   // The last scattered ray
};

KRR_NAMESPACE_END