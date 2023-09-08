#include <cuda.h>
#include <cuda_runtime.h>

#include "device/cuda.h"
#include "integrator.h"
#include "wavefront.h"
#include "workqueue.h"
#include "render/media.h"
#include "render/profiler/profiler.h"

KRR_NAMESPACE_BEGIN

void WavefrontPathTracer::sampleMediumInteraction(int depth) {
	PROFILE("Sample medium interaction");
	ForAllQueued(mediumSampleQueue, maxQueueSize,
				 KRR_DEVICE_LAMBDA(const MediumSampleWorkItem &w){
			/* Each ray is either absorped or scattered after some null collisions... */
			Color L(0);
			Ray ray			= w.ray;
			float tMax		= w.tMax;
			Sampler sampler = &pixelState->sampler[w.pixelId];
			Medium medium	= ray.medium;

			return;

			// sample corresponding majorant... 
			RayMajorant iter = medium.sampleRay(ray, tMax);
			Color T_maj		 = iter.sigma_maj;
	}, gpContext->cudaStream);
}

void WavefrontPathTracer::sampleMediumScattering(int depth) {
	PROFILE("Sample medium scattering");
	ForAllQueued(mediumScatterQueue, maxQueueSize,
				 KRR_DEVICE_LAMBDA(const MediumScatterWorkItem &w){
			return;
			const Vector3f& wo = w.wo;
			Sampler sampler	   = &pixelState->sampler[w.pixelId];
			// [PART-A] Sample direct lighting with ShadowRayTr
			LightSampleContext ctx{w.p, Vector3f::Zero()};
			SampledLight sampledLight = lightSampler.sample(sampler.get1D());
			if (sampledLight) {
				Light light	   = sampledLight.light;
				LightSample ls = light.sampleLi(sampler.get1D(), ctx);
			}
			// [PART-C] Sample indirect lighting with scattering function
			PhaseFunctionSample ps = w.phase.sample(wo, sampler.get1D());
			// Russian roulette

			Ray ray{w.p, ps.wi, w.time, w.medium};
			RayWorkItem r{};
			r.ray	  = ray;
			r.ctx	  = ctx;
			r.thp	  = w.thp * ps.p / ps.pdf;
			r.depth	  = w.depth + 1;
			r.pixelId = w.pixelId;
			if (r.thp.any()) nextRayQueue(depth)->push(r);
	}, gpContext->cudaStream);
}

KRR_NAMESPACE_END