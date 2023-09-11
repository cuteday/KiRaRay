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
			Color thp		= w.thp;
			Sampler sampler = &pixelState->sampler[w.pixelId];
			Medium medium	= ray.medium;

			return;

			// sample corresponding majorant... 
			RayMajorant iter = medium.sampleRay(ray, tMax);
			Color T_maj		 = iter.sigma_maj;

			// Either absorped or scattered after several null collisions

			// Return if scattered or no throughput or max depth reached

			// The ray escaped from our sight...
			if (w.tMax == M_FLOAT_MAX) {
				MissRayWorkItem r;
				r.bsdfType = w.bsdfType;
				missRayQueue->push(r);
			}

			// We finally reached the surface...!
			if (w.intr.light) {
				// Handle contribution for light hit

			}

			// Next surface scattering event...!


	}, gpContext->cudaStream);
}

void WavefrontPathTracer::sampleMediumScattering(int depth) {
	PROFILE("Sample medium scattering");
	ForAllQueued(mediumScatterQueue, maxQueueSize,
				 KRR_DEVICE_LAMBDA(const MediumScatterWorkItem &w){
			return;
			const Vector3f& wo = w.wo;
			Sampler sampler	   = &pixelState->sampler[w.pixelId];
			LightSampleContext ctx{w.p, Vector3f::Zero()};
			// [PART-A] Sample direct lighting with ShadowRayTr
			if (enableNEE) { /* TODO */
				SampledLight sampledLight = lightSampler.sample(sampler.get1D());
				Light light	   = sampledLight.light;
				LightSample ls = light.sampleLi(sampler.get1D(), ctx);
			}
			// [PART-B] Sample indirect lighting with scattering function
			PhaseFunctionSample ps = w.phase.sample(wo, sampler.get1D());
			Color thp			   = w.thp * ps.p / ps.pdf;
			// Russian roulette
			float rrProb = min(thp.mean(), 1.f);
			if (sampler.get1D() > rrProb) return;
			thp /= rrProb;

			Ray ray{w.p, ps.wi, w.time, w.medium};
			if (!thp.isZero())
				nextRayQueue(depth)->push(ray, ctx, thp, 1, 1, w.depth + 1, w.pixelId, BSDF_GLOSSY);
	}, gpContext->cudaStream);
}

KRR_NAMESPACE_END