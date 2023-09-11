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
			Ray ray	   = w.ray;
			float tMax = w.tMax;
			Color thp  = w.thp;
			Color pu = w.pu, pl = w.pl;
			Sampler sampler = &pixelState->sampler[w.pixelId];
			Medium medium	= ray.medium;

			// sample corresponding majorant...
			Color T_maj = medium.sampleRay(ray, tMax).sigma_maj;
			bool scattered{false};

			// [LOOP] Either absorped or scattered after several null collisions
			while (false) {

			}

			// Return if scattered or no throughput or max depth reached
			if (scattered || thp.isZero() || w.depth == maxDepth) return;

			// [Grand Survival] There are three cases needed to handle...
			// [I an free...] The ray escaped from our sight...
			if (w.tMax == M_FLOAT_MAX) {
				/* [CHECK] Is it correct to use w.bsdfType here? */
				missRayQueue->push(ray, w.ctx, thp, pu, pl, w.bsdfType, w.depth, w.pixelId);
				return;
			}

			// [Moon landing] We finally reached the surface...!
			if (w.intr.light) {
				// Handle contribution for light hit
				/* The light is sampled from the last vertex on surface, so use its context! */
				hitLightRayQueue->push(w.intr, w.ctx, BSDF_SMOOTH, w.depth, w.pixelId, thp,
									   pu, pl);
			}

			// [Tomorrow is another day] Next surface scattering event...!
			scatterRayQueue->push(w.intr, thp, w.depth, w.pixelId);

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
			if (enableNEE) {
				SampledLight sampledLight = lightSampler.sample(sampler.get1D());
				Light light				  = sampledLight.light;
				LightSample ls			  = light.sampleLi(sampler.get1D(), ctx);
				Ray shadowRay			  = Interaction(ls.intr.p, w.time, w.medium).spawnRayTo(ls.intr);
				Vector3f wi				  = shadowRay.dir.normalized(); 
				float lightPdf			  = sampledLight.pdf * ls.pdf;
				float phasePdf			  = light.isDeltaLight() ? 0 : w.phase.pdf(wo, wi);
				
				Color Ld = w.thp * ls.L;
				if (Ld.any() && ls.pdf > 0) {
					ShadowRayWorkItem sw = {};
					sw.ray				 = shadowRay;
					sw.pl				 = lightPdf;
					sw.pu				 = phasePdf;
					sw.Ld				 = Ld;
					sw.pixelId			 = w.pixelId;
					sw.tMax				 = 1;
					shadowRayQueue->push(sw);
				}
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
				nextRayQueue(depth)->push(ray, ctx, thp, 1, 1 / ps.pdf, w.depth + 1, w.pixelId, BSDF_GLOSSY);
	}, gpContext->cudaStream);
}

KRR_NAMESPACE_END