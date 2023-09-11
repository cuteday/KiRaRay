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
			float tMax = w.tMax * ray.dir.norm();
			ray.dir.normalize();
			Color thp = w.thp;
			Color pu = w.pu, pl = w.pl;
			Sampler sampler		   = &pixelState->sampler[w.pixelId];
			SampledChannel channel = pixelState->channel[w.pixelId];
			Medium medium		   = ray.medium;
			
			// sample corresponding majorant...
			RayMajorant majorant = medium.sampleRay(ray, tMax);
			Color sigma_maj		 = majorant.sigma_maj;
			Color T_maj(1);
			bool scattered{false};
			float tMin = majorant.tMin;

			// [LOOP] Either absorped or scattered after several null collisions
			while (true) {
				float t = tMin + sampleExponential(sampler.get1D(), sigma_maj[channel]);
				if (t < majorant.tMax) {
					/* Sampled interaction is within the medium... */
					T_maj *= (-(t - tMin) * sigma_maj).exp();
					MediumProperties mp = medium.samplePoint(ray(t));
					/* [Rock paper scissors!] Absorp, real scattering, or null-collision? */
					/* [STEP.1] Account for medium emission */
					if (w.depth < maxDepth && mp.Le.any()) {
						float pr = sigma_maj[channel] * T_maj[channel];
						Color pe = pu * sigma_maj * T_maj / pr;
						if (pe.any()) L += thp * mp.sigma_a * T_maj * mp.Le / (pr * pe.mean());
					}
					/* [STEP.2] Sample a type of the three scattering events */
					float pAbsorb	= mp.sigma_a[channel] * sigma_maj[channel];
					float pScatter	= mp.sigma_s[channel] * sigma_maj[channel];
					float pNull		= max(0.f, 1.f - pAbsorb - pScatter);
					float pModes[3] = {pAbsorb, pScatter, pNull};
					//int mode		= sampleDiscrete(pModes, 3, sampler.get1D());
					int mode = sampleDiscrete({pAbsorb, pScatter, pNull}, sampler.get1D());
					if (mode == 0) {		// Absorbed (goodbye)
						thp = 0;			// Will not continue
						break;
					} else if (mode == 1) {	// Real scattering
						float pr = T_maj[channel] * mp.sigma_s[channel];
						thp *= T_maj * mp.sigma_s / pr;
						pu *= T_maj * mp.sigma_s / pr;
						if (thp.any() && pu.any()) 
							mediumScatterQueue->push(ray(t), thp, -ray.dir, ray.time, ray.medium,
													 mp.phase, w.depth, w.pixelId);
						scattered = true;	// Continue on another direction
						break;
					} else {				// Null-collision
						Color sigma_n = (sigma_maj - mp.sigma_a - mp.sigma_s).cwiseMax(0);
						float pr = T_maj[channel] * sigma_n[channel];
						thp *= T_maj * sigma_n / pr;
						if (pr == 0) thp = 0;
						pu *= T_maj * sigma_n / pr;
						pl *= T_maj * sigma_n / pr;
					}
					T_maj = Color::Ones();
					tMin  = t;
				} else {
					/* Sampled interaction is outside the medium (either escaped or on surface.) */
					float dt = majorant.tMax - tMin;
					if (isinf(dt)) dt = M_FLOAT_MAX;
					T_maj *= (-dt * majorant.sigma_maj).exp();
					break;
				}
			}
			// Add any contribution along this volumetric path...
			if (L.any()) pixelState->addRadiance(w.pixelId, L);

			if (!scattered && thp.any()) {
				/* Update statistics... */
				thp *= T_maj / T_maj[channel];
				pu *= T_maj / T_maj[channel];
				pl *= T_maj / T_maj[channel];
			}

			// Return if scattered or no throughput or max depth reached
			if (scattered || thp.isZero() || w.depth == maxDepth) return;

			// [Grand Survival] There are three cases needed to handle...
			// [I an free...] The ray escaped from our sight...
			if (w.tMax == M_FLOAT_INF) {
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