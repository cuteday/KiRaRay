#include <cuda.h>
#include <cuda_runtime.h>

#include "device/cuda.h"
#include "integrator.h"
#include "guideditem.h"
#include "render/media.h"
#include "render/profiler/profiler.h"
#include "render/wavefront/wavefront.h"

KRR_NAMESPACE_BEGIN

void ZeroGuidingPT::sampleMediumInteraction(int depth) {
	PROFILE("Sample medium interaction");
	ForAllQueued(
		mediumSampleQueue, maxQueueSize,
		KRR_DEVICE_LAMBDA(const MediumSampleWorkItem &w) {
			/* Each ray is either absorped or scattered after some null collisions... */
			Spectrum L(0);
			Ray ray	  = w.ray;
			Spectrum thp = w.thp;
			Spectrum pu = w.pu, pl = w.pl;
			Sampler sampler			  = &pixelState->sampler[w.pixelId];
			SampledWavelengths lambda = pixelState->lambda[w.pixelId];
			int channel				  = lambda.mainIndex();
			Medium medium			  = ray.medium;
			bool scattered{false};

			Spectrum T_maj = sampleT_maj(ray, w.tMax, sampler, lambda, 
				[&](Vector3f p, MediumProperties mp, Spectrum sigma_maj,
					Spectrum T_maj) -> bool {
					if (w.depth < maxDepth && mp.Le.any()) {
						float pr = sigma_maj[channel] * T_maj[channel];
						Spectrum pe = pu * sigma_maj * T_maj / pr;
						if (pe.any()) L += thp * mp.sigma_a * T_maj * mp.Le / (pr * pe.mean());
					}
					/* [STEP.2] Sample a type of the three scattering events */
					float pAbsorb	= mp.sigma_a[channel] / sigma_maj[channel];
					float pScatter	= mp.sigma_s[channel] / sigma_maj[channel];
					float pNull		= max(0.f, 1.f - pAbsorb - pScatter);
					int mode = sampleDiscrete({pAbsorb, pScatter, pNull}, sampler.get1D());
					if (mode == 0) {					// Absorbed (goodbye)
						thp = Spectrum::Zero();	// Will not continue
						return false;
					} else if (mode == 1) {	// Real scattering
						float pr = T_maj[channel] * mp.sigma_s[channel];
						thp *= T_maj * mp.sigma_s / pr;
						pu *= T_maj * mp.sigma_s / pr;
						if (thp.any() && pu.any())
							mediumScatterQueue->push(p, thp, pu, -ray.dir, ray.time, ray.medium,
													 mp.phase, w.depth, w.pixelId);
						scattered = true;	// Continue on another direction
						return false;
					} else {				// Null-collision
						Spectrum sigma_n = (sigma_maj - mp.sigma_a - mp.sigma_s).cwiseMax(0);
						float pr = T_maj[channel] * sigma_n[channel];
						thp *= T_maj * sigma_n / pr;
						if (pr == 0) thp = Spectrum::Zero();
						pu *= T_maj * sigma_n / pr;
						pl *= T_maj * sigma_maj / pr;
						return thp.any() && pu.any();
					} 
				});

			// Add any contribution along this volumetric path...
			if (L.any()) pixelState->addRadiance(w.pixelId, L);

			if (!scattered && thp.any()) {
				/* Update statistics... */
				thp *= T_maj / T_maj[channel];
				pu *= T_maj / T_maj[channel];
				pl *= T_maj / T_maj[channel];
			}

			// Return if scattered or no throughput or max depth reached
			if (scattered || !thp.any() || !pu.any() || w.depth == maxDepth) return;

			// [Grand Survival] There are three cases needed to handle...
			// [I am free...] The ray escaped from our sight...
			if (w.tMax == M_FLOAT_INF) {
				/* [CHECK] Is it correct to use w.bsdfType here? */
				missRayQueue->push(ray, w.ctx, thp, pu, pl, w.bsdfType, w.depth, w.pixelId);
				return;
			}

			// [You can not see me] The surface do not pocess a material (usually an interface?)
			if (!w.intr.material) {	
				/* Just let it go (use *argument* _depth_ here (not w.depth). ) */
				nextRayQueue(depth)->push(w.intr.spawnRayTowards(ray.dir), w.ctx, thp, pu, pl,
										  w.depth, w.pixelId, w.bsdfType);
				return;
			}

			// [Moon landing] We finally reached the surface...!
			if (w.intr.light) {
				// Handle contribution for light hit
				/* The light is sampled from the last vertex on surface, so use its context! */
				hitLightRayQueue->push(w.intr, w.ctx, w.bsdfType, w.depth, w.pixelId, thp,
									   pu, pl);
			}

			// [Tomorrow is another day] Next surface scattering event...!
			scatterRayQueue->push(w.intr, thp, pu, w.depth, w.pixelId);	
	}, gpContext->cudaStream);
}

void ZeroGuidingPT::sampleMediumScattering(int depth) {
	PROFILE("Sample medium scattering");
	ForAllQueued(
		mediumScatterQueue, maxQueueSize,
		KRR_DEVICE_LAMBDA(const MediumScatterWorkItem &w) {
			const Vector3f &wo		  = w.wo;
			Sampler sampler			  = &pixelState->sampler[w.pixelId];
			SampledWavelengths lambda = pixelState->lambda[w.pixelId];
			LightSampleContext ctx{w.p, Vector3f::Zero()};
			// [PART-A] Sample direct lighting with ShadowRayTr
			if (enableNEE) {
				SampledLight sampledLight = lightSampler.sample(sampler.get1D());
				Light light				  = sampledLight.light;
				LightSample ls			  = light.sampleLi(sampler.get2D(), ctx, lambda);
				Ray shadowRay			  = Interaction(w.p, w.time, w.medium).spawnRayTo(ls.intr);
				Vector3f wi				  = shadowRay.dir.normalized();
				Spectrum thp		  = w.thp * w.phase.p(wo, wi);
				float lightPdf			  = sampledLight.pdf * ls.pdf;
				float phasePdf			  = light.isDeltaLight() ? 0 : w.phase.pdf(wo, wi);
				
				Spectrum Ld = thp * ls.L;
				if (Ld.any() && lightPdf > 0) {
					ShadowRayWorkItem sw = {};
					sw.ray				 = shadowRay;
					sw.pl				 = w.pu * lightPdf;
					sw.pu				 = w.pu * phasePdf;
					sw.Ld				 = Ld;
					sw.pixelId			 = w.pixelId;
					sw.tMax				 = 1;
					shadowRayQueue->push(sw);
				}
			}
			
			// [PART-B] Sample indirect lighting with scattering function
			PhaseFunctionSample ps = w.phase.sample(wo, sampler.get2D());
			Spectrum thp	   = w.thp * ps.p / ps.pdf;
			// Russian roulette
			float rrProb = (thp / w.pu.mean()).maxCoeff();
			if (w.depth >= 1 && rrProb < 1) {
				if (w.depth >= 1 && sampler.get1D() >= rrProb) return;
				thp /= rrProb;
			}
			
			Ray ray{w.p, ps.wi, w.time, w.medium};
			if (!thp.isZero() && !thp.hasNaN()) 
				/* [NOTE] We need to multiply P_path by w.pu. While the PDF of P_light and P_bsdf is 
					the same until this vertex, the channel-wise PDF along the path may be different. */
				nextRayQueue(depth)->push(ray, ctx, thp, w.pu, w.pu / ps.pdf, w.depth + 1, w.pixelId, BSDF_SMOOTH);
	}, gpContext->cudaStream);
}

KRR_NAMESPACE_END