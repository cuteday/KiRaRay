#pragma once
#include <optix.h>
#include <optix_stubs.h>

#include "sampler.h"
#include "device/scene.h"
#include "render/bsdf.h"
#include "render/media.h"
#include "render/spectrum.h"
#include "render/lightsampler.h"
#include "workqueue.h"

KRR_NAMESPACE_BEGIN

class PixelStateBuffer;

using namespace shader;

typedef struct {
	RayQueue* currentRayQueue;
	RayQueue* nextRayQueue;
	ShadowRayQueue* shadowRayQueue;
	MissRayQueue* missRayQueue;
	HitLightRayQueue* hitLightRayQueue;
	ScatterRayQueue* scatterRayQueue;
	MediumSampleQueue* mediumSampleQueue;
	
	PixelStateBuffer* pixelState;
	const RGBColorSpace *colorSpace;
	rt::SceneData sceneData;
	OptixTraversableHandle traversable;
} LaunchParams;

template <typename F>
KRR_HOST_DEVICE Spectrum sampleT_maj(Ray ray, float tMax, Sampler sampler,
											SampledWavelengths lambda, F callback) {
	/* This function returns the [remaining](not told callback before hitting surface) 
		part of the transmittance. */
	tMax *= ray.dir.norm();
	ray.dir.normalize();

	Spectrum T_maj(1);
	MajorantIterator iter = ray.medium.sampleRay(ray, tMax, lambda);
	int channel			  = lambda.mainIndex();

	while (true) {
		gpu::optional<MajorantSegment> seg = iter.next();
		if (!seg) return T_maj;
		// skipping empty space...
		if (seg->sigma_maj[channel] == 0) {
			float dt = seg->tMax - seg->tMin;
			if (isinf(dt)) dt = M_FLOAT_MAX;
			T_maj *= (-dt * seg->sigma_maj).exp();
			continue;
		}

		float tMin = seg->tMin;
		while (true) {
			// keep calling the callback function until it requests termination by returning false
			// for example, several null-collision followed by a real scattering event...
			float t = tMin + sampleExponential(sampler.get1D(), seg->sigma_maj[channel]);
			if (t < seg->tMax) {
				T_maj *= (-(t - tMin) * seg->sigma_maj).exp();
				MediumProperties mp = ray.medium.samplePoint(ray(t), lambda);
				if (!callback(ray(t), mp, seg->sigma_maj, T_maj)) return Spectrum::Ones();
				T_maj = Spectrum::Ones();
				tMin  = t;
			} else {
				/* Sampled interaction is outside the medium */
				float dt = seg->tMax - tMin;
				if (isinf(dt)) dt = M_FLOAT_MAX;
				T_maj *= (-dt * seg->sigma_maj).exp();
				break;
			}
		}
	}
	return Spectrum::Ones();
}

template <typename TraceFunc>
KRR_HOST_DEVICE void traceTransmittance(ShadowRayWorkItem sr, const SurfaceInteraction& intr,
									 PixelStateBuffer *pixelState, TraceFunc trace) {
	SampledWavelengths lambda = pixelState->lambda[sr.pixelId];
	int channel				  = lambda.mainIndex();
	Sampler sampler			  = &pixelState->sampler[sr.pixelId];
	Ray ray					  = sr.ray;
	float tMax				  = sr.tMax;
	Vector3f pLight			  = ray(tMax);

	Spectrum T_ray(1);
	Spectrum pu(1), pl(1);

	while (ray.dir.any()) {
		bool visible = trace(ray, tMax);
		if (!visible && intr.material != nullptr) {
			/* Hit opaque surface, goodbye... */
			T_ray = Spectrum::Zero();
			break;
		}
		if (ray.medium) {
			float tEnd = visible ? tMax : (intr.p - ray.origin).norm() / ray.dir.norm();
			Spectrum T_maj =
				sampleT_maj(ray, tEnd, sampler, lambda,
					[&](Vector3f p, MediumProperties mp, Spectrum sigma_maj, Spectrum T_maj) {
						Spectrum sigma_n = (sigma_maj - mp.sigma_a - mp.sigma_s).cwiseMax(0);

						// ratio-tracking
						float pr = T_maj[channel] * sigma_maj[channel];
						T_ray *= T_maj * sigma_n / pr;
						pl *= T_maj * sigma_maj / pr;
						pu *= T_maj * sigma_n / pr;

						// Terminate transmittance estimation with Russian roulette
						Spectrum Tr = T_ray / (pu + pl).mean();
						if (Tr.maxCoeff() < 0.05f) {
							if (sampler.get1D() < 0.75f)
								T_ray = Spectrum::Zero();
							else
								T_ray /= 0.25f;
						}
						/* This callback returns false for termination in the sampleT_maj
							* loop. */
						return T_ray.any();
					});

			T_ray *= T_maj / T_maj[channel];
			pu *= T_maj / T_maj[channel];
			pl *= T_maj / T_maj[channel];
		}

		// Light is visible or throughput is zero...
		if (visible || T_ray.isZero()) break;
		// Across a surface with null-material, continuing
		ray = intr.spawnRayTo(pLight);
	}
	if (T_ray.any()) 
		pixelState->addRadiance(sr.pixelId, sr.Ld * T_ray / (sr.pu * pu + sr.pl * pl).mean());
}

KRR_NAMESPACE_END