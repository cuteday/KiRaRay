#pragma once
#include <optix.h>
#include <optix_stubs.h>

#include "sampler.h"
#include "device/scene.h"
#include "render/lightsampler.h"
#include "render/bsdf.h"
#include "render/media.h"
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
	rt::SceneData sceneData;
	OptixTraversableHandle traversable;
} LaunchParams;

template <typename F>
KRR_CALLABLE Color sampleT_maj(Ray ray, float tMax, Sampler sampler, SampledChannel channel,
							   F callback) {
	tMax *= ray.dir.norm();
	ray.dir.normalize();

	Color T_maj(1);
	RayMajorant majorant = ray.medium.sampleRay(ray, tMax);
	Color sigma_maj		 = majorant.sigma_maj;
	float tMin			 = majorant.tMin;
	while (true) {
		// keep calling the callback function until it requests termination by returning false
		float t = tMin + sampleExponential(sampler.get1D(), sigma_maj[channel]);
		if (t < majorant.tMax) {
			T_maj *= (-(t - tMin) * sigma_maj).exp();
			MediumProperties mp = ray.medium.samplePoint(ray(t));
			if (!callback(ray(t), mp, sigma_maj, T_maj)) break;
			T_maj = 1;
			tMin  = t;
		} else {
			/* Sampled interaction is outside the medium */
			float dt = majorant.tMax - tMin;
			if (isinf(dt)) dt = M_FLOAT_MAX;
			T_maj *= (-dt * majorant.sigma_maj).exp();
			break;
		}
	}
	return T_maj;
}

template <typename TraceFunc>
KRR_CALLABLE void traceTransmittance(ShadowRayWorkItem sr, SurfaceInteraction &intr,
									 PixelStateBuffer *pixelState, TraceFunc trace) {
	SampledChannel channel = pixelState->channel[sr.pixelId];
	Sampler sampler		   = &pixelState->sampler[sr.pixelId];
	Ray ray				   = sr.ray;
	float tMax			   = sr.tMax;
	Vector3f pLight		   = ray(tMax);

	Color T_ray(1);
	Color pu(1), pl(1);

	while (ray.dir.any()) {
		bool visible = trace(ray, tMax);
		if (!visible && intr.material != nullptr) {
			/* Hit opaque surface, goodbye... */
			T_ray = 0;
			break;
		}
		return;
		if (ray.medium) {
			float tEnd = visible ? tMax : (intr.p - ray.origin).norm() / ray.dir.norm();
			Color T_maj =
				sampleT_maj(ray, tEnd, sampler, channel,
					[&](Vector3f p, MediumProperties mp, Color sigma_maj, Color T_maj) {
						Color sigma_n = (sigma_maj - mp.sigma_s - mp.sigma_s).cwiseMax(0);

						// ratio-tracking
						float pr = T_maj[channel] * sigma_maj[channel];
						T_ray *= T_maj * sigma_n / pr;
						pl *= T_maj * sigma_maj / pr;
						pu *= T_maj * sigma_n / pr;

						// Terminate transmittance estimation with Russian roulette
						Color Tr = T_ray / (pu + pl).mean();
						if (Tr.maxCoeff() < 0.05f) {
							if (sampler.get1D() < 0.75f)
								T_ray = 0;
							else
								T_ray /= 0.25f;
						}
						/* This callback returns false for termination in the sampleT_maj
							* loop. */
						return T_ray.any() ? true : false;
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