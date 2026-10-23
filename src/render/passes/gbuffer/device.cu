#include "device.h"
#include "render/shared.h"
#include "render/shading.h"
#include <optix_device.h>

NAMESPACE_BEGIN(krr)

extern "C" __constant__ LaunchParameters<GBufferPass> launchParams;

KRR_RT_KERNEL KRR_RT_CH(Primary)() { optixSetPayload_0(__float_as_uint(optixGetRayTmax())); }

KRR_RT_KERNEL KRR_RT_AH(Primary)() {
	const HitInfo hit = getHitInfo();
	if (!hit.instance->mesh->material) return;
	const auto &material = hit.getMaterial();
	if (material.mProgram.enabled) {
		auto context  = materialContext(hit.getMesh(), hit.primitiveId, hit.barycentric,
										getInstanceTransform());
		float opacity = material.mProgram.evaluate(
			context, rt::MaterialEvaluation::Opacity)[MaterialParameter::Opacity][0];
		if (opacity < 0.5f) optixIgnoreIntersection();
	} else if (alphaKilled(hit))
		optixIgnoreIntersection();
}

KRR_RT_KERNEL KRR_RT_MS(Primary)() { optixSetPayload_0(__float_as_uint(M_FLOAT_INF)); }

KRR_RT_KERNEL KRR_RT_RG(Primary)() {
	const Vector3ui index = optixGetLaunchIndex();
	const Vector2i pixel{int(index[0]), int(index[1])};
	const uint32_t offset =
		index[0] + (launchParams.frameSize[1] - 1 - index[1]) * launchParams.frameSize[0];
	const auto &camera = launchParams.cameraData;
	rt::CameraSample sample{Vector2f(camera.externalProjection ? 0.5f : 0.f), Vector2f(0.5f), 0.f};
	Ray ray				 = camera.getRay(pixel, launchParams.frameSize, sample);
	unsigned int payload = __float_as_uint(M_FLOAT_INF);
	optixTrace(launchParams.traversable, ray.origin, ray.dir, 0.f, M_FLOAT_INF, 0.f,
			   OptixVisibilityMask(1), OPTIX_RAY_FLAG_NONE, 0, 1, 0, payload);
	const float distance = __uint_as_float(payload);
	float linear = M_FLOAT_INF, projected = 1.f;
	if (isfinite(distance)) {
		Vector3f point = ray.origin + distance * ray.dir;
		Vector4f position{point[0], point[1], point[2], 1.f};
		linear		  = -(launchParams.view * position)[2];
		Vector4f clip = launchParams.viewProjection * position;
		projected	  = clamp(
			(clip[2] / clip[3] - launchParams.nearClip) / (1.f - launchParams.nearClip), 0.f, 1.f);
	}
	launchParams.linearDepth[offset]	= linear;
	launchParams.projectedDepth[offset] = projected;
}

NAMESPACE_END(krr)
