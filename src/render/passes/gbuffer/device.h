#pragma once

#include "common.h"
#include "device/scene.h"
#include "device/optix.h"

NAMESPACE_BEGIN(krr)

class GBufferPass;

template <> struct LaunchParameters<GBufferPass> {
	Vector2i frameSize;
	rt::CameraData cameraData;
	Matrix4f view;
	Matrix4f viewProjection;
	float nearClip{-1.f};
	float *linearDepth{};
	float *projectedDepth{};
	OptixTraversableHandle traversable;
};

NAMESPACE_END(krr)
