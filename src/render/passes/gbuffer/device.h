#pragma once

#include "common.h"
#include "device/scene.h"
#include "device/optix.h"

KRR_NAMESPACE_BEGIN

class GBufferPass;

template <> 
struct LaunchParameters<GBufferPass> {
	Vector2i frameSize;
	size_t frameIndex;

	rt::SceneData sceneData;
	rt::CameraData cameraData;
	OptixTraversableHandle traversable;
};

KRR_NAMESPACE_END