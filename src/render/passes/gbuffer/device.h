#pragma once

#include "common.h"
#include "device/scene.h"
#include "device/optix.h"

NAMESPACE_BEGIN(krr)

class GBufferPass;

template <> 
struct LaunchParameters<GBufferPass> {
	Vector2i frameSize;
	size_t frameIndex;

	rt::SceneData sceneData;
	rt::CameraData cameraData;
	OptixTraversableHandle traversable;
};

NAMESPACE_END(krr)