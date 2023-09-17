#pragma once

#include "common.h"
#include "device/scene.h"
#include "device/optix.h"

KRR_NAMESPACE_BEGIN

typedef struct {
	Vector2i frameSize;
	size_t frameIndex;

	rt::SceneData sceneData;
	Camera::CameraData cameraData;
	OptixTraversableHandle traversable;
} LaunchParamsGBuffer;

KRR_NAMESPACE_END