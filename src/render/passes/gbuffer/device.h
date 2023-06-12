#pragma once

#include "common.h"
#include "scene.h"
#include "device/optix.h"

KRR_NAMESPACE_BEGIN

typedef struct {
	rt::SceneData sceneData;
	OptixTraversableHandle traversable;
} LaunchParamsGBuffer;

KRR_NAMESPACE_END