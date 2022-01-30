#pragma once

#include "math/math.h"
#include "camera.h"
#include "envmap.h"

namespace krr {

	struct LaunchParams
	{
		uint       frameID{ 0 };
		vec4f* colorBuffer;
		vec2i     fbSize;

		uint maxDepth;

		Camera camera;
		EnvLight envLight;

		OptixTraversableHandle traversable;
	};

}