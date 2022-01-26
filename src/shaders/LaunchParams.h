#pragma once

#include "math/math.h"
#include "camera.h"

namespace krr {

	struct LaunchParams
	{
		int       frameID{ 0 };
		vec4f* colorBuffer;
		vec2i     fbSize;

		Camera camera;

		OptixTraversableHandle traversable;
	};

}