#pragma once

#include "math/math.h"
#include "camera.h"

namespace krr {

	struct MeshSBTData {
		vec3f* vertices;
		vec3i* indices;
		vec3f* normals;
	};

	struct LaunchParams
	{
		uint       frameID{ 0 };
		vec4f* colorBuffer;
		vec2i     fbSize;

		Camera camera;

		OptixTraversableHandle traversable;
	};

}