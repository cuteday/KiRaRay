#pragma once

#include "common.h" 
#include "math/math.h"

NAMESPACE_KRR_BEGIN

using namespace math;

class Camera{
public:
	Camera():
	from({0, 0, 0}),
	at({0, 0, 1}),
	up({0, 1, 0}){}

	vec3f from;
	vec3f at;
	vec3f up;
};

NAMESPACE_KRR_END