#pragma once

#include "math/math.h"
#include "sampler.h"

namespace krr{
	namespace shader{
		
		struct Ray {
			vec3f origin;
			vec3f dir;
		};

		struct ShadingFrame{
			vec3f N;
			vec3f T;
			vec3f B;
		};

		struct PathData {
			// info of this path			
			vec3f L = 0;			// total contribution to the current pixel
			vec3f throughput = 1;	//
			// info of the current sampled scattering ray
			Ray ray;
			float pdf;			
			// random sample generator
			LCGSampler sampler;
		};

		struct ShadingData {		// for use as per ray data, generated from ch
			vec3f pos;

			vec3f N;				// shading normal [flipped if back-facing]
			vec3f T;				// tangent
			vec3f B;				// bitangent
			
			vec3f wi;				// view direction
			vec2f uv;				// texture coords

			vec3f geoN;				// geometry normal [on the front-facing side]
	
			vec3f diffuse;			// diffuse reflectance
			vec3f emission;			// for emissive material

			bool miss;				// not valid if missing, or ?
			bool frontFacing;		// 
			uint flags;				// my

			ShadingFrame frame;
			
			__both__ vec3f fromLocal(vec3f v) const{
				return T * v.x + B * v.y + N * v.z;
			}

			__both__ vec3f toLocal(vec3f v) const{
				return { dot(T, v) + dot(B, v) + dot(N, v) };
			}

		};

	}
}