#pragma once

#include "common.h"
#include "math/utils.h"
#include "render/sampling.h"
#include "device/buffer.h"
#include "taggedptr.h"
#include "raytracing.h"
#include "mesh.h"

KRR_NAMESPACE_BEGIN

//class Mesh;
//class MeshData;

struct ShapeSample {
	Interaction intr;
	float pdf;
};

struct ShapeSampleContext {
	vec3f p;
	vec3f n;
};

class Triangle{
public:
	Triangle() = default;

	Triangle(uint triId, MeshData* mesh) :
		primId(triId), mesh(mesh) {}

	KRR_CALLABLE float area()const {
		vec3i v = mesh->indices[primId];
		vec3f p0 = mesh->vertices[v[0]],
			p1 = mesh->vertices[v[1]],
			p2 = mesh->vertices[v[2]];
		float s = 0.5f * length(cross(p1 - p0, p2 - p0));
		assert(s > 0);
		return s;
	}

	KRR_CALLABLE float solidAngle(vec3f p) const {
		vec3i v = mesh->indices[primId];
		vec3f p0 = mesh->vertices[v[0]], p1 = mesh->vertices[v[1]], p2 = mesh->vertices[v[2]];

		return utils::sphericalTriangleArea(normalize(p0 - p), 
			normalize(p1 - p),
			normalize(p2 - p));
	}

	KRR_CALLABLE ShapeSample sample(vec2f u) const {
		// uniform sample on triangle
		ShapeSample ss = {};
		
		vec3i v = mesh->indices[primId];
		vec3f p0 = mesh->vertices[v[0]],
			p1 = mesh->vertices[v[1]],
			p2 = mesh->vertices[v[2]];
		vec3f b = uniformSampleTriangle(u);
		vec3f p = b[0] * p0 + b[1] * p1 + b[2] * p2;

		// face normal
		vec3f n = normalize(vec3f(cross(p1 - p0, p2 - p0)));
		if (mesh->normals) {
			vec3f ns = normalize(b[0] * mesh->normals[v[0]] + b[1] * mesh->normals[v[1]] + b[2] * mesh->normals[v[2]]);
			if (dot(n, ns) < 0)
				n *= -1;
		}

		vec2f uv[3];
		if (mesh->texcoords) {
			uv[0] = mesh->texcoords[v[0]],
			uv[1] = mesh->texcoords[v[1]],
			uv[2] = mesh->texcoords[v[2]];
		}
		else {
			uv[0] = { 0,0 }, uv[1] = { 1,0 }, uv[2] = { 1,1 };
		}

		vec2f uvSample = b[0] * uv[0] + b[1] * uv[1] + b[2] * uv[2];

		ss.intr = Interaction(p, n, uvSample);
		ss.pdf = 1 / area();
		return ss;
	}

	KRR_CALLABLE ShapeSample sample(vec2f u, ShapeSampleContext& ctx) const {
		// sample w.r.t. the reference point,
		// also the pdf counted is in solid angle.
		
		vec3i v = mesh->indices[primId];
		vec3f p0 = mesh->vertices[v[0]], p1 = mesh->vertices[v[1]], p2 = mesh->vertices[v[2]];

		// Use uniform area sampling for numerically unstable cases
		float sr = solidAngle(ctx.p);
		//if (sr < kMinSphericalSampleArea || sr > kMaxSphericalSampleArea) {
			// Sample shape by area and compute incident direction _wi_
			ShapeSample ss = sample(u);
			vec3f wi = normalize(ss.intr.p - ctx.p);

			// Convert area sampling PDF in _ss_ to solid angle measure
			ss.pdf /= abs(dot(ss.intr.n, -wi)) / lengthSquared(ctx.p - ss.intr.p);
			if(isinf(ss.pdf)) ss.pdf = 0;
			return ss;
		//}
	}

	KRR_CALLABLE float pdf(Interaction& sample) const {
		return 1 / area();
	}

	KRR_CALLABLE float pdf(Interaction& sample, ShapeSampleContext& ctx) const {
		float sr = solidAngle(ctx.p);

		// Return PDF based on uniform area sampling for challenging triangles
		//if (sr < kMinSphericalSampleArea || sr > kMaxSphericalSampleArea) {

			vec3f wi = normalize(sample.p - ctx.p);
			// Compute PDF in solid angle measure from shape intersection point
			float pdf = (1 / area()) / (abs(dot(sample.n, -wi)) /
				lengthSquared(ctx.p - sample.p));
			if (isinf(pdf))
				pdf = 0;

			return pdf;
		//}
	}

	MeshData* getMesh() {
		return mesh;
	}

private:

	friend class Mesh;
	uint primId;
	MeshData* mesh{nullptr};
	static constexpr float kMinSphericalSampleArea = 3e-4;
	static constexpr float kMaxSphericalSampleArea = 6.22;
};


class Shape: public TaggedPointer<Triangle>{
public:
	using TaggedPointer::TaggedPointer;

	KRR_CALLABLE float area()const {
		auto area = [&](auto ptr) ->float {return ptr->area(); };
		return dispatch(area);
	}

	KRR_CALLABLE ShapeSample sample(vec2f u) const {
		auto sample = [&](auto ptr) ->ShapeSample {return ptr->sample(u); };
		return dispatch(sample);
	}

	KRR_CALLABLE ShapeSample sample(vec2f u, ShapeSampleContext& ctx) const {
		auto sample = [&](auto ptr) ->ShapeSample {return ptr->sample(u, ctx); };
		return dispatch(sample);
	}

	KRR_CALLABLE float pdf( Interaction& sample) const {
		auto pdf = [&](auto ptr) ->float {return ptr->pdf(sample); };
		return dispatch(pdf);
	}

	KRR_CALLABLE float pdf(Interaction& sample, ShapeSampleContext& ctx) const {
		auto pdf = [&](auto ptr) ->float {return ptr->pdf(sample, ctx); };
		return dispatch(pdf);
	}
};

KRR_NAMESPACE_END