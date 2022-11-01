#pragma once

#include "common.h"
#include "math/utils.h"
#include "render/sampling.h"
#include "device/buffer.h"
#include "taggedptr.h"
#include "raytracing.h"
#include "mesh.h"

KRR_NAMESPACE_BEGIN

using namespace math;

struct ShapeSample {
	Interaction intr;
	float pdf;
};

struct ShapeSampleContext {
	Vector3f p;
	Vector3f n;
};

class Triangle{
public:
	Triangle() = default;

	Triangle(uint triId, MeshData* mesh) :
		primId(triId), mesh(mesh) {}

	KRR_CALLABLE float area()const {
		Vector3i v = mesh->indices[primId];
		Vector3f p0 = mesh->vertices[v[0]].vertex,
			p1 = mesh->vertices[v[1]].vertex,
			p2 = mesh->vertices[v[2]].vertex;
		float s = 0.5f * length(cross(p1 - p0, p2 - p0));
		DCHECK(s > 0);
		return s;
	}

	KRR_CALLABLE float solidAngle(Vector3f p) const {
		Vector3i v = mesh->indices[primId];
		Vector3f p0 = mesh->vertices[v[0]].vertex, 
			p1 = mesh->vertices[v[1]].vertex, 
			p2 = mesh->vertices[v[2]].vertex;

		return utils::sphericalTriangleArea(normalize(p0 - p), 
			normalize(p1 - p),
			normalize(p2 - p));
	}

	KRR_CALLABLE ShapeSample sample(Vector2f u) const {
		// uniform sample on triangle
		ShapeSample ss = {};
		
		Vector3i v = mesh->indices[primId];
		const VertexAttribute& v0 = mesh->vertices[v[0]],
			&v1 = mesh->vertices[v[1]],
			&v2 = mesh->vertices[v[2]];

		Vector3f p0 = v0.vertex, p1 = v1.vertex, p2 = v2.vertex;
		Vector3f b = uniformSampleTriangle(u);
		Vector3f p = b[0] * p0 + b[1] * p1 + b[2] * p2;

		// face normal
		Vector3f n = normalize(Vector3f(cross(p1 - p0, p2 - p0)));
		Vector3f ns = normalize(b[0] * v0.normal + b[1] * v1.normal + b[2] * v2.normal);
		if (dot(n, ns) < 0)
			n *= -1;
		
		Vector2f uv[3] = {
			v0.texcoord,
			v1.texcoord,
			v2.texcoord
		};
		Vector2f uvSample = b[0] * uv[0] + b[1] * uv[1] + b[2] * uv[2];

		ss.intr = Interaction(p, n, uvSample);
		ss.pdf = 1 / area();
		return ss;
	}

	// TODO: Direct sampling on the projeced area to the sphere.
	KRR_CALLABLE ShapeSample sample(Vector2f u, ShapeSampleContext& ctx) const {
		// sample w.r.t. the reference point,
		// also the pdf counted is in solid angle.
		
		Vector3i v = mesh->indices[primId];
		Vector3f p0 = mesh->vertices[v[0]].vertex, 
				p1 = mesh->vertices[v[1]].vertex,
				p2 = mesh->vertices[v[2]].vertex;
		// Use uniform area sampling for numerically unstable cases
		float sr = solidAngle(ctx.p);
		// Sample shape by area and compute incident direction _wi_
		ShapeSample ss = sample(u);
		Vector3f wi = normalize(ss.intr.p - ctx.p);

		// Convert area sampling PDF in _ss_ to solid angle measure
		ss.pdf /= abs(dot(ss.intr.n, wi)) / squaredLength(ctx.p - ss.intr.p);
		if (isinf(ss.pdf)) {
			/* We are sampling the primitive itself ?! */
			ss.pdf = 0;
		}
		return ss;
	}

	KRR_CALLABLE float pdf(Interaction& sample) const {
		return 1 / area();
	}

	KRR_CALLABLE float pdf(Interaction& sample, ShapeSampleContext& ctx) const {
		float sr = solidAngle(ctx.p);
		// Naive version: always return PDF based on uniform area sampling
		Vector3f wi = normalize(sample.p - ctx.p);
		// Compute PDF in solid angle measure from shape intersection point
		float pdf = (1 / area()) / (abs(sample.n.dot(-wi))) /
			squaredLength(ctx.p - sample.p);
		if (isinf(pdf)) pdf = 0;
		return pdf;
	}

	MeshData* getMesh() {
		return mesh;
	}

private:
	friend class Mesh;
	uint primId;
	MeshData* mesh{nullptr};
};


class Shape: public TaggedPointer<Triangle>{
public:
	using TaggedPointer::TaggedPointer;

	KRR_CALLABLE float area()const {
		auto area = [&](auto ptr) ->float {return ptr->area(); };
		return dispatch(area);
	}

	KRR_CALLABLE ShapeSample sample(Vector2f u) const {
		auto sample = [&](auto ptr) ->ShapeSample {return ptr->sample(u); };
		return dispatch(sample);
	}

	KRR_CALLABLE ShapeSample sample(Vector2f u, ShapeSampleContext& ctx) const {
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