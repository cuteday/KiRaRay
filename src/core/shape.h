#pragma once

#include "common.h"
#include "util/math_utils.h"
#include "render/sampling.h"
#include "device/buffer.h"
#include "device/taggedptr.h"
#include "raytracing.h"
#include "mesh.h"

KRR_NAMESPACE_BEGIN

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

	Triangle(uint triId, rt::InstanceData *instance) : primId(triId), instance(instance) {}

	KRR_CALLABLE float area()const {
		rt::MeshData *mesh = instance->mesh;
		Vector3i v = mesh->indices[primId];
		Vector3f p0 = mesh->positions[v[0]], p1 = mesh->positions[v[1]],
				 p2 = mesh->positions[v[2]];
		float s = 0.5f * length(cross(p1 - p0, p2 - p0));
		DCHECK(s > 0);
		return s;
	}

	KRR_CALLABLE float solidAngle(Vector3f p) const {
		rt::MeshData *mesh = instance->mesh;
		const Affine3f& transform = instance->transform;
		Vector3i v = mesh->indices[primId];
		Vector3f p0 = transform * mesh->positions[v[0]], 
				 p1 = transform * mesh->positions[v[1]],
				 p2 = transform * mesh->positions[v[2]];

		return utils::sphericalTriangleArea(normalize(p0 - p), 
			normalize(p1 - p),
			normalize(p2 - p));
	}

	KRR_CALLABLE ShapeSample sample(Vector2f u) const {
		// uniform sample on triangle
		ShapeSample ss = {};
		rt::MeshData *mesh = instance->mesh;
		Vector3i v = mesh->indices[primId];
		Vector3f p0 = mesh->positions[v[0]], p1 = mesh->positions[v[1]],
				 p2 = mesh->positions[v[2]];
		Vector3f n0 = mesh->normals[v[0]], n1 = mesh->normals[v[1]],
				 n2 = mesh->normals[v[2]];
		
		Vector3f b = uniformSampleTriangle(u);
		Vector3f p = b[0] * p0 + b[1] * p1 + b[2] * p2;

		// face normal
		Vector3f n = normalize(Vector3f(cross(p1 - p0, p2 - p0)));
		Vector3f ns = normalize(b[0] * n0 + b[1] * n1 + b[2] * n2);
		if (dot(n, ns) < 0)
			n *= -1;
		
		Vector2f uvSample{};
		if (mesh->texcoords.size()) {
			Vector2f uv[3]	  = {mesh->texcoords[v[0]], mesh->texcoords[v[1]],
								 mesh->texcoords[v[2]]};
			uvSample = b[0] * uv[0] + b[1] * uv[1] + b[2] * uv[2];
		}

		// transform to world space [TODO: refactor]
		p		= instance->getTransform() * p;
		n		= instance->getTransposedInverseTransform() * n;
		ss.intr = Interaction(p, n, uvSample);
		ss.pdf	= 1 / area();
		return ss;
	}

	// TODO: Direct sampling on the projeced area to the sphere.
	KRR_CALLABLE ShapeSample sample(Vector2f u, const ShapeSampleContext &ctx) const {
		// sample w.r.t. the reference point,
		// also the pdf counted is in solid angle.
		rt::MeshData *mesh = instance->mesh;
		Vector3i v = mesh->indices[primId];
		Vector3f p0 = mesh->positions[v[0]], p1 = mesh->positions[v[1]],
				 p2 = mesh->positions[v[2]];
		// Use uniform area sampling for numerically unstable cases
		float sr = solidAngle(ctx.p);
		// Sample shape by area and compute incident direction _wi_
		ShapeSample ss = sample(u);
		Vector3f wi	   = normalize(ss.intr.p - ctx.p);

		// Convert area sampling PDF in _ss_ to solid angle measure
		ss.pdf /= fabs(dot(ss.intr.n, wi)) / (ctx.p - ss.intr.p).squaredNorm();
		if (wi.norm() == 0 || isinf(ss.pdf)) 
			/* We are sampling the primitive itself ?! */
			ss.pdf = 0;
		return ss;
	}

	KRR_CALLABLE float pdf(const Interaction &sample) const {
		return 1 / area();
	}

	KRR_CALLABLE float pdf(const Interaction &sample, const ShapeSampleContext &ctx) const {
		float sr = solidAngle(ctx.p);
		// Naive version: always return PDF based on uniform area sampling
		Vector3f wi = normalize(sample.p - ctx.p);
		// Compute PDF in solid angle measure from shape intersection point
		float pdf = (1 / area()) / (fabs(sample.n.dot(-wi)) / (ctx.p - sample.p).squaredNorm());
		if (wi.norm() == 0 || isinf(pdf))
			pdf = 0;
		return pdf;
	}

	rt::InstanceData* getInstance() const { return instance; }
	rt::MeshData *getMesh() const { return instance->mesh; }

private:
	friend class Mesh;
	uint primId;
	rt::InstanceData *instance{nullptr};
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

	KRR_CALLABLE ShapeSample sample(Vector2f u, const ShapeSampleContext &ctx) const {
		auto sample = [&](auto ptr) ->ShapeSample {return ptr->sample(u, ctx); };
		return dispatch(sample);
	}

	KRR_CALLABLE float pdf(const Interaction &sample) const {
		auto pdf = [&](auto ptr) ->float {return ptr->pdf(sample); };
		return dispatch(pdf);
	}

	KRR_CALLABLE float pdf(const Interaction &sample, const ShapeSampleContext &ctx) const {
		auto pdf = [&](auto ptr) ->float {return ptr->pdf(sample, ctx); };
		return dispatch(pdf);
	}
};

KRR_NAMESPACE_END