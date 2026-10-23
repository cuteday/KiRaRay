#pragma once

#include "mesh.h"
#include "material/program.h"

NAMESPACE_BEGIN(krr)

KRR_CALLABLE MaterialValue materialValue(const Vector3f &value) {
	return {value[0], value[1], value[2]};
}
KRR_CALLABLE Vector3f materialVector(MaterialValue value) { return {value[0], value[1], value[2]}; }

KRR_CALLABLE MaterialContext materialContext(const rt::MeshData &mesh, uint primitive,
											 Vector3f barycentric,
											 const Transformation &transform) {
	Vector3i vertices = mesh.indices[primitive];
	Vector3f e1		  = mesh.positions[vertices[1]] - mesh.positions[vertices[0]];
	Vector3f e2		  = mesh.positions[vertices[2]] - mesh.positions[vertices[0]];
	Vector3f normal	  = cross(e1, e2);
	if (mesh.normals.size())
		normal = barycentric[0] * mesh.normals[vertices[0]] +
				 barycentric[1] * mesh.normals[vertices[1]] +
				 barycentric[2] * mesh.normals[vertices[2]];
	normal			 = normalize(transform.transposedInverse() * normal);
	Vector3f tangent = utils::getPerpendicular(normal), bitangent = cross(normal, tangent);
	Vector2f uv(0);
	if (mesh.texcoords.size()) {
		Vector2f t0 = mesh.texcoords[vertices[0]], t1 = mesh.texcoords[vertices[1]],
				 t2 = mesh.texcoords[vertices[2]];
		uv			= barycentric[0] * t0 + barycentric[1] * t1 + barycentric[2] * t2;
		Vector2f d1 = t1 - t0, d2 = t2 - t0;
		float determinant = d1[0] * d2[1] - d1[1] * d2[0];
		if (fabsf(determinant) > 1e-12f) {
			Vector3f t = transform.rotation() * ((e1 * d2[1] - e2 * d1[1]) / determinant);
			Vector3f b = transform.rotation() * ((e2 * d1[0] - e1 * d2[0]) / determinant);
			if (mesh.tangents.size())
				t = transform.rotation() * (barycentric[0] * mesh.tangents[vertices[0]] +
					barycentric[1] * mesh.tangents[vertices[1]] + barycentric[2] * mesh.tangents[vertices[2]]);
			t -= normal * dot(t, normal);
			if (t.squaredNorm() > 1e-12f) {
				tangent	  = normalize(t);
				bitangent = cross(normal, tangent);
				if (dot(bitangent, b) < 0) bitangent = -bitangent;
			}
		}
	}
	MaterialContext context;
	context.uv		  = {uv[0], uv[1], 0};
	context.normal	  = materialValue(normal);
	context.tangent	  = materialValue(tangent);
	context.bitangent = materialValue(bitangent);
	return context;
}

NAMESPACE_END(krr)
