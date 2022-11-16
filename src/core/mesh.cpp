#pragma once

#include "mesh.h"
#include "shape.h"

KRR_NAMESPACE_BEGIN

std::vector<Triangle> Mesh::createTriangles(MeshData* mesh) const {
	uint nTriangles = indices.size();
	std::vector<Triangle> triangles;
	for (uint i = 0; i < nTriangles; i++) {
		triangles.push_back(Triangle(i, mesh));
	}
	return triangles;
}

AABB Mesh::getAABB() const {
	AABB aabb;
	for (const VertexAttribute &v : vertices) {
		aabb.extend(v.vertex);
	}
	return aabb;
}

KRR_NAMESPACE_END