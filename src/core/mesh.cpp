#pragma once

#include "mesh.h"
#include "shape.h"

KRR_NAMESPACE_BEGIN

std::vector<Triangle> Mesh::createTriangles(MeshData* mesh)
{
	uint nTriangles = indices.size();
	std::vector<Triangle> triangles;
	for (uint i = 0; i < nTriangles; i++) {
		triangles.push_back(Triangle(i, mesh));
	}
	return triangles;
}

KRR_NAMESPACE_END