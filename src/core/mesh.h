#pragma once

#include "common.h"

#include "device/memory.h"
#include "device/buffer.h"

KRR_NAMESPACE_BEGIN

class Triangle;
class DiffuseAreaLight;

enum class VertexAttribute {
	Position,
	Normal,
	Texcoord,
	Tangent,
	Count
};

namespace rt {
struct MeshData {
	TypedBuffer<Vector3f> positions;
	TypedBuffer<Vector3f> normals;
	TypedBuffer<Vector2f> texcoords;
	TypedBuffer<Vector3f> tangents;
	TypedBuffer<Vector3i> indices;
	TypedBuffer<Triangle> primitives;
	TypedBuffer<DiffuseAreaLight> lights;
	uint materialId;
};
}

class Mesh {
public:
	using SharedPtr = std::shared_ptr<Mesh>;
	std::vector<Triangle> createTriangles(rt::MeshData* mesh) const;
	
	std::vector<Vector3f> positions;
	std::vector<Vector3f> normals;
	std::vector<Vector2f> texcoords;
	std::vector<Vector3f> tangents;
	std::vector<Vector3i> indices;

	AABB getAABB() const;

	uint materialId{};
	Color Le{};		/* A mesh-specific area light, used when importing pbrt formats. */
};

KRR_NAMESPACE_END