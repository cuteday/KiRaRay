#pragma once

#include "common.h"

#include "device/memory.h"
#include "device/buffer.h"

KRR_NAMESPACE_BEGIN

class Triangle;
class DiffuseAreaLight;

enum class VertexAttributeType {
	Position,
	Normal,
	Texcoord,
	Tangent,
	Bitangent,
	Count
};

struct VertexAttribute {
	Vector3f vertex;
	Vector3f normal;
	Vector2f texcoord;
	Vector3f tangent;
	Vector3f bitangent;
};

namespace rt {
struct MeshData {
	TypedBuffer<VertexAttribute> vertices;
	TypedBuffer<Vector3i> indices;
	TypedBuffer<Triangle> primitives;
	TypedBuffer<DiffuseAreaLight> lights;
	uint materialId;
};
}

class Mesh {
public:
	std::vector<Triangle> createTriangles(rt::MeshData* mesh) const;
	
	std::vector<VertexAttribute> vertices;
	std::vector<Vector3i> indices;

	AABB getAABB() const;

	uint materialId{};
	Color Le{};		/* A mesh-specific area light, used when importing pbrt formats. */
};


KRR_NAMESPACE_END