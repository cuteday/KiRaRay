#pragma once

#include "common.h"
#include "math/math.h"
#include "device/memory.h"
#include "device/buffer.h"

KRR_NAMESPACE_BEGIN

class Triangle;
class DiffuseAreaLight;

struct VertexAttribute {
	Vector3f vertex;
	Vector3f normal;
	Vector2f texcoord;
	Vector3f tangent;
	Vector3f bitangent;
};

struct MeshData {
	VertexAttribute *vertices{ nullptr };
	Vector3i *indices{ nullptr };

	uint materialId{ 0 };
	DiffuseAreaLight* lights{ nullptr };
};

class Mesh {
public:

	void toDevice() {
		mData.vertices = (VertexAttribute *) vertices.data();
		mData.indices = (Vector3i*)indices.data();
		mData.materialId = materialId;
		mData.lights = lights.data();
	}

	std::vector<Triangle> createTriangles(MeshData* mesh) const;
	
	inter::vector<VertexAttribute> vertices;
	inter::vector<Vector3i> indices;

	inter::vector<Triangle> emissiveTriangles;
	inter::vector<DiffuseAreaLight> lights;

	AABB getAABB() const;

	uint materialId{};
	MeshData mData;
	Color Le{};
};

KRR_NAMESPACE_END